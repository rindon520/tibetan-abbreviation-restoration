from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluate_restore import compute_metrics
from tibetan_syllable import apply_syllable_vocabulary_correction, build_syllable_vocabulary
from train_restore import read_parallel_data, split_pairs
from train_syllable_corrector import terminal_shad


PAD_CHAR = "<pad>"
UNK_CHAR = "<unk>"
COPY_LABEL = "<copy>"
DELETE_LABEL = "<delete>"
PAD_LABEL = "<pad>"
BOS = "<B>"
EOS = "<E>"


def align_edit_outputs(source: str, reference: str) -> list[str]:
    """Align a target substring to every source character with unit edit costs."""
    rows = len(source)
    cols = len(reference)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        dp[row][0] = row
    for col in range(cols + 1):
        dp[0][col] = col

    for row in range(1, rows + 1):
        source_char = source[row - 1]
        for col in range(1, cols + 1):
            substitution = dp[row - 1][col - 1] + int(
                source_char != reference[col - 1]
            )
            dp[row][col] = min(
                substitution,
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
            )

    operations: list[tuple[str, str]] = []
    row = rows
    col = cols
    while row or col:
        if row and col:
            substitution = dp[row - 1][col - 1] + int(
                source[row - 1] != reference[col - 1]
            )
            if dp[row][col] == substitution:
                operations.append(("source", reference[col - 1]))
                row -= 1
                col -= 1
                continue
        if row and dp[row][col] == dp[row - 1][col] + 1:
            operations.append(("source", ""))
            row -= 1
            continue
        operations.append(("insert", reference[col - 1]))
        col -= 1

    operations.reverse()
    outputs: list[str] = []
    prefix = ""
    for operation, output in operations:
        if operation == "insert":
            if outputs:
                outputs[-1] += output
            else:
                prefix += output
            continue
        outputs.append(prefix + output)
        prefix = ""
    if prefix and outputs:
        outputs[-1] += prefix
    return outputs


def output_to_label(source_char: str, output: str) -> str:
    if output == source_char:
        return COPY_LABEL
    if output == "":
        return DELETE_LABEL
    return output


def label_to_output(source_char: str, label: str) -> str:
    if label == COPY_LABEL:
        return source_char
    if label in {DELETE_LABEL, PAD_LABEL}:
        return ""
    return label


def build_labelled_pairs(pairs):
    labelled = []
    label_counts = Counter()
    for source, reference in pairs:
        outputs = align_edit_outputs(source, reference)
        if len(outputs) != len(source):
            raise RuntimeError("Alignment length does not match the source length.")
        labels = [
            output_to_label(source_char, output)
            for source_char, output in zip(source, outputs)
        ]
        label_counts.update(labels)
        labelled.append((source, reference, labels))
    return labelled, label_counts


def build_rule_counts(labelled_pairs):
    global_counts = defaultdict(Counter)
    context_counts = defaultdict(Counter)
    for source, _, labels in labelled_pairs:
        for index, (source_char, label) in enumerate(zip(source, labels)):
            output = label_to_output(source_char, label)
            previous = source[index - 1] if index else BOS
            following = source[index + 1] if index + 1 < len(source) else EOS
            global_counts[source_char][output] += 1
            for key in (
                ("tri", previous, source_char, following),
                ("left", previous, source_char),
                ("right", source_char, following),
            ):
                context_counts[key][output] += 1
    return global_counts, context_counts


def choose_rule_output(counter, min_count=3, min_confidence=0.5):
    if not counter:
        return None
    output, count = counter.most_common(1)[0]
    total = sum(counter.values())
    if total < min_count or count / total < min_confidence:
        return None
    return output


def rule_output_at(
    source,
    index,
    global_counts,
    context_counts,
    min_count=3,
    min_confidence=0.5,
):
    source_char = source[index]
    previous = source[index - 1] if index else BOS
    following = source[index + 1] if index + 1 < len(source) else EOS
    for key in (
        ("tri", previous, source_char, following),
        ("left", previous, source_char),
        ("right", source_char, following),
    ):
        output = choose_rule_output(
            context_counts.get(key),
            min_count=min_count,
            min_confidence=min_confidence,
        )
        if output is not None:
            return output
    output = choose_rule_output(global_counts.get(source_char), 1, 0.0)
    return source_char if output is None else output


def require_torch():
    import torch
    from torch import nn

    return torch, nn


@dataclass
class EditTaggerConfig:
    char_vocab_size: int
    label_vocab_size: int
    pad_char_id: int
    pad_label_id: int
    d_model: int = 256
    layers: int = 4
    heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.15
    max_positions: int = 128


def create_model(config: EditTaggerConfig):
    torch, nn = require_torch()

    class EditTagger(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.pad_char_id = cfg.pad_char_id
            self.scale = math.sqrt(cfg.d_model)
            self.embedding = nn.Embedding(
                cfg.char_vocab_size, cfg.d_model, padding_idx=cfg.pad_char_id
            )
            self.position = nn.Embedding(cfg.max_positions, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=cfg.layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(cfg.d_model)
            self.output = nn.Linear(cfg.d_model, cfg.label_vocab_size)

        def forward(self, chars):
            positions = torch.arange(chars.size(1), device=chars.device).unsqueeze(0)
            hidden = self.embedding(chars) * self.scale + self.position(positions)
            hidden = self.encoder(
                hidden, src_key_padding_mask=chars.eq(self.pad_char_id)
            )
            return self.output(self.norm(hidden))

    return EditTagger(config)


class EditDataset:
    def __init__(self, labelled_pairs, char_to_id, label_to_id, max_length):
        self.items = []
        unknown = char_to_id[UNK_CHAR]
        for source, reference, labels in labelled_pairs:
            source = source[:max_length]
            labels = labels[:max_length]
            self.items.append(
                (
                    source,
                    reference,
                    [char_to_id.get(char, unknown) for char in source],
                    [
                        label_to_id.get(label, label_to_id[COPY_LABEL])
                        for label in labels
                    ],
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_collate(pad_char_id, pad_label_id):
    torch, _ = require_torch()

    def collate(batch):
        width = max(len(item[2]) for item in batch)
        chars = torch.full((len(batch), width), pad_char_id, dtype=torch.long)
        labels = torch.full((len(batch), width), pad_label_id, dtype=torch.long)
        sources = []
        references = []
        for row, (source, reference, char_ids, label_ids) in enumerate(batch):
            chars[row, : len(char_ids)] = torch.tensor(char_ids)
            labels[row, : len(label_ids)] = torch.tensor(label_ids)
            sources.append(source)
            references.append(reference)
        return chars, labels, sources, references

    return collate


def decode_batches(
    model,
    loader,
    id_to_label,
    device,
    global_counts,
    context_counts,
    confidence_threshold,
    rule_min_count,
    rule_min_confidence,
):
    torch, _ = require_torch()
    model.eval()
    rows = []
    with torch.no_grad():
        for chars, _, sources, references in loader:
            chars = chars.to(device)
            probabilities = model(chars).softmax(dim=-1)
            confidence, label_ids = probabilities.max(dim=-1)
            for row_index, (source, reference) in enumerate(zip(sources, references)):
                pieces = []
                decisions = []
                for index, source_char in enumerate(source):
                    label = id_to_label[int(label_ids[row_index, index].item())]
                    score = float(confidence[row_index, index].item())
                    neural_output = label_to_output(source_char, label)
                    if score < confidence_threshold:
                        output = rule_output_at(
                            source,
                            index,
                            global_counts,
                            context_counts,
                            min_count=rule_min_count,
                            min_confidence=rule_min_confidence,
                        )
                        origin = "rule_fallback"
                    else:
                        output = neural_output
                        origin = "neural"
                    pieces.append(output)
                    decisions.append(
                        {
                            "index": index,
                            "source": source_char,
                            "output": output,
                            "confidence": score,
                            "origin": origin,
                        }
                    )
                rows.append(
                    {
                        "source": source,
                        "reference": reference,
                        "raw_prediction": "".join(pieces),
                        "decisions": decisions,
                    }
                )
    return rows


def postprocess_rows(rows, vocab, max_edit_distance):
    result = []
    for row in rows:
        shad_prediction = terminal_shad(row["raw_prediction"])
        prediction = apply_syllable_vocabulary_correction(
            shad_prediction,
            vocab,
            max_edit_distance=max_edit_distance,
            max_syllable_chars=8,
        )
        result.append(
            row
            | {
                "shad_prediction": shad_prediction,
                "prediction": prediction,
            }
        )
    return result


def train_one_epoch(model, loader, optimizer, scaler, device, pad_label_id, amp):
    torch, _ = require_torch()
    import torch.nn.functional as functional

    model.train()
    total_loss = 0.0
    total_tokens = 0
    for chars, labels, _, _ in loader:
        chars = chars.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            logits = model(chars)
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_label_id,
                label_smoothing=0.03,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tokens = int(labels.ne(pad_label_id).sum().item())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    return total_loss / max(1, total_tokens)


def save_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a non-autoregressive context edit tagger for Tibetan restoration."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_segments_trusted"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tibetan_edit_tagger"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rule-min-count", type=int, default=3)
    parser.add_argument("--rule-min-confidence", type=float, default=0.5)
    parser.add_argument("--confidence-thresholds", default="0.0,0.4,0.5,0.6,0.7,0.8")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    random.seed(args.seed)
    torch, _ = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    pairs = read_parallel_data(
        args.data_dir / "Source.txt", args.data_dir / "Restore.txt"
    )
    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs, valid_ratio=0.1, test_ratio=0.05, seed=args.seed
    )
    train_labelled, label_counts = build_labelled_pairs(train_pairs)
    valid_labelled, _ = build_labelled_pairs(valid_pairs)
    test_labelled, _ = build_labelled_pairs(test_pairs)

    chars = sorted({char for source, _ in train_pairs for char in source})
    char_vocab = [PAD_CHAR, UNK_CHAR] + chars
    char_to_id = {char: index for index, char in enumerate(char_vocab)}
    labels = [PAD_LABEL, COPY_LABEL, DELETE_LABEL] + sorted(
        label
        for label in label_counts
        if label not in {PAD_LABEL, COPY_LABEL, DELETE_LABEL}
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    global_counts, context_counts = build_rule_counts(train_labelled)
    vocab = build_syllable_vocabulary(train_pairs)

    config = EditTaggerConfig(
        char_vocab_size=len(char_vocab),
        label_vocab_size=len(labels),
        pad_char_id=char_to_id[PAD_CHAR],
        pad_label_id=label_to_id[PAD_LABEL],
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_positions=args.max_length,
    )
    model = create_model(config).to(device)

    from torch.utils.data import DataLoader

    collate = make_collate(config.pad_char_id, config.pad_label_id)
    train_loader = DataLoader(
        EditDataset(train_labelled, char_to_id, label_to_id, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        EditDataset(valid_labelled, char_to_id, label_to_id, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    test_loader = DataLoader(
        EditDataset(test_labelled, char_to_id, label_to_id, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_cer = float("inf")
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best.pt"
    id_to_label = labels

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config.pad_label_id,
            args.amp,
        )
        raw_rows = decode_batches(
            model,
            valid_loader,
            id_to_label,
            device,
            global_counts,
            context_counts,
            confidence_threshold=0.0,
            rule_min_count=args.rule_min_count,
            rule_min_confidence=args.rule_min_confidence,
        )
        metrics = compute_metrics(postprocess_rows(raw_rows, vocab, 1))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "valid_exact_match": metrics["exact_match"],
                "valid_micro_cer": metrics["micro_cer"],
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if metrics["micro_cer"] < best_cer - 1e-6:
            best_cer = metrics["micro_cer"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "char_vocab": char_vocab,
                    "labels": labels,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    thresholds = [
        float(value.strip())
        for value in args.confidence_thresholds.split(",")
        if value.strip()
    ]
    tuning = []
    best = None
    for threshold in thresholds:
        raw_rows = decode_batches(
            model,
            valid_loader,
            id_to_label,
            device,
            global_counts,
            context_counts,
            confidence_threshold=threshold,
            rule_min_count=args.rule_min_count,
            rule_min_confidence=args.rule_min_confidence,
        )
        for edit_distance in (0, 1):
            rows = postprocess_rows(raw_rows, vocab, edit_distance)
            metrics = compute_metrics(rows)
            record = {
                "confidence_threshold": threshold,
                "max_edit_distance": edit_distance,
                "exact_match": metrics["exact_match"],
                "micro_cer": metrics["micro_cer"],
            }
            tuning.append(record)
            key = (round(metrics["micro_cer"], 12), -metrics["exact_match"])
            if best is None or key < best[0]:
                best = (key, threshold, edit_distance)

    if best is None:
        raise RuntimeError("No validation setting was evaluated.")

    final_metrics = {}
    for split_name, loader in (("valid", valid_loader), ("test", test_loader)):
        raw_rows = decode_batches(
            model,
            loader,
            id_to_label,
            device,
            global_counts,
            context_counts,
            confidence_threshold=best[1],
            rule_min_count=args.rule_min_count,
            rule_min_confidence=args.rule_min_confidence,
        )
        rows = postprocess_rows(raw_rows, vocab, best[2])
        final_metrics[split_name] = compute_metrics(rows)
        save_jsonl(args.output_dir / f"{split_name}_predictions.jsonl", rows)

    payload = {
        "model_type": "non_autoregressive_context_edit_tagger",
        "train_samples": len(train_pairs),
        "valid_samples": len(valid_pairs),
        "test_samples": len(test_pairs),
        "device": str(device),
        "best_epoch": checkpoint["epoch"],
        "best_confidence_threshold": best[1],
        "best_max_edit_distance": best[2],
        "config": asdict(config),
        "label_count": len(labels),
        "label_frequencies": label_counts.most_common(),
        "tuning": tuning,
        "metrics": final_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
