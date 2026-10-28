import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def configure_utf8_stdout(stream=sys.stdout):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def read_parallel_data(source_path, restore_path):
    source_path = Path(source_path)
    restore_path = Path(restore_path)
    source_lines = [
        line.lstrip("\ufeff") for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    restore_lines = [
        line.lstrip("\ufeff") for line in restore_path.read_text(encoding="utf-8").splitlines()
    ]

    if len(source_lines) != len(restore_lines):
        raise ValueError(
            f"Line count mismatch: {source_path} has {len(source_lines)} lines, "
            f"{restore_path} has {len(restore_lines)} lines."
        )

    pairs = list(zip(source_lines, restore_lines))
    if not pairs:
        raise ValueError("No training pairs found.")
    return pairs


def build_vocab(vocab_path, pairs: Iterable[tuple[str, str]] | None = None):
    tokens = []
    seen = set()

    def add(token):
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

    for token in SPECIAL_TOKENS:
        add(token)

    vocab_path = Path(vocab_path)
    if vocab_path.exists():
        for line in vocab_path.read_text(encoding="utf-8").splitlines():
            token = line.strip("\ufeff")
            if not token:
                continue
            if len(token) == 1:
                add(token)
            for ch in token:
                add(ch)

    if pairs is not None:
        for source, restore in pairs:
            for ch in source:
                add(ch)
            for ch in restore:
                add(ch)

    stoi = {token: idx for idx, token in enumerate(tokens)}
    return stoi, tokens


class CharTokenizer:
    def __init__(self, stoi, itos=None):
        self.stoi = dict(stoi)
        if itos is None:
            size = max(self.stoi.values()) + 1
            self.itos = ["<unk>"] * size
            for token, idx in self.stoi.items():
                self.itos[idx] = token
        else:
            self.itos = list(itos)

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        skip = {self.pad_id, self.bos_id, self.eos_id}
        pieces = []
        for idx in ids:
            idx = int(idx)
            if idx in skip:
                continue
            if 0 <= idx < len(self.itos):
                token = self.itos[idx]
                pieces.append("" if token in SPECIAL_TOKENS else token)
            else:
                pieces.append("")
        return "".join(pieces)


def split_pairs(pairs, valid_ratio=0.1, test_ratio=0.05, seed=42):
    if valid_ratio < 0 or test_ratio < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio and test_ratio must be non-negative and sum to < 1.")

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = int(round(total * test_ratio))
    valid_count = int(round(total * valid_ratio))

    if total >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    if total >= 3 and valid_ratio > 0:
        valid_count = max(1, valid_count)
    if test_count + valid_count >= total:
        overflow = test_count + valid_count - total + 1
        valid_count = max(0, valid_count - overflow)

    test = shuffled[:test_count]
    valid = shuffled[test_count : test_count + valid_count]
    train = shuffled[test_count + valid_count :]
    return train, valid, test


def clamp_eos(ids, max_len, eos_id):
    if len(ids) <= max_len:
        return ids
    clipped = ids[:max_len]
    clipped[-1] = eos_id
    return clipped


def require_torch():
    import torch
    from torch import nn

    return torch, nn


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    d_model: int = 256
    layers: int = 4
    heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_positions: int = 256


def _config_get(config, key):
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def create_model(config):
    torch, nn = require_torch()

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout, max_len):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)].to(dtype=x.dtype)
            return self.dropout(x)

    class RestoreTransformer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            vocab_size = _config_get(cfg, "vocab_size")
            pad_id = _config_get(cfg, "pad_id")
            d_model = _config_get(cfg, "d_model")
            dropout = _config_get(cfg, "dropout")
            max_positions = _config_get(cfg, "max_positions")
            self.pad_id = pad_id
            self.scale = math.sqrt(d_model)
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
            self.position = PositionalEncoding(d_model, dropout, max_positions)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=_config_get(cfg, "heads"),
                dim_feedforward=_config_get(cfg, "ff_dim"),
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=_config_get(cfg, "heads"),
                dim_feedforward=_config_get(cfg, "ff_dim"),
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            try:
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=_config_get(cfg, "layers"),
                    enable_nested_tensor=False,
                )
            except TypeError:
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=_config_get(cfg, "layers")
                )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=_config_get(cfg, "layers")
            )
            self.output = nn.Linear(d_model, vocab_size)

        def _causal_mask(self, length, device):
            return torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=device),
                diagonal=1,
            )

        def encode(self, src, src_pad_mask):
            src_emb = self.position(self.embedding(src) * self.scale)
            return self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        def decode(self, memory, src_pad_mask, tgt):
            tgt_pad_mask = tgt.eq(self.pad_id)
            tgt_emb = self.position(self.embedding(tgt) * self.scale)
            tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
            return self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )

        def forward(self, src, tgt_in):
            src_pad_mask = src.eq(self.pad_id)
            tgt_pad_mask = tgt_in.eq(self.pad_id)
            tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)
            src_emb = self.position(self.embedding(src) * self.scale)
            tgt_emb = self.position(self.embedding(tgt_in) * self.scale)
            memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            return self.output(out)

    return RestoreTransformer(config)


class PairDataset:
    def __init__(self, pairs, tokenizer, max_src_len=128, max_tgt_len=128):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, restore = self.pairs[idx]
        src = self.tokenizer.encode(source, add_eos=True)
        tgt_out = self.tokenizer.encode(restore, add_eos=True)
        src = clamp_eos(src, self.max_src_len, self.tokenizer.eos_id)
        tgt_out = clamp_eos(tgt_out, self.max_tgt_len, self.tokenizer.eos_id)
        tgt_in = [self.tokenizer.bos_id] + tgt_out[:-1]
        return src, tgt_in, tgt_out


def pad_sequences(sequences, pad_id):
    torch, _ = require_torch()
    width = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def make_collate_fn(pad_id):
    def collate(batch):
        src, tgt_in, tgt_out = zip(*batch)
        return (
            pad_sequences(src, pad_id),
            pad_sequences(tgt_in, pad_id),
            pad_sequences(tgt_out, pad_id),
        )

    return collate


def set_seed(seed):
    random.seed(seed)
    try:
        torch, _ = require_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def move_batch(batch, device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, pad_id, amp, grad_clip):
    torch, _ = require_torch()
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_tokens = 0
    use_amp = amp and device.type == "cuda"

    for batch in loader:
        src, tgt_in, tgt_out = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_id,
            )
        token_count = int(tgt_out.ne(pad_id).sum().item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(1, total_tokens)


def evaluate_loss(model, loader, device, pad_id):
    torch, _ = require_torch()
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            src, tgt_in, tgt_out = move_batch(batch, device)
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_id,
            )
            token_count = int(tgt_out.ne(pad_id).sum().item())
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
    return total_loss / max(1, total_tokens)


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def generate_text(model, tokenizer, text, device, max_len=128):
    torch, _ = require_torch()
    model.eval()
    src_ids = tokenizer.encode(text, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(tokenizer.pad_id)
    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        memory = model.encode(src, src_pad)
        for _ in range(max_len):
            dec = model.decode(memory, src_pad, ys)
            logits = model.output(dec[:, -1])
            next_id = int(logits.argmax(dim=-1).item())
            ys = torch.cat(
                [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == tokenizer.eos_id:
                break
    return tokenizer.decode(ys[0].tolist())


def evaluate_generation(model, tokenizer, pairs, device, max_len=128, sample_limit=128):
    if not pairs:
        return {"exact_match": 0.0, "cer": 0.0, "samples": 0}

    subset = list(pairs[:sample_limit])
    exact = 0
    distance = 0
    total_chars = 0
    for source, target in subset:
        pred = generate_text(model, tokenizer, source, device, max_len=max_len)
        exact += int(pred == target)
        distance += levenshtein(pred, target)
        total_chars += max(1, len(target))

    return {
        "exact_match": exact / len(subset),
        "cer": distance / total_chars,
        "samples": len(subset),
    }


def save_checkpoint(path, model, tokenizer, config, args, epoch, metrics):
    torch, _ = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config_dict,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_model_bundle(checkpoint_path, device=None):
    torch, _ = require_torch()
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    tokenizer = CharTokenizer(checkpoint["stoi"], checkpoint["itos"])
    model_config = checkpoint.get("model_config", checkpoint.get("config"))
    if model_config is None:
        raise ValueError("Checkpoint does not contain a compatible model configuration.")
    config = ModelConfig(**model_config)
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, config, device


def make_scheduler(optimizer, warmup_steps, total_steps):
    torch, _ = require_torch()
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step):
        step += 1
        if step <= warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Tibetan abbreviation restoration model."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tibetan_restore"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-src-len", type=int, default=128)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--generation-every", type=int, default=5)
    parser.add_argument("--generation-samples", type=int, default=128)
    return parser.parse_args()


def main():
    configure_utf8_stdout()
    args = parse_args()
    torch, _ = require_torch()
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    source_path = args.data_dir / "Source.txt"
    restore_path = args.data_dir / "Restore.txt"
    vocab_path = args.data_dir / "tibetan_VCC_7387.txt"
    pairs = read_parallel_data(source_path, restore_path)
    if args.max_train_samples > 0:
        pairs = pairs[: args.max_train_samples]

    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    stoi, itos = build_vocab(vocab_path, train_pairs)
    tokenizer = CharTokenizer(stoi, itos)

    max_positions = max(args.max_src_len, args.max_tgt_len) + 4
    config = ModelConfig(
        vocab_size=len(itos),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_positions=max_positions,
    )
    model = create_model(config).to(device)

    collate_fn = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(
        PairDataset(train_pairs, tokenizer, args.max_src_len, args.max_tgt_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        PairDataset(valid_pairs, tokenizer, args.max_src_len, args.max_tgt_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        PairDataset(test_pairs, tokenizer, args.max_src_len, args.max_tgt_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "pairs": len(pairs),
        "train": len(train_pairs),
        "valid": len(valid_pairs),
        "test": len(test_pairs),
        "vocab_size": len(itos),
        "device": str(device),
        "config": asdict(config),
        "args": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, ensure_ascii=False, indent=2))

    best_valid = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            tokenizer.pad_id,
            args.amp,
            args.grad_clip,
        )
        valid_loss = evaluate_loss(model, valid_loader, device, tokenizer.pad_id)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        if epoch == 1 or epoch % args.generation_every == 0 or epoch == args.epochs:
            metrics.update(
                {
                    "valid_generation": evaluate_generation(
                        model,
                        tokenizer,
                        valid_pairs,
                        device,
                        max_len=args.max_tgt_len,
                        sample_limit=args.generation_samples,
                    )
                }
            )

        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        save_checkpoint(
            args.output_dir / "last.pt", model, tokenizer, config, args, epoch, metrics
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            save_checkpoint(
                args.output_dir / "best.pt", model, tokenizer, config, args, epoch, metrics
            )

        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    best_model, best_tokenizer, _, best_device = load_model_bundle(
        args.output_dir / "best.pt", device=str(device)
    )
    test_loss = evaluate_loss(best_model, test_loader, best_device, best_tokenizer.pad_id)
    test_generation = evaluate_generation(
        best_model,
        best_tokenizer,
        test_pairs,
        best_device,
        max_len=args.max_tgt_len,
        sample_limit=len(test_pairs),
    )
    final_metrics = {"test_loss": test_loss, "test_generation": test_generation}
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
