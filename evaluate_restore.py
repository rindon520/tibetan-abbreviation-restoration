import argparse
import json
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean, median

from train_restore import (
    generate_text,
    levenshtein,
    load_model_bundle,
    read_parallel_data,
    split_pairs,
)


def tibetan_units(text):
    separators = set("་།༎༏༐༑༔ \t\r\n.")
    units = []
    current = []
    for ch in text:
        if ch in separators:
            if current:
                units.append("".join(current))
                current = []
            if ch.strip():
                units.append(ch)
        else:
            current.append(ch)
    if current:
        units.append("".join(current))
    return units


def edit_operation_counts(prediction, reference):
    rows = len(prediction)
    cols = len(reference)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    op = [[None] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, cols + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i, pred_ch in enumerate(prediction, start=1):
        for j, ref_ch in enumerate(reference, start=1):
            if pred_ch == ref_ch:
                best = (dp[i - 1][j - 1], "M")
            else:
                best = (dp[i - 1][j - 1] + 1, "S")
            for candidate in ((dp[i - 1][j] + 1, "D"), (dp[i][j - 1] + 1, "I")):
                if candidate[0] < best[0]:
                    best = candidate
            dp[i][j] = best[0]
            op[i][j] = best[1]

    substitutions = insertions = deletions = 0
    substitution_pairs = Counter()
    i, j = rows, cols
    while i > 0 or j > 0:
        current = op[i][j]
        if current == "M":
            i -= 1
            j -= 1
        elif current == "S":
            substitutions += 1
            substitution_pairs[(prediction[i - 1], reference[j - 1])] += 1
            i -= 1
            j -= 1
        elif current == "D":
            deletions += 1
            i -= 1
        elif current == "I":
            insertions += 1
            j -= 1
        else:
            break
    return substitutions, insertions, deletions, substitution_pairs


def multiset_prf(prediction, reference):
    pred_counter = Counter(prediction)
    ref_counter = Counter(reference)
    overlap = sum((pred_counter & ref_counter).values())
    precision = overlap / max(1, sum(pred_counter.values()))
    recall = overlap / max(1, sum(ref_counter.values()))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_metrics(rows):
    if not rows:
        raise ValueError("No rows to evaluate.")

    exact = 0
    total_distance = 0
    total_reference_chars = 0
    total_prediction_chars = 0
    total_unit_distance = 0
    total_reference_units = 0
    substitutions = insertions = deletions = 0
    substitution_pairs = Counter()
    macro_cer = []
    macro_unit_error_rate = []
    normalized_similarity = []
    char_precision = []
    char_recall = []
    char_f1 = []
    length_abs_errors = []
    length_squared_errors = []

    for row in rows:
        prediction = row["prediction"]
        reference = row["reference"]
        distance = levenshtein(prediction, reference)
        exact += int(prediction == reference)
        total_distance += distance
        total_reference_chars += len(reference)
        total_prediction_chars += len(prediction)
        macro_cer.append(distance / max(1, len(reference)))
        normalized_similarity.append(1 - distance / max(1, max(len(prediction), len(reference))))

        pred_units = tibetan_units(prediction)
        ref_units = tibetan_units(reference)
        unit_distance = levenshtein(pred_units, ref_units)
        total_unit_distance += unit_distance
        total_reference_units += len(ref_units)
        macro_unit_error_rate.append(unit_distance / max(1, len(ref_units)))

        sub, ins, dele, pairs = edit_operation_counts(prediction, reference)
        substitutions += sub
        insertions += ins
        deletions += dele
        substitution_pairs.update(pairs)

        precision, recall, f1 = multiset_prf(prediction, reference)
        char_precision.append(precision)
        char_recall.append(recall)
        char_f1.append(f1)

        length_error = abs(len(prediction) - len(reference))
        length_abs_errors.append(length_error)
        length_squared_errors.append(length_error * length_error)

    samples = len(rows)
    exact_match = exact / samples
    micro_cer = total_distance / max(1, total_reference_chars)
    metrics = {
        "samples": samples,
        "exact_match": exact_match,
        "line_error_rate": 1 - exact_match,
        "micro_cer": micro_cer,
        "micro_character_accuracy": 1 - micro_cer,
        "macro_cer_mean": mean(macro_cer),
        "macro_cer_median": median(macro_cer),
        "micro_tibetan_unit_error_rate": total_unit_distance / max(1, total_reference_units),
        "macro_tibetan_unit_error_rate_mean": mean(macro_unit_error_rate),
        "normalized_edit_similarity_mean": mean(normalized_similarity),
        "char_overlap_precision_mean": mean(char_precision),
        "char_overlap_recall_mean": mean(char_recall),
        "char_overlap_f1_mean": mean(char_f1),
        "prediction_reference_length_ratio": total_prediction_chars / max(1, total_reference_chars),
        "length_mae": mean(length_abs_errors),
        "length_rmse": sqrt(mean(length_squared_errors)),
        "edit_substitutions": substitutions,
        "edit_insertions": insertions,
        "edit_deletions": deletions,
        "top_substitutions": [
            {"prediction": pred, "reference": ref, "count": count}
            for (pred, ref), count in substitution_pairs.most_common(20)
        ],
    }
    return metrics


def build_prediction_rows(checkpoint, data_dir, split, output_file=None, device="auto", max_len=128):
    data_dir = Path(data_dir)
    pairs = read_parallel_data(data_dir / "Source.txt", data_dir / "Restore.txt")
    train, valid, test = split_pairs(pairs, valid_ratio=0.1, test_ratio=0.05, seed=42)
    splits = {"train": train, "valid": valid, "test": test}
    selected = splits[split]

    resolved_device = None if device == "auto" else device
    model, tokenizer, _, resolved_device = load_model_bundle(checkpoint, device=resolved_device)
    rows = []
    for source, reference in selected:
        prediction = generate_text(model, tokenizer, source, resolved_device, max_len=max_len)
        rows.append({"source": source, "reference": reference, "prediction": prediction})

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tibetan restoration predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_prediction_rows(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        output_file=args.predictions_jsonl,
        device=args.device,
        max_len=args.max_len,
    )
    metrics = compute_metrics(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
