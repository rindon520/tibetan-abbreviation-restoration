import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_restore import compute_metrics
from train_restore import read_parallel_data, split_pairs


def align_pair(source, reference, mapping=None, max_output_length=8):
    source_len = len(source)
    reference_len = len(reference)
    inf = 10**9
    dp = [[(inf, None) for _ in range(reference_len + 1)] for __ in range(source_len + 1)]
    dp[0][0] = (0.0, None)

    for source_index, source_char in enumerate(source):
        for ref_index in range(reference_len + 1):
            base_cost, _ = dp[source_index][ref_index]
            if base_cost >= inf:
                continue
            max_len = min(max_output_length, reference_len - ref_index)
            for length in range(max_len + 1):
                output = reference[ref_index : ref_index + length]
                cost = mapping_cost(source_char, output, mapping)
                new_cost = base_cost + cost
                next_cell = dp[source_index + 1][ref_index + length]
                if new_cost < next_cell[0]:
                    dp[source_index + 1][ref_index + length] = (
                        new_cost,
                        (ref_index, output),
                    )

    if dp[source_len][reference_len][0] >= inf:
        return []

    aligned = []
    source_index = source_len
    ref_index = reference_len
    while source_index > 0:
        previous_ref_index, output = dp[source_index][ref_index][1]
        aligned.append((source[source_index - 1], output))
        ref_index = previous_ref_index
        source_index -= 1
    return list(reversed(aligned))


def mapping_cost(source_char, output, mapping=None):
    if mapping and source_char in mapping:
        learned = mapping[source_char]
        cost = 0.0 if output == learned else 2.0 + abs(len(output) - len(learned))
        if output == source_char:
            cost = min(cost, 1.0)
    else:
        cost = 0.0 if output == source_char else 1.0 + abs(len(output) - 1) * 0.25
    if not output and source_char not in " \t\r\n":
        cost += 1.5
    return cost


def learn_char_mapping(pairs, iterations=5, max_output_length=8):
    mapping = None
    counts = {}
    for _ in range(iterations):
        counts = defaultdict(Counter)
        for source, reference in pairs:
            for source_char, output in align_pair(
                source,
                reference,
                mapping=mapping,
                max_output_length=max_output_length,
            ):
                counts[source_char][output] += 1
        mapping = {
            source_char: outputs.most_common(1)[0][0]
            for source_char, outputs in counts.items()
        }
    return mapping or {}, counts


def apply_mapping(source, mapping):
    return "".join(mapping.get(char, char) for char in source)


def evaluate_mapping(pairs, mapping):
    rows = [
        {
            "source": source,
            "reference": reference,
            "prediction": apply_mapping(source, mapping),
        }
        for source, reference in pairs
    ]
    return rows, compute_metrics(rows)


def serializable_counts(counts, top_k=5):
    payload = {}
    for source_char, outputs in counts.items():
        payload[source_char] = [
            {"output": output, "count": count}
            for output, count in outputs.most_common(top_k)
        ]
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an explainable character transducer for Tibetan restoration."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tibetan_transducer"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-output-length", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = read_parallel_data(args.data_dir / "Source.txt", args.data_dir / "Restore.txt")
    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs, valid_ratio=0.1, test_ratio=0.05, seed=42
    )
    mapping, counts = learn_char_mapping(
        train_pairs,
        iterations=args.iterations,
        max_output_length=args.max_output_length,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "model_type": "char_transducer",
        "iterations": args.iterations,
        "max_output_length": args.max_output_length,
        "mapping": mapping,
        "counts_top": serializable_counts(counts),
        "data": {
            "pairs": len(pairs),
            "train": len(train_pairs),
            "valid": len(valid_pairs),
            "test": len(test_pairs),
        },
    }
    (args.output_dir / "mapping.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_metrics = {}
    for split_name, split_pairs_ in (
        ("train", train_pairs),
        ("valid", valid_pairs),
        ("test", test_pairs),
    ):
        rows, metrics = evaluate_mapping(split_pairs_, mapping)
        all_metrics[split_name] = metrics
        (args.output_dir / f"{split_name}_predictions.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
