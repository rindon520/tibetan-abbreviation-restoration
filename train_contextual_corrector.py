import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_restore import compute_metrics
from tibetan_syllable import apply_syllable_vocabulary_correction, build_syllable_vocabulary
from train_restore import read_parallel_data, split_pairs
from train_syllable_corrector import terminal_shad
from train_transducer import align_pair, learn_char_mapping, serializable_counts
from model_artifacts import serialize_contextual_state


BOS = "<B>"
EOS = "<E>"


def context_keys(source, index):
    char = source[index]
    prev_char = source[index - 1] if index > 0 else BOS
    next_char = source[index + 1] if index + 1 < len(source) else EOS
    return [
        ("tri", prev_char, char, next_char),
        ("left", prev_char, char),
        ("right", char, next_char),
    ]


def build_context_counts(pairs, base_mapping, max_output_length=8):
    counts = defaultdict(Counter)
    for source, reference in pairs:
        aligned = align_pair(
            source,
            reference,
            mapping=base_mapping,
            max_output_length=max_output_length,
        )
        for index, (_, output) in enumerate(aligned):
            for key in context_keys(source, index):
                counts[key][output] += 1
    return dict(counts)


def choose_context_output(counter, min_count=2, min_confidence=0.5):
    if not counter:
        return None
    output, count = counter.most_common(1)[0]
    total = sum(counter.values())
    if total < min_count:
        return None
    if count / total < min_confidence:
        return None
    return output


def apply_contextual_mapping(
    source,
    base_mapping,
    context_counts,
    min_count=2,
    min_confidence=0.5,
):
    pieces = []
    for index, char in enumerate(source):
        output = None
        for key in context_keys(source, index):
            output = choose_context_output(
                context_counts.get(key),
                min_count=min_count,
                min_confidence=min_confidence,
            )
            if output is not None:
                break
        pieces.append(output if output is not None else base_mapping.get(char, char))
    return "".join(pieces)


def predict_rows(
    pairs,
    base_mapping,
    context_counts,
    vocab,
    min_count,
    min_confidence,
    max_edit_distance,
    max_syllable_chars,
):
    rows = []
    for source, reference in pairs:
        raw_prediction = apply_contextual_mapping(
            source,
            base_mapping,
            context_counts,
            min_count=min_count,
            min_confidence=min_confidence,
        )
        base_prediction = terminal_shad(raw_prediction)
        prediction = apply_syllable_vocabulary_correction(
            base_prediction,
            vocab,
            max_edit_distance=max_edit_distance,
            max_syllable_chars=max_syllable_chars,
        )
        rows.append(
            {
                "source": source,
                "reference": reference,
                "raw_prediction": raw_prediction,
                "base_prediction": base_prediction,
                "prediction": prediction,
            }
        )
    return rows


def evaluate_setting(
    pairs,
    base_mapping,
    context_counts,
    vocab,
    min_count,
    min_confidence,
    max_edit_distance,
    max_syllable_chars,
):
    rows = predict_rows(
        pairs,
        base_mapping,
        context_counts,
        vocab,
        min_count=min_count,
        min_confidence=min_confidence,
        max_edit_distance=max_edit_distance,
        max_syllable_chars=max_syllable_chars,
    )
    return rows, compute_metrics(rows)


def tuning_key(metrics, min_count):
    return (
        round(metrics["micro_cer"], 12),
        -min_count,
        -metrics["exact_match"],
    )


def serialize_context_counts(context_counts, top_k=3):
    payload = []
    for key, outputs in sorted(context_counts.items(), key=lambda item: repr(item[0])):
        payload.append(
            {
                "key": list(key),
                "outputs": [
                    {"output": output, "count": count}
                    for output, count in outputs.most_common(top_k)
                ],
                "total": sum(outputs.values()),
            }
        )
    return payload


def serialize_vocab(vocab, top_k=200):
    counts = vocab["counts"]
    delimiters = vocab["delimiters"]
    return {
        "total": vocab["total"],
        "unique_cores": len(counts),
        "top_cores": [
            {
                "core": core,
                "count": count,
                "delimiter": delimiters[core].most_common(1)[0][0],
            }
            for core, count in counts.most_common(top_k)
        ],
    }


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a context-aware transducer with syllable-vocabulary correction "
            "for Tibetan abbreviation restoration."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_segments_trusted"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/tibetan_contextual_corrector_segments"),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-output-length", type=int, default=8)
    parser.add_argument("--max-syllable-chars", type=int, default=8)
    parser.add_argument("--min-counts", default="1,2,3,5")
    parser.add_argument("--min-confidences", default="0.5,0.6,0.7")
    parser.add_argument("--max-edit-distances", default="0,1")
    parser.add_argument("--skip-train-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = read_parallel_data(args.data_dir / "Source.txt", args.data_dir / "Restore.txt")
    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs,
        valid_ratio=0.1,
        test_ratio=0.05,
        seed=42,
    )
    print(
        json.dumps(
            {
                "stage": "data",
                "pairs": len(pairs),
                "train": len(train_pairs),
                "valid": len(valid_pairs),
                "test": len(test_pairs),
            },
            ensure_ascii=False,
        )
    )

    base_mapping, base_counts = learn_char_mapping(
        train_pairs,
        iterations=args.iterations,
        max_output_length=args.max_output_length,
    )
    context_counts = build_context_counts(
        train_pairs,
        base_mapping,
        max_output_length=args.max_output_length,
    )
    vocab = build_syllable_vocabulary(train_pairs)
    print(
        json.dumps(
            {
                "stage": "trained_context_model",
                "base_mapping_size": len(base_mapping),
                "context_key_count": len(context_counts),
                "syllable_vocab_size": len(vocab["counts"]),
            },
            ensure_ascii=False,
        )
    )

    tuning = []
    best = None
    for min_count in parse_int_list(args.min_counts):
        for min_confidence in parse_float_list(args.min_confidences):
            for max_edit_distance in parse_int_list(args.max_edit_distances):
                valid_rows, valid_metrics = evaluate_setting(
                    valid_pairs,
                    base_mapping,
                    context_counts,
                    vocab,
                    min_count=min_count,
                    min_confidence=min_confidence,
                    max_edit_distance=max_edit_distance,
                    max_syllable_chars=args.max_syllable_chars,
                )
                item = {
                    "min_count": min_count,
                    "min_confidence": min_confidence,
                    "max_edit_distance": max_edit_distance,
                    "valid_exact_match": valid_metrics["exact_match"],
                    "valid_micro_cer": valid_metrics["micro_cer"],
                    "valid_character_accuracy": valid_metrics["micro_character_accuracy"],
                }
                tuning.append(item)
                print(json.dumps({"stage": "tune", **item}, ensure_ascii=False))
                key = tuning_key(valid_metrics, min_count=min_count)
                if best is None or key < best["key"]:
                    best = {
                        "key": key,
                        "min_count": min_count,
                        "min_confidence": min_confidence,
                        "max_edit_distance": max_edit_distance,
                        "valid_rows": valid_rows,
                        "valid_metrics": valid_metrics,
                    }

    if best is None:
        raise ValueError("No tuning settings were provided.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "runtime_state": serialize_contextual_state(base_mapping, context_counts, vocab),
        "model_type": "contextual_char_transducer_plus_syllable_vocabulary_corrector",
        "iterations": args.iterations,
        "max_output_length": args.max_output_length,
        "max_syllable_chars": args.max_syllable_chars,
        "best_min_count": best["min_count"],
        "best_min_confidence": best["min_confidence"],
        "best_max_edit_distance": best["max_edit_distance"],
        "tuning": tuning,
        "base_mapping": base_mapping,
        "base_counts_top": serializable_counts(base_counts),
        "context_counts_top": serialize_context_counts(context_counts),
        "syllable_vocabulary": serialize_vocab(vocab),
        "data": {
            "pairs": len(pairs),
            "train": len(train_pairs),
            "valid": len(valid_pairs),
            "test": len(test_pairs),
        },
    }
    (args.output_dir / "model.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_metrics = {}
    evaluation_splits = [("valid", valid_pairs), ("test", test_pairs)]
    if not args.skip_train_predictions:
        evaluation_splits.append(("train", train_pairs))

    for split_name, split_pairs_ in evaluation_splits:
        rows, metrics = evaluate_setting(
            split_pairs_,
            base_mapping,
            context_counts,
            vocab,
            min_count=best["min_count"],
            min_confidence=best["min_confidence"],
            max_edit_distance=best["max_edit_distance"],
            max_syllable_chars=args.max_syllable_chars,
        )
        all_metrics[split_name] = metrics
        (args.output_dir / f"{split_name}_predictions.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "stage": "evaluate",
                    "split": split_name,
                    "exact_match": metrics["exact_match"],
                    "micro_cer": metrics["micro_cer"],
                    "micro_character_accuracy": metrics["micro_character_accuracy"],
                },
                ensure_ascii=False,
            )
        )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"stage": "done", "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
