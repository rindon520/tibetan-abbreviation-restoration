import argparse
import json
from pathlib import Path

from evaluate_restore import compute_metrics
from tibetan_syllable import apply_syllable_vocabulary_correction, build_syllable_vocabulary
from train_restore import read_parallel_data, split_pairs
from train_transducer import apply_mapping, learn_char_mapping, serializable_counts


def terminal_shad(prediction):
    if prediction.endswith("\u0f0b"):
        return prediction[:-1] + "\u0f0d"
    return prediction


def predict_rows(pairs, char_mapping, vocab, max_edit_distance, max_syllable_chars):
    rows = []
    for source, reference in pairs:
        raw_prediction = apply_mapping(source, char_mapping)
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


def evaluate_setting(pairs, char_mapping, vocab, max_edit_distance, max_syllable_chars):
    rows = predict_rows(
        pairs,
        char_mapping,
        vocab,
        max_edit_distance=max_edit_distance,
        max_syllable_chars=max_syllable_chars,
    )
    return rows, compute_metrics(rows)


def serialize_vocab(vocab, top_k=200):
    counts = vocab["counts"]
    delimiters = vocab["delimiters"]
    return {
        "total": vocab["total"],
        "top_cores": [
            {
                "core": core,
                "count": count,
                "delimiter": delimiters[core].most_common(1)[0][0],
            }
            for core, count in counts.most_common(top_k)
        ],
        "unique_cores": len(counts),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a syllable-vocabulary corrector on top of the character transducer."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_segments_trusted"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/tibetan_syllable_corrector_segments")
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-output-length", type=int, default=8)
    parser.add_argument("--max-syllable-chars", type=int, default=8)
    parser.add_argument("--max-edit-distances", default="0,1")
    parser.add_argument("--skip-train-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = read_parallel_data(args.data_dir / "Source.txt", args.data_dir / "Restore.txt")
    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs, valid_ratio=0.1, test_ratio=0.05, seed=42
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

    char_mapping, char_counts = learn_char_mapping(
        train_pairs,
        iterations=args.iterations,
        max_output_length=args.max_output_length,
    )
    vocab = build_syllable_vocabulary(train_pairs)
    print(
        json.dumps(
            {
                "stage": "trained_base_models",
                "char_mapping_size": len(char_mapping),
                "syllable_vocab_size": len(vocab["counts"]),
            },
            ensure_ascii=False,
        )
    )

    candidate_distances = [
        int(part.strip())
        for part in args.max_edit_distances.split(",")
        if part.strip()
    ]
    tuning = []
    best = None
    for max_edit_distance in candidate_distances:
        valid_rows, valid_metrics = evaluate_setting(
            valid_pairs,
            char_mapping,
            vocab,
            max_edit_distance=max_edit_distance,
            max_syllable_chars=args.max_syllable_chars,
        )
        item = {
            "max_edit_distance": max_edit_distance,
            "valid_exact_match": valid_metrics["exact_match"],
            "valid_micro_cer": valid_metrics["micro_cer"],
            "valid_character_accuracy": valid_metrics["micro_character_accuracy"],
        }
        tuning.append(item)
        print(json.dumps({"stage": "tune", **item}, ensure_ascii=False))
        key = (valid_metrics["micro_cer"], -valid_metrics["exact_match"])
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "max_edit_distance": max_edit_distance,
                "valid_rows": valid_rows,
                "valid_metrics": valid_metrics,
            }

    if best is None:
        raise ValueError("No tuning settings were provided.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "model_type": "char_transducer_plus_syllable_vocabulary_corrector",
        "iterations": args.iterations,
        "max_output_length": args.max_output_length,
        "max_syllable_chars": args.max_syllable_chars,
        "best_max_edit_distance": best["max_edit_distance"],
        "tuning": tuning,
        "char_mapping": char_mapping,
        "char_counts_top": serializable_counts(char_counts),
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
            char_mapping,
            vocab,
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
