from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from abbrev_restore import (
    build_examples,
    build_restoration_table,
    candidate_thresholds,
    detector_payload,
    evaluate_detection,
    evaluate_restoration,
    predict_restoration_rows,
    train_detector,
)
from segment_tibetan import DEFAULT_SEGMENTER_DIR, segment_lines
from tibetan_syllable import build_syllable_vocabulary
from train_restore import configure_utf8_stdout, read_parallel_data, split_pairs
from train_transducer import learn_char_mapping, serializable_counts


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def log(payload: dict):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def split_records(records: list[dict], valid_ratio: float, test_ratio: float, seed: int):
    return split_pairs(records, valid_ratio=valid_ratio, test_ratio=test_ratio, seed=seed)


def make_examples(records: list[dict], max_ref_units_per_source: int):
    return build_examples(
        pairs=[(record["source"], record["reference"]) for record in records],
        source_segmented=[record["source_segmented"] for record in records],
        reference_segmented=[record["reference_segmented"] for record in records],
        max_ref_units_per_source=max_ref_units_per_source,
    )


def summarize_labels(examples: list[dict]) -> dict:
    labels = Counter()
    token_count = 0
    for example in examples:
        labels.update(example["labels"])
        token_count += len(example["labels"])
    return {
        "examples": len(examples),
        "tokens": token_count,
        "positive_tokens": labels[1],
        "negative_tokens": labels[0],
        "positive_rate": labels[1] / max(1, token_count),
    }


def evaluate_setting(
    examples: list[dict],
    detector,
    threshold: float,
    restoration_table: dict[str, str],
    char_mapping: dict[str, str],
    syllable_vocab: dict,
    max_edit_distance: int,
    max_syllable_chars: int,
) -> tuple[list[dict], dict]:
    rows = predict_restoration_rows(
        examples,
        detector=detector,
        threshold=threshold,
        restoration_table=restoration_table,
        char_mapping=char_mapping,
        syllable_vocab=syllable_vocab,
        max_edit_distance=max_edit_distance,
        max_syllable_chars=max_syllable_chars,
    )
    return rows, {
        "detection": evaluate_detection(examples, detector, threshold=threshold),
        "restoration": evaluate_restoration(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a segmentation-aware Tibetan abbreviation detector and restorer."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_segments_trusted"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tibetan_abbrev_restore"))
    parser.add_argument("--segmenter-dir", type=Path, default=DEFAULT_SEGMENTER_DIR)
    parser.add_argument("--prefer-external-segmenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-fallback-segmenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmenter-timeout", type=float, default=10.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector-epochs", type=int, default=6)
    parser.add_argument("--transducer-iterations", type=int, default=5)
    parser.add_argument("--max-output-length", type=int, default=8)
    parser.add_argument("--max-ref-units-per-source", type=int, default=4)
    parser.add_argument("--max-syllable-chars", type=int, default=8)
    parser.add_argument("--max-edit-distances", default="0,1")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-threshold-candidates", type=int, default=25)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--skip-train-predictions", action="store_true")
    return parser.parse_args()


def main():
    configure_utf8_stdout()
    args = parse_args()
    pairs = read_parallel_data(args.data_dir / "Source.txt", args.data_dir / "Restore.txt")
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    log({"stage": "data", "pairs": len(pairs), "data_dir": str(args.data_dir)})

    source_segmented, source_segment_status = segment_lines(
        [source for source, _ in pairs],
        segmenter_dir=args.segmenter_dir,
        prefer_external=args.prefer_external_segmenter,
        allow_fallback=args.allow_fallback_segmenter,
        timeout_seconds=args.segmenter_timeout,
    )
    prefer_reference_external = args.prefer_external_segmenter and source_segment_status["mode"] == "external"
    reference_segmented, reference_segment_status = segment_lines(
        [reference for _, reference in pairs],
        segmenter_dir=args.segmenter_dir,
        prefer_external=prefer_reference_external,
        allow_fallback=args.allow_fallback_segmenter,
        timeout_seconds=args.segmenter_timeout,
    )
    log(
        {
            "stage": "segment",
            "source": source_segment_status,
            "reference": reference_segment_status,
        }
    )

    records = [
        {
            "source": source,
            "reference": reference,
            "source_segmented": source_seg,
            "reference_segmented": reference_seg,
        }
        for (source, reference), source_seg, reference_seg in zip(
            pairs, source_segmented, reference_segmented
        )
    ]
    train_records, valid_records, test_records = split_records(
        records,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_examples = make_examples(train_records, args.max_ref_units_per_source)
    valid_examples = make_examples(valid_records, args.max_ref_units_per_source)
    test_examples = make_examples(test_records, args.max_ref_units_per_source)
    label_summary = {
        "train": summarize_labels(train_examples),
        "valid": summarize_labels(valid_examples),
        "test": summarize_labels(test_examples),
    }
    log({"stage": "labels", **label_summary})

    train_pairs = [(record["source"], record["reference"]) for record in train_records]
    log({"stage": "train_restoration_table"})
    restoration_table = build_restoration_table(train_examples)
    log({"stage": "train_detector"})
    detector = train_detector(
        train_examples,
        restoration_table=restoration_table,
        epochs=args.detector_epochs,
        seed=args.seed,
    )
    log({"stage": "train_char_mapping"})
    char_mapping, char_counts = learn_char_mapping(
        train_pairs,
        iterations=args.transducer_iterations,
        max_output_length=args.max_output_length,
    )
    log({"stage": "build_syllable_vocab"})
    syllable_vocab = build_syllable_vocabulary(train_pairs)
    log(
        {
            "stage": "trained",
            "restoration_table_size": len(restoration_table),
            "detector_weights": len(detector.weights),
            "char_mapping_size": len(char_mapping),
            "syllable_vocab_size": len(syllable_vocab["counts"]),
        }
    )

    best = None
    tuning = []
    threshold_candidates = candidate_thresholds(detector, valid_examples)[:]
    if args.max_threshold_candidates > 0 and len(threshold_candidates) > args.max_threshold_candidates:
        step = (len(threshold_candidates) - 1) / (args.max_threshold_candidates - 1)
        threshold_candidates = [
            threshold_candidates[round(index * step)]
            for index in range(args.max_threshold_candidates)
        ]
    edit_distance_candidates = parse_int_list(args.max_edit_distances)
    for threshold in threshold_candidates:
        for max_edit_distance in edit_distance_candidates:
            _, metrics = evaluate_setting(
                valid_examples,
                detector=detector,
                threshold=threshold,
                restoration_table=restoration_table,
                char_mapping=char_mapping,
                syllable_vocab=syllable_vocab,
                max_edit_distance=max_edit_distance,
                max_syllable_chars=args.max_syllable_chars,
            )
            item = {
                "threshold": threshold,
                "max_edit_distance": max_edit_distance,
                "valid_detection_f1": metrics["detection"]["f1"],
                "valid_detection_accuracy": metrics["detection"]["accuracy"],
                "valid_restore_exact_match": metrics["restoration"]["exact_match"],
                "valid_restore_micro_cer": metrics["restoration"]["micro_cer"],
                "valid_restore_character_accuracy": metrics["restoration"][
                    "micro_character_accuracy"
                ],
            }
            tuning.append(item)
            log({"stage": "tune", **item})
            key = (
                metrics["restoration"]["micro_cer"],
                -metrics["restoration"]["exact_match"],
                -metrics["detection"]["f1"],
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "threshold": threshold,
                    "max_edit_distance": max_edit_distance,
                    "valid_metrics": metrics,
                }

    if best is None:
        raise ValueError("No tuning setting was evaluated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "model_type": "segmentation_aware_abbreviation_detector_restorer",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "best_threshold": best["threshold"],
        "best_max_edit_distance": best["max_edit_distance"],
        "segmenter": {
            "source": source_segment_status,
            "reference": reference_segment_status,
        },
        "label_summary": label_summary,
        "tuning": tuning,
        "detector": detector_payload(detector),
        "restoration_table": restoration_table,
        "char_mapping": char_mapping,
        "char_counts_top": serializable_counts(char_counts),
        "syllable_vocab": {
            "unique_cores": len(syllable_vocab["counts"]),
            "total": syllable_vocab["total"],
        },
        "data": {
            "pairs": len(pairs),
            "train": len(train_examples),
            "valid": len(valid_examples),
            "test": len(test_examples),
        },
    }
    write_json(args.output_dir / "model.json", model)
    write_json(args.output_dir / "segmented_records.sample.json", records[: args.sample_count])

    all_metrics = {}
    splits = [("valid", valid_examples), ("test", test_examples)]
    if not args.skip_train_predictions:
        splits.append(("train", train_examples))
    for split_name, examples in splits:
        rows, metrics = evaluate_setting(
            examples,
            detector=detector,
            threshold=best["threshold"],
            restoration_table=restoration_table,
            char_mapping=char_mapping,
            syllable_vocab=syllable_vocab,
            max_edit_distance=best["max_edit_distance"],
            max_syllable_chars=args.max_syllable_chars,
        )
        all_metrics[split_name] = metrics
        write_jsonl(args.output_dir / f"{split_name}_predictions.jsonl", rows)
        sample_rows = rows[: args.sample_count]
        write_json(args.output_dir / f"{split_name}_samples.json", sample_rows)
        log(
            {
                "stage": "evaluate",
                "split": split_name,
                "detection_accuracy": metrics["detection"]["accuracy"],
                "detection_precision": metrics["detection"]["precision"],
                "detection_recall": metrics["detection"]["recall"],
                "detection_f1": metrics["detection"]["f1"],
                "restore_exact_match": metrics["restoration"]["exact_match"],
                "restore_micro_cer": metrics["restoration"]["micro_cer"],
                "restore_character_accuracy": metrics["restoration"][
                    "micro_character_accuracy"
                ],
            }
        )
        for index, row in enumerate(sample_rows, start=1):
            log(
                {
                    "stage": "sample",
                    "split": split_name,
                    "index": index,
                    "source": row["source"],
                    "prediction": row["prediction"],
                    "reference": row["reference"],
                    "detected": row["detected"],
                }
            )

    write_json(args.output_dir / "metrics.json", all_metrics)
    log({"stage": "done", "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
