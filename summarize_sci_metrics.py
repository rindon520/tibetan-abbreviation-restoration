import argparse
import csv
import json
import random
from pathlib import Path

from evaluate_restore import compute_metrics
from train_restore import levenshtein


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def terminal_shad(prediction):
    if prediction.endswith("་"):
        return prediction[:-1] + "།"
    return prediction


def transform_rows(rows, mode):
    transformed = []
    for row in rows:
        if mode == "identity":
            prediction = row["source"]
        elif mode == "terminal_shad":
            prediction = terminal_shad(row["prediction"])
        elif mode == "model":
            prediction = row["prediction"]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        transformed.append(
            {
                "source": row["source"],
                "reference": row["reference"],
                "prediction": prediction,
            }
        )
    return transformed


def bootstrap_ci(rows, iterations=1000, seed=42):
    rng = random.Random(seed)
    per_row = [
        (
            1 if row["prediction"] == row["reference"] else 0,
            levenshtein(row["prediction"], row["reference"]),
            max(1, len(row["reference"])),
        )
        for row in rows
    ]
    n_rows = len(per_row)
    exact_values = []
    cer_values = []
    for _ in range(iterations):
        exact = 0
        distance = 0
        reference_len = 0
        for _ in range(n_rows):
            row_exact, row_distance, row_reference_len = per_row[rng.randrange(n_rows)]
            exact += row_exact
            distance += row_distance
            reference_len += row_reference_len
        exact_values.append(exact / n_rows)
        cer_values.append(distance / max(1, reference_len))

    exact_values.sort()
    cer_values.sort()
    lo = int(0.025 * iterations)
    hi = int(0.975 * iterations) - 1
    return {
        "exact_match_ci95": [exact_values[lo], exact_values[hi]],
        "micro_cer_ci95": [cer_values[lo], cer_values[hi]],
    }


def paired_bootstrap_comparison(primary_rows, candidate_rows, iterations=1000, seed=42):
    if len(primary_rows) != len(candidate_rows):
        raise ValueError("Paired predictions must have the same number of rows.")
    paired = []
    better = worse = equal = 0
    for primary, candidate in zip(primary_rows, candidate_rows):
        if primary["source"] != candidate["source"] or primary["reference"] != candidate["reference"]:
            raise ValueError("Paired predictions are not aligned.")
        reference = primary["reference"]
        primary_distance = levenshtein(primary["prediction"], reference)
        candidate_distance = levenshtein(candidate["prediction"], reference)
        if candidate_distance < primary_distance:
            better += 1
        elif candidate_distance > primary_distance:
            worse += 1
        else:
            equal += 1
        paired.append(
            (
                int(primary["prediction"] == reference),
                int(candidate["prediction"] == reference),
                primary_distance,
                candidate_distance,
                max(1, len(reference)),
            )
        )

    rng = random.Random(seed)
    exact_gains = []
    cer_reductions = []
    n_rows = len(paired)
    for _ in range(iterations):
        primary_exact = candidate_exact = 0
        primary_distance = candidate_distance = reference_len = 0
        for _ in range(n_rows):
            pe, ce, pd, cd, length = paired[rng.randrange(n_rows)]
            primary_exact += pe
            candidate_exact += ce
            primary_distance += pd
            candidate_distance += cd
            reference_len += length
        exact_gains.append((candidate_exact - primary_exact) / n_rows)
        cer_reductions.append((primary_distance - candidate_distance) / reference_len)

    exact_gains.sort()
    cer_reductions.sort()
    lo = int(0.025 * iterations)
    hi = int(0.975 * iterations) - 1
    primary_exact = sum(item[0] for item in paired) / n_rows
    candidate_exact = sum(item[1] for item in paired) / n_rows
    reference_len = sum(item[4] for item in paired)
    primary_cer = sum(item[2] for item in paired) / reference_len
    candidate_cer = sum(item[3] for item in paired) / reference_len
    return {
        "samples": n_rows,
        "candidate_better_edit_distance": better,
        "candidate_worse_edit_distance": worse,
        "candidate_equal_edit_distance": equal,
        "exact_match_gain": candidate_exact - primary_exact,
        "exact_match_gain_ci95": [exact_gains[lo], exact_gains[hi]],
        "absolute_cer_reduction": primary_cer - candidate_cer,
        "absolute_cer_reduction_ci95": [cer_reductions[lo], cer_reductions[hi]],
        "relative_cer_reduction": (primary_cer - candidate_cer) / primary_cer,
    }


def load_transformer_metrics(runs_dir, baseline_cer_by_grain):
    transformer_specs = [
        ("tibetan_restore", "transformer_original_vocab", "line"),
        ("tibetan_restore_char_vocab", "transformer_char_vocab", "line"),
        ("tibetan_restore_char_vocab_visible_retry", "transformer_char_vocab_retry", "line"),
        ("tibetan_restore_segments_trusted", "transformer_segments_trusted", "segments"),
    ]
    rows = []
    for run_dir, name, grain in transformer_specs:
        metrics_path = runs_dir / run_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        generation_metrics = metrics.get("test_generation", metrics)
        cer = generation_metrics.get("cer")
        exact_match = generation_metrics.get("exact_match")
        item = {
            "name": name,
            "grain": grain,
            "mode": "transformer",
            "samples": generation_metrics.get("samples"),
            "test_loss": metrics.get("test_loss"),
            "exact_match": exact_match,
            "line_error_rate": None if exact_match is None else 1 - exact_match,
            "micro_cer": cer,
            "micro_character_accuracy": None if cer is None else 1 - cer,
        }
        baseline_cer = baseline_cer_by_grain.get(grain)
        if baseline_cer is not None and cer is not None:
            item["relative_cer_reduction_vs_identity"] = (baseline_cer - cer) / baseline_cer
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build SCI-style metric summary tables.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    prediction_specs = [
        (
            "line_identity",
            "line",
            args.runs_dir / "tibetan_transducer_line" / "test_predictions.jsonl",
            "identity",
        ),
        (
            "line_transducer",
            "line",
            args.runs_dir / "tibetan_transducer_line" / "test_predictions.jsonl",
            "model",
        ),
        (
            "line_transducer_terminal_shad",
            "line",
            args.runs_dir / "tibetan_transducer_line" / "test_predictions.jsonl",
            "terminal_shad",
        ),
        (
            "segments_identity",
            "segments",
            args.runs_dir / "tibetan_transducer_segments" / "test_predictions.jsonl",
            "identity",
        ),
        (
            "segments_transducer",
            "segments",
            args.runs_dir / "tibetan_transducer_segments" / "test_predictions.jsonl",
            "model",
        ),
        (
            "segments_transducer_terminal_shad",
            "segments",
            args.runs_dir / "tibetan_transducer_segments" / "test_predictions.jsonl",
            "terminal_shad",
        ),
        (
            "segments_syllable_corrector",
            "segments",
            args.runs_dir / "tibetan_syllable_corrector_segments" / "test_predictions.jsonl",
            "model",
        ),
        (
            "segments_contextual_corrector",
            "segments",
            args.runs_dir / "tibetan_contextual_corrector_segments" / "test_predictions.jsonl",
            "model",
        ),
        (
            "segments_edit_tagger",
            "segments",
            args.runs_dir / "tibetan_edit_tagger_v1" / "test_predictions.jsonl",
            "model",
        ),
        (
            "segments_calibrated_candidate_selector",
            "segments",
            args.runs_dir / "tibetan_hybrid_selector" / "test_predictions.jsonl",
            "model",
        ),
    ]

    summary = []
    baseline_cer_by_grain = {}
    for name, grain, prediction_path, mode in prediction_specs:
        rows = transform_rows(read_jsonl(prediction_path), mode)
        metrics = compute_metrics(rows)
        metrics.update(
            bootstrap_ci(
                rows,
                iterations=args.bootstrap_iterations,
                seed=42,
            )
        )
        item = {
            "name": name,
            "grain": grain,
            "mode": mode,
            **metrics,
        }
        summary.append(item)
        if mode == "identity":
            baseline_cer_by_grain[grain] = metrics["micro_cer"]

    for item in summary:
        baseline_cer = baseline_cer_by_grain[item["grain"]]
        if item["mode"] == "identity":
            item["relative_cer_reduction_vs_identity"] = None
        else:
            item["relative_cer_reduction_vs_identity"] = (
                baseline_cer - item["micro_cer"]
            ) / baseline_cer

    summary.extend(load_transformer_metrics(args.runs_dir, baseline_cer_by_grain))

    primary_rows = transform_rows(
        read_jsonl(args.runs_dir / "tibetan_contextual_corrector_segments" / "test_predictions.jsonl"),
        "model",
    )
    hybrid_rows = transform_rows(
        read_jsonl(args.runs_dir / "tibetan_hybrid_selector" / "test_predictions.jsonl"),
        "model",
    )
    paired_comparison = paired_bootstrap_comparison(
        primary_rows,
        hybrid_rows,
        iterations=args.bootstrap_iterations,
        seed=42,
    )

    json_path = args.runs_dir / "sci_metrics_summary.json"
    csv_path = args.runs_dir / "sci_metrics_summary.csv"
    paired_path = args.runs_dir / "hybrid_paired_bootstrap.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paired_path.write_text(
        json.dumps(paired_comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    columns = [
        "name",
        "grain",
        "mode",
        "samples",
        "test_loss",
        "exact_match",
        "line_error_rate",
        "micro_cer",
        "micro_character_accuracy",
        "macro_cer_mean",
        "macro_cer_median",
        "micro_tibetan_unit_error_rate",
        "macro_tibetan_unit_error_rate_mean",
        "normalized_edit_similarity_mean",
        "char_overlap_precision_mean",
        "char_overlap_recall_mean",
        "char_overlap_f1_mean",
        "prediction_reference_length_ratio",
        "length_mae",
        "length_rmse",
        "edit_substitutions",
        "edit_insertions",
        "edit_deletions",
        "relative_cer_reduction_vs_identity",
        "exact_match_ci95",
        "micro_cer_ci95",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    preview = [
        {
            key: item.get(key)
            for key in [
                "name",
                "grain",
                "samples",
                "exact_match",
                "micro_cer",
                "micro_character_accuracy",
                "relative_cer_reduction_vs_identity",
                "micro_cer_ci95",
            ]
        }
        for item in summary
    ]
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {paired_path}")


if __name__ == "__main__":
    main()
