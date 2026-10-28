from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_restore import compute_metrics
from tibetan_syllable import split_syllable_units
from train_restore import levenshtein, read_parallel_data, split_pairs
from model_artifacts import validate_candidate_alignment


class CharacterLanguageModel:
    def __init__(self, references, max_order=5, alpha=0.1):
        self.max_order = max_order
        self.alpha = alpha
        self.counts = {order: Counter() for order in range(1, max_order + 1)}
        self.context_counts = {
            order: Counter() for order in range(2, max_order + 1)
        }
        for reference in references:
            padded = "^" * (max_order - 1) + reference + "$" * (max_order - 1)
            for order in range(1, max_order + 1):
                self.counts[order].update(
                    padded[index : index + order]
                    for index in range(len(padded) - order + 1)
                )
                if order > 1:
                    self.context_counts[order].update(
                        padded[index : index + order - 1]
                        for index in range(len(padded) - order + 2)
                    )
        self.alphabet_size = max(1, len(self.counts[1]))

    def score(self, text, order):
        padded = "^" * (self.max_order - 1) + text + "$" * (self.max_order - 1)
        total = 0.0
        terms = 0
        for index in range(len(padded) - order + 1):
            ngram = padded[index : index + order]
            context = ngram[:-1]
            numerator = self.counts[order][ngram] + self.alpha
            denominator = (
                self.context_counts[order][context]
                + self.alpha * self.alphabet_size
            )
            total += math.log(numerator / denominator)
            terms += 1
        return total / max(1, terms)


def load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def syllable_coverage(text, vocabulary):
    cores = [unit.core for unit in split_syllable_units(text) if unit.core]
    return sum(core in vocabulary for core in cores) / max(1, len(cores))


def selector_features(primary, auxiliary, language_model, vocabulary):
    source = primary["source"]
    first = primary["prediction"]
    second = auxiliary["prediction"]
    source_length = max(1, len(source))
    features = []
    for order in (2, 3, 4, 5):
        first_score = language_model.score(first, order)
        second_score = language_model.score(second, order)
        features.extend([first_score, second_score, second_score - first_score])

    first_source_distance = levenshtein(source, first) / max(
        1, max(len(source), len(first))
    )
    second_source_distance = levenshtein(source, second) / max(
        1, max(len(source), len(second))
    )
    first_coverage = syllable_coverage(first, vocabulary)
    second_coverage = syllable_coverage(second, vocabulary)
    confidences = [
        decision["confidence"] for decision in auxiliary.get("decisions", [])
    ]
    fallback_ratio = sum(
        decision.get("origin") == "rule_fallback"
        for decision in auxiliary.get("decisions", [])
    ) / max(1, len(confidences))

    features.extend(
        [
            first_source_distance,
            second_source_distance,
            second_source_distance - first_source_distance,
            len(first) / source_length,
            len(second) / source_length,
            (len(second) - len(first)) / source_length,
            first_coverage,
            second_coverage,
            second_coverage - first_coverage,
            levenshtein(first, second) / max(1, max(len(first), len(second))),
            len(source),
            float(np.mean(confidences)) if confidences else 0.0,
            float(np.min(confidences)) if confidences else 0.0,
            fallback_ratio,
            levenshtein(primary.get("base_prediction", first), first)
            / max(1, len(first)),
            levenshtein(auxiliary.get("raw_prediction", second), second)
            / max(1, len(second)),
        ]
    )
    return features


def build_rows(primary_rows, auxiliary_rows, choose_auxiliary, probabilities):
    rows = []
    for primary, auxiliary, choose, probability in zip(
        primary_rows, auxiliary_rows, choose_auxiliary, probabilities
    ):
        rows.append(
            {
                "source": primary["source"],
                "reference": primary["reference"],
                "prediction": (
                    auxiliary["prediction"] if choose else primary["prediction"]
                ),
                "primary_prediction": primary["prediction"],
                "auxiliary_prediction": auxiliary["prediction"],
                "auxiliary_probability": float(probability),
                "selected_model": "edit_tagger" if choose else "contextual_transducer",
            }
        )
    return rows


def save_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate a hybrid Tibetan restoration selector."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_segments_trusted"))
    parser.add_argument(
        "--primary-run", type=Path, default=Path("runs/tibetan_contextual_corrector_segments")
    )
    parser.add_argument(
        "--auxiliary-run", type=Path, default=Path("runs/tibetan_edit_tagger_v1")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/tibetan_hybrid_selector")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    pairs = read_parallel_data(
        args.data_dir / "Source.txt", args.data_dir / "Restore.txt"
    )
    train_pairs, valid_pairs, test_pairs = split_pairs(
        pairs, valid_ratio=0.1, test_ratio=0.05, seed=args.seed
    )
    language_model = CharacterLanguageModel(reference for _, reference in train_pairs)
    vocabulary = Counter(
        unit.core
        for _, reference in train_pairs
        for unit in split_syllable_units(reference)
        if unit.core
    )

    primary_valid = load_rows(args.primary_run / "valid_predictions.jsonl")
    auxiliary_valid = load_rows(args.auxiliary_run / "valid_predictions.jsonl")
    primary_test = load_rows(args.primary_run / "test_predictions.jsonl")
    auxiliary_test = load_rows(args.auxiliary_run / "test_predictions.jsonl")
    if not (
        len(primary_valid) == len(auxiliary_valid) == len(valid_pairs)
        and len(primary_test) == len(auxiliary_test) == len(test_pairs)
    ):
        raise ValueError("Prediction files do not match the frozen data split.")
    validate_candidate_alignment(primary_valid, auxiliary_valid, valid_pairs)
    validate_candidate_alignment(primary_test, auxiliary_test, test_pairs)

    valid_features = np.asarray(
        [
            selector_features(primary, auxiliary, language_model, vocabulary)
            for primary, auxiliary in zip(primary_valid, auxiliary_valid)
        ],
        dtype=np.float64,
    )
    valid_labels = np.asarray(
        [
            levenshtein(auxiliary["prediction"], primary["reference"])
            < levenshtein(primary["prediction"], primary["reference"])
            for primary, auxiliary in zip(primary_valid, auxiliary_valid)
        ],
        dtype=np.int64,
    )

    candidates = []
    if len(np.unique(valid_labels)) != 2 or min(np.bincount(valid_labels)) < 5:
        raise ValueError(
            "Five-fold stratification requires at least five calibration examples "
            "in each preference class. Provide a larger calibration set."
        )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for regularization in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization, max_iter=2000, random_state=args.seed
            ),
        )
        probabilities = cross_val_predict(
            model,
            valid_features,
            valid_labels,
            cv=folds,
            method="predict_proba",
        )[:, 1]
        for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            selected = probabilities >= threshold
            rows = build_rows(
                primary_valid, auxiliary_valid, selected, probabilities
            )
            metrics = compute_metrics(rows)
            candidates.append(
                {
                    "regularization": regularization,
                    "threshold": threshold,
                    "metrics": metrics,
                    "probabilities": probabilities,
                    "selected": selected,
                    "rows": rows,
                }
            )

    best = min(
        candidates,
        key=lambda item: (
            round(item["metrics"]["micro_cer"], 12),
            -item["metrics"]["exact_match"],
        ),
    )
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=best["regularization"], max_iter=2000, random_state=args.seed
        ),
    ).fit(valid_features, valid_labels)
    test_features = np.asarray(
        [
            selector_features(primary, auxiliary, language_model, vocabulary)
            for primary, auxiliary in zip(primary_test, auxiliary_test)
        ],
        dtype=np.float64,
    )
    test_probabilities = final_model.predict_proba(test_features)[:, 1]
    test_selected = test_probabilities >= best["threshold"]
    test_rows = build_rows(
        primary_test, auxiliary_test, test_selected, test_probabilities
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scaler = final_model.named_steps["standardscaler"]
    classifier = final_model.named_steps["logisticregression"]
    selector_state = {
        "format_version": 1,
        "threshold": best["threshold"],
        "feature_count": int(valid_features.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "language_model": {
            "max_order": language_model.max_order,
            "alpha": language_model.alpha,
            "alphabet_size": language_model.alphabet_size,
            "counts": {str(order): dict(counts) for order, counts in language_model.counts.items()},
            "context_counts": {str(order): dict(counts) for order, counts in language_model.context_counts.items()},
        },
        "vocabulary": dict(vocabulary),
    }
    (args.output_dir / "selector.json").write_text(
        json.dumps(selector_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_jsonl(args.output_dir / "valid_predictions.jsonl", best["rows"])
    save_jsonl(args.output_dir / "test_predictions.jsonl", test_rows)
    payload = {
        "model_type": "validation_calibrated_candidate_selector",
        "primary_model": str(args.primary_run),
        "auxiliary_model": str(args.auxiliary_run),
        "train_samples": len(train_pairs),
        "calibration_samples": len(valid_pairs),
        "test_samples": len(test_pairs),
        "selection_protocol": "five-fold out-of-fold calibration on validation set",
        "best_regularization": best["regularization"],
        "best_threshold": best["threshold"],
        "calibration_selected_auxiliary": int(best["selected"].sum()),
        "test_selected_auxiliary": int(test_selected.sum()),
        "calibration_metrics": best["metrics"],
        "test_metrics": compute_metrics(test_rows),
        "primary_test_metrics": compute_metrics(primary_test),
        "auxiliary_test_metrics": compute_metrics(auxiliary_test),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
