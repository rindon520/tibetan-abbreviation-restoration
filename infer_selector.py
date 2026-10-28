"""Select between aligned candidate JSONL files without using reference labels."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from model_artifacts import validate_candidate_alignment
from train_hybrid_selector import CharacterLanguageModel, load_rows, selector_features


def predict_candidates(state, primary_rows, auxiliary_rows):
    if state.get("format_version") != 1:
        raise ValueError("Unsupported selector state format.")
    validate_candidate_alignment(primary_rows, auxiliary_rows)
    serialized = state["language_model"]
    model = CharacterLanguageModel([], serialized["max_order"], serialized["alpha"])
    model.counts = {int(order): Counter(counts) for order, counts in serialized["counts"].items()}
    model.context_counts = {int(order): Counter(counts) for order, counts in serialized["context_counts"].items()}
    model.alphabet_size = serialized["alphabet_size"]
    rows = []
    for primary, auxiliary in zip(primary_rows, auxiliary_rows):
        features = selector_features(primary, auxiliary, model, state["vocabulary"])
        if len(features) != state["feature_count"]:
            raise ValueError("Artifact feature count differs from the implementation.")
        score = state["intercept"] + sum(
            coefficient * ((feature - mean) / scale)
            for feature, mean, scale, coefficient in zip(
                features, state["scaler_mean"], state["scaler_scale"], state["coefficients"]
            )
        )
        probability = 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))
        choose_auxiliary = probability >= state["threshold"]
        rows.append({
            "source": primary["source"],
            "prediction": auxiliary["prediction"] if choose_auxiliary else primary["prediction"],
            "auxiliary_probability": probability,
            "selected_model": "edit_tagger" if choose_auxiliary else "contextual_transducer",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.model.read_text(encoding="utf-8"))
    rows = predict_candidates(state, load_rows(args.primary), load_rows(args.auxiliary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
