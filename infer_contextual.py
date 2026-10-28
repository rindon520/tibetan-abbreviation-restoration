"""Infer with a complete contextual transducer JSON artifact; no PyTorch needed."""

import argparse
import json
from pathlib import Path

from model_artifacts import deserialize_contextual_state
from train_contextual_corrector import predict_rows
from train_restore import configure_utf8_stdout


def predict(model, texts):
    if "parameters" in model and "syllable_counts" in model:
        parameters = model["parameters"]
        model = {
            "runtime_state": {
                "format_version": 1,
                "base_mapping": model["base_mapping"],
                "context_counts": model["context_counts"],
                "vocabulary": {
                    "counts": model["syllable_counts"],
                    "delimiters": model["syllable_delimiters"],
                    "total": sum(model["syllable_counts"].values()),
                },
            },
            "best_min_count": parameters["min_count"],
            "best_min_confidence": parameters["min_confidence"],
            "best_max_edit_distance": parameters["max_edit_distance"],
            "max_syllable_chars": parameters["max_syllable_chars"],
        }
    if "runtime_state" not in model:
        raise ValueError(
            "This legacy artifact only stores truncated counts. "
            "Retrain with this repository to export a complete runtime_state."
        )
    mapping, contexts, vocabulary = deserialize_contextual_state(model["runtime_state"])
    rows = predict_rows(
        [(text, "") for text in texts],
        mapping,
        contexts,
        vocabulary,
        min_count=model["best_min_count"],
        min_confidence=model["best_min_confidence"],
        max_edit_distance=model["best_max_edit_distance"],
        max_syllable_chars=model["max_syllable_chars"],
    )
    return [row["prediction"] for row in rows]


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    texts = [args.text] if args.text is not None else args.input_file.read_text(encoding="utf-8-sig").splitlines()
    output = "\n".join(predict(model, texts)) + "\n"
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
