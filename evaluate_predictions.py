"""Evaluate restoration predictions and report accuracy and error rates.

Character Accuracy (CA, micro_character_accuracy) is 1 - micro CER, an edit-based complement that
may be negative. Segment restoration accuracy (EM, exact_match) is the fraction of fully correct segments.
All printed rates use fractions, not percentages; full metrics go to --output.
"""

import argparse
import json
from pathlib import Path

from evaluate_restore import compute_metrics
from train_restore import configure_utf8_stdout


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for index, row in enumerate(rows):
        if not all(isinstance(row.get(field), str) for field in ("reference", "prediction")):
            raise ValueError(f"Row {index + 1} needs string reference and prediction fields.")
    metrics = compute_metrics(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ["samples", "micro_character_accuracy", "exact_match", "micro_cer", "line_error_rate", "micro_tibetan_unit_error_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
