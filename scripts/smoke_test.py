"""Run a real, small statistical train-save-load-infer check on synthetic data."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    result = subprocess.run(
        [sys.executable, *map(str, arguments)], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="tibetan-restoration-smoke-") as temporary:
        output = Path(temporary)
        run(
            "train_contextual_corrector.py", "--data-dir", "examples/toy",
            "--output-dir", output, "--iterations", "2",
            "--min-counts", "1,2", "--min-confidences", "0.5",
            "--max-edit-distances", "0,1", "--skip-train-predictions",
        )
        expected = [
            json.loads(line) for line in (output / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        (output / "input.txt").write_text("\n".join(row["source"] for row in expected) + "\n", encoding="utf-8")
        run(
            "infer_contextual.py", "--model", output / "model.json",
            "--input-file", output / "input.txt", "--output-file", output / "predictions.txt",
        )
        actual = (output / "predictions.txt").read_text(encoding="utf-8").splitlines()
        if actual != [row["prediction"] for row in expected]:
            raise AssertionError("Complete JSON reload changed held-out predictions.")
        if not expected:
            raise AssertionError("Smoke test has no held-out examples.")
        print(json.dumps({"status": "passed", "test_samples": len(expected), "checks": ["training", "full_state_export", "reload", "batch_inference"]}))


if __name__ == "__main__":
    main()
