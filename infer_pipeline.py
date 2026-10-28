"""Restore new source text with a frozen grouped two-candidate experiment."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from infer_selector import predict_candidates
from scripts.run_grouped_experiment import load_corpus, restore_context_state, tagger_test_rows
from train_contextual_corrector import predict_rows
from train_restore import configure_utf8_stdout


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    texts = [args.text] if args.text is not None else args.input_file.read_text(encoding="utf-8-sig").splitlines()
    if not texts or any(not text for text in texts):
        raise ValueError("Input must contain at least one nonempty text record; remove empty lines.")
    root = args.experiment_dir.resolve()
    manifest = json.loads((root / "corpus/split_manifest.json").read_text(encoding="utf-8"))
    corpus_args = SimpleNamespace(output_dir=root, raw_data_dir=None, seed=manifest["seed"], batch_size=args.batch_size, device=args.device)
    _, _, pairs = load_corpus(corpus_args)
    checkpoint_info = json.loads((root / "edit_tagger/metrics.json").read_text(encoding="utf-8"))
    limit = checkpoint_info["config"]["max_positions"]
    if any(len(text) > limit for text in texts):
        raise ValueError(f"Input exceeds the trained edit tagger's {limit}-codepoint limit; split longer inputs first.")
    state = json.loads((root / "contextual/model_full.json").read_text(encoding="utf-8"))
    mapping, contexts, vocabulary, parameters = restore_context_state(state)
    unlabeled = [(text, "") for text in texts]
    primary = predict_rows(unlabeled, mapping, contexts, vocabulary, **parameters)
    auxiliary = tagger_test_rows(corpus_args, {"train": pairs["train"], "test": unlabeled})
    selector = json.loads((root / "selector/selector_model.json").read_text(encoding="utf-8"))
    rows = predict_candidates(selector, primary, auxiliary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
