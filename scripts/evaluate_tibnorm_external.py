"""Evaluate an unmodified published TibNorm revision on the frozen test.

The external checkout is not redistributed. There is no training or tuning;
the default upstream flag=1 and full normalisation pipeline are fixed.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    upstream = args.upstream_dir.resolve()
    corpus = args.corpus_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise RuntimeError("This fixed test baseline is already evaluated; reuse its frozen metrics")
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != 'd296f6566da1308f7c14180080acc43ace8dd68a':
        raise RuntimeError('The upstream revision differs from the fixed TibNorm comparator commit.')
    status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("Upstream checkout contains changes; use an unmodified pinned revision")
    config = configparser.ConfigParser()
    config.read(upstream / "src/config.ini")
    if config["parameters"]["flag"] != "1":
        raise RuntimeError("The checked-out upstream default flag differs from the fixed flag=1 protocol")
    # Configure a resource path in memory, preserving every rule and default flag.
    config["paths"]["table_path"] = str(upstream / "tables")
    original_io = load_module("tibnorm_original_io", upstream / "src/module.py")
    original_normaliser = load_module("tibnorm_original_normaliser", upstream / "src/normaliser.py")
    tables = original_io.load_tables(config)
    manifest_path = corpus / "split_manifest.json"
    records_path = corpus / "records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != manifest['records_sha256']:
        raise ValueError('Frozen corpus records have changed.')
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [records[i] for i in manifest["splits"]["test"]["record_indices"]]
    texts = {f"segment_{i:05d}": record["source"] for i, record in enumerate(test)}
    # No target text is supplied to the upstream inference code.
    started = time.perf_counter()
    predictions = original_normaliser.Normaliser(texts, tables).normalisation()
    elapsed = time.perf_counter() - started
    rows = [
        {"source": record["source"], "reference": record["reference"], "prediction": predictions[f"segment_{i:05d}"], "record_id": record["record_id"], "group_id": record["group_id"]}
        for i, record in enumerate(test)
    ]
    sys.path.insert(0, str(repo))
    from evaluate_restore import compute_metrics
    from train_restore import levenshtein
    source_hashes = {
        str(path.relative_to(upstream)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [upstream / "src/config.ini", upstream / "src/normaliser.py", upstream / "src/module.py", *sorted((upstream / "tables").glob("*.tsv"))]
    }
    payload = {
        "model_type": "external_published_rule_normalisation",
        "name": "TibNorm (unmodified default full pipeline)",
        "upstream_url": "https://github.com/Divergent-Discourses/TibNorm",
        "upstream_commit": commit,
        "upstream_doi": "10.5281/zenodo.10806456",
        "config_flag": 1,
        "tables_loaded_rows": {name: len(frame) for name, frame in tables.items()},
        "effective_abbreviation_keys": len(tables["abbreviations"].set_index("transcription")["normalisation"].to_dict()),
        "pipeline_order": ["abbreviations", "table1", "table2", "table3"],
        "input_policy": "One frozen source segment per text; no boundary padding; no source/reference text normalization beyond original normaliser",
        "resource_condition": "Published external abbreviation tables, not restricted to this corpus training split",
        "test_only_evaluation_no_tuning": True,
        "normalisation_seconds": elapsed,
        "samples": len(rows),
        "changed_source_segments": sum(row["source"] != row["prediction"] for row in rows),
        "total_reference_characters": sum(len(row["reference"]) for row in rows),
        "total_character_distance": sum(levenshtein(row["prediction"], row["reference"]) for row in rows),
        "test_metrics": compute_metrics(rows),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "upstream_source_sha256": source_hashes,
        "caveats": [
            "Resource-unmatched comparator: external published abbreviation dictionary may overlap corpus; corpus provenance does not permit auditing this.",
            "TibNorm additionally changes punctuation, numbers, and spelling. Its complete default normalization is broader than this reference-pair restoration task.",
            "Upstream rules with tsheg boundary requirements may not expand some segment-initial abbreviations; no test-driven patches were made.",
            "No license file is present in this checkout. External code and abbreviation tables are excluded from the GitHub submission package.",
        ],
    }
    (out / "test_predictions.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["upstream_commit", "tables_loaded_rows", "effective_abbreviation_keys", "normalisation_seconds", "samples", "changed_source_segments", "total_character_distance", "total_reference_characters"]}, ensure_ascii=False))
    print(json.dumps({k: payload["test_metrics"][k] for k in ["micro_cer", "micro_tibetan_unit_error_rate", "exact_match"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
