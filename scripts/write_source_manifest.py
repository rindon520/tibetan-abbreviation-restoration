"""Record the original and packaged code hashes during local packaging."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(ROOT.glob("*.py")):
        source = args.source_root / path.name
        records.append({
            "path": path.name,
            "original_path": path.name if source.exists() else None,
            "original_sha256": digest(source) if source.exists() else None,
            "packaged_sha256": digest(path),
            "changed_from_original": digest(source) != digest(path) if source.exists() else None,
        })
    for path in sorted((ROOT / "legacy/mt5").glob("*.py")):
        source = args.source_root / path.name
        records.append({"path": path.relative_to(ROOT).as_posix(), "original_path": path.name, "original_sha256": digest(source), "packaged_sha256": digest(path), "changed_from_original": digest(source) != digest(path)})
    driver = ROOT / "scripts/run_grouped_experiment.py"
    original_driver = args.source_root / "submission/experiments/run_grouped_experiment.py"
    records.append({"path": driver.relative_to(ROOT).as_posix(), "original_path": "corrected_experiment/run_grouped_experiment.py", "original_sha256": digest(original_driver) if original_driver.exists() else None, "packaged_sha256": digest(driver), "changed_from_original": digest(original_driver) != digest(driver) if original_driver.exists() else None})
    for name in ['train_grouped_seq2seq.py', 'evaluate_tibnorm_external.py', 'diagnose_frozen_candidates.py', 'ablate_neural_candidate.py', 'run_seed_replications.py', 'summarize_seed_runs.py']:
        packaged_script = ROOT / 'scripts' / name
        original_script = args.source_root / 'submission/experiments' / name
        if packaged_script.exists():
            records.append({'path': f'scripts/{name}', 'original_path': f'corrected_experiment/{name}', 'original_sha256': digest(original_script) if original_script.exists() else None, 'packaged_sha256': digest(packaged_script), 'changed_from_original': digest(original_script) != digest(packaged_script) if original_script.exists() else None})
    manifest = {
        "schema_version": 1,
        "packaged_for": "tibetan-abbreviation-restoration",
        "paper_title": "上下文感知转写与候选重排序的藏文缩略词还原方法",
        "paper_title_en": "Context-Aware Transduction and Candidate Reranking for Tibetan Abbreviation Restoration",
        "preservation": "Original files are copied, never modified by the packaging process.",
        "modifications": {
            "train_contextual_corrector.py": "Export complete, untruncated runtime state; original learning/scoring unchanged.",
            "train_hybrid_selector.py": "Validate candidate alignment and class support; export portable full selector JSON.",
            "train_edit_tagger.py": "Remove implicit duplicate OpenMP runtime override; use a clean dependency environment.",
            "train_restore.py": "Build vocabulary after splitting and only from training examples.",
            "segment_tibetan.py": "Resolve external segmenter working directory for portable relative-path execution.",
            "legacy/mt5/predict.py": "Align model directory and task prefix with the paired historical training script.",
            "scripts/run_grouped_experiment.py": "Copy corrected protocol; portable default root, stage-specific imports, strict frozen-split checks, stage environment snapshots.",
            "scripts/train_grouped_seq2seq.py": "Portable default root, frozen corpus hash check, standard checkpoint configuration alias.",
            "scripts/evaluate_tibnorm_external.py": "Portable default root, enforce fixed upstream commit and frozen corpus hash.",
            "scripts/diagnose_frozen_candidates.py": "Portable root/run-dir, strict candidate alignment; fixed post-hoc scope retained.",
            "scripts/ablate_neural_candidate.py": "Portable root/run-dir, use dev-selected correction distance, explicitly exploratory prior-test-observation scope.",
            "scripts/run_seed_replications.py": "Portable repo/run paths; report all planned seeds without unsupported preregistration claim.",
            "scripts/summarize_seed_runs.py": "Portable run paths and explicit matching manifest/primary-model hash checks.",
        },
        "files": records,
    }
    (ROOT / "docs/source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_manifest_entries": len(records), "changed_original_files": [row['path'] for row in records if row['changed_from_original']]}))


if __name__ == "__main__":
    main()
