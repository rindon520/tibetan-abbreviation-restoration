"""Export figure aggregates from the frozen grouped experiment.

This adapter consumes test outputs only. Source text is not redistributed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def evidence_path(path, repo, run):
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        try:
            return 'experiment/' + path.relative_to(run).as_posix()
        except ValueError:
            return 'external-input/' + path.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    repo, run, out = args.repo_root.resolve(), args.run_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from train_restore import levenshtein
    from evaluate_restore import compute_metrics
    sources = set()
    def load(path):
        sources.add(path)
        return json.loads(path.read_text(encoding="utf-8"))
    def load_rows(path):
        sources.add(path)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def write(name, rows):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    frozen = load(run / "evaluation/test_metrics.json")
    manifest = load(run / "corpus/split_manifest.json")
    contextual_training = load(run / "contextual/training_summary.json")
    edit_training = load(run / "edit_tagger/metrics.json")
    selector_training = load(run / "selector/selection_summary.json")
    metrics = frozen["test_metrics"]
    mapping = [
        ("identity", "identity", "原文保持"),
        ("global_transducer", "transducer", "字符转写"),
        ("global_plus_shad", "terminal", "字符转写＋句末修正"),
        ("global_plus_shad_vocab", "syllable", "字符转写＋词表重切分"),
        ("contextual", "contextual", "上下文转写分支"),
        ("edit_tagger", "edit", "编辑标注分支"),
        ("hybrid", "selector", "双路候选选择"),
    ]
    def record(method, label, values, condition, distance=None, ref_chars=None):
        return dict(method_id=method, label=label, n=values["samples"], micro_cer=values["micro_cer"], micro_tibetan_unit_error_rate=values["micro_tibetan_unit_error_rate"], exact_match=values["exact_match"], total_reference_characters=ref_chars if ref_chars is not None else values["total_reference_codepoints"], total_character_distance=distance if distance is not None else values["total_edit_distance"], resource_condition=condition)
    comparison = [record(method, label, metrics[name], "current_corpus_train_only") for name, method, label in mapping]
    seq_path = run / "seq2seq/metrics.json"
    if seq_path.exists():
        sq = load(seq_path)["test"]
        sqrows = load_rows(run / "seq2seq/test_predictions.jsonl")
        comparison.append(record("seq2seq", "字符 Transformer（贪心）", sq, "current_corpus_train_only", sum(levenshtein(r["prediction"], r["reference"]) for r in sqrows), sum(len(r["reference"]) for r in sqrows)))
    ext_path = run / "tibnorm_external/metrics.json"
    if ext_path.exists():
        ext = load(ext_path)
        comparison.append(record("tibnorm", "TibNorm†（外部规则）", ext["test_metrics"], "external_published_rule_tables", ext["total_character_distance"], ext["total_reference_characters"]))
    write("main_comparison.csv", comparison)
    write("context_components.csv", [record(name, label, metrics[name], "same_trained_contextual_model") for name, label in [("context_raw", "上下文转写（原始输出）"), ("context_plus_shad", "＋句末符号修正"), ("contextual", "＋训练词表重切分")]])
    context_tuning = []
    for item in contextual_training["tuning"]:
        context_tuning.append(dict(min_count=item["min_count"], min_confidence=item["min_confidence"], max_edit_distance=item["max_edit_distance"], valid_micro_cer=item["dev_micro_cer"]))
    write("context_tuning.csv", context_tuning)
    write("edit_tuning.csv", edit_training["tuning"])
    c = load_rows(run / "evaluation/contextual_test_predictions.jsonl")
    e = load_rows(run / "evaluation/edit_tagger_test_predictions.jsonl")
    h = load_rows(run / "evaluation/hybrid_test_predictions.jsonl")
    if not ([(r["source"],r["reference"]) for r in c] == [(r["source"],r["reference"]) for r in e] == [(r["source"],r["reference"]) for r in h]):
        raise ValueError("Candidate rows differ from frozen test ordering")
    relations = ["编辑候选更优", "两候选等误差", "上下文候选更优"]
    table, outcomes = Counter(), Counter()
    for cr, er, hr in zip(c,e,h):
        cd, ed, hd = [levenshtein(row["prediction"], row["reference"]) for row in [cr,er,hr]]
        rel = relations[0] if ed < cd else relations[2] if ed > cd else relations[1]
        choice = "编辑候选" if hr["selected_model"] == "edit_tagger" else "上下文候选"
        table[rel,choice] += 1
        outcomes["改善" if hd < cd else "变差" if hd > cd else "等误差"] += 1
    write("selection_contingency.csv", [dict(candidate_relation=rel, selected_model=choice, count=table[rel,choice]) for rel in relations for choice in ["编辑候选","上下文候选"]])
    write("selection_outcomes.csv", [dict(outcome=name,count=outcomes[name]) for name in ["改善","变差","等误差"]])
    params = contextual_training["best_parameters"]
    summary = dict(protocol="grouped",protocol_label="联结分组划分（种子 42）",n_train=manifest["splits"]["train"]["samples"],n_valid=manifest["splits"]["dev"]["samples"],n_calibration=manifest["splits"]["calibration"]["samples"],n_test=manifest["splits"]["test"]["samples"],selector_selected_edit=frozen["selector_effect"]["selected_auxiliary"],selector_C=selector_training["best_C"],selector_threshold=selector_training["best_threshold"],edit_threshold=edit_training["best_confidence_threshold"],edit_distance=edit_training["best_max_edit_distance"],context_min_count=params["min_count"],context_min_confidence=params["min_confidence"],context_edit_distance=params["max_edit_distance"],absolute_cer_reduction_vs_contextual=frozen["selector_effect"]["absolute_cer_reduction"],selector_effect=frozen["selector_effect"],notes=["TibNorm uses external published rule tables and broader normalisation; it is a distinct resource condition.","Character Transformer is not pretrained and decoded greedily; its weak result does not establish superiority to strong pretrained sequence models.","All primary points are one predefined seed42 training state; no seed variation is hidden inside a sample confidence interval.","Tibetan unit error includes separator tokens defined by evaluate_restore.tibetan_units; it is not WER.","Development controls candidates; independent calibration controls the selector; final test is frozen."],sources=[dict(path=evidence_path(path,repo,run),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(sources)])
    (out / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(out_dir=str(out),n_test=summary["n_test"],methods=len(comparison)),ensure_ascii=False))


if __name__ == "__main__":
    main()
