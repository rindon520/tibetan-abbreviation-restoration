"""Export only aggregate metrics and provenance hashes, never corpus text."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_metrics(metrics):
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results_summary.json")
    args = parser.parse_args()
    run = args.run_dir
    evaluation = read(run / "evaluation/test_metrics.json")
    manifest_path = run / "corpus/split_manifest.json"
    manifest = read(manifest_path)
    output = {
        "schema_version": 1,
        "paper_title": "上下文感知转写与候选重排序的藏文缩略词还原方法",
        "paper_title_en": "Context-Aware Transduction and Candidate Reranking for Tibetan Abbreviation Restoration",
        "paper_figure_entry": "figures/v21/build_v21_figures.py",
        "paper_figure_mapping": {
            "1": "author-local original case bitmap; not redistributed",
            "2": "fig2_data_protocol", "3": "fig3_method_framework",
            "4": "fig4_main_results", "5": "fig5_calibration_sensitivity",
            "6": "fig6_error_analysis",
        },
        "protocol": evaluation["protocol"],
        "split_seed": manifest["seed"],
        "model_seed": read(run / 'edit_tagger/metrics.json').get('training_args', {}).get('seed', manifest['seed']),
        "data": evaluation["data"],
        "data_source": "据作者说明，配对数据整理自《藏文缩略语词典》（谢拉编著，民族出版社），并参考《缩略语词典》（项布旺杰，四川民族出版社）；出版年份、版次、词条来源对应、录入复核和共享授权仍待补充。",
        "known_data_sources": [
            {"title": "藏文缩略语词典", "author": "谢拉", "responsibility_statement": "谢拉编著", "publisher": "民族出版社", "publication_year": None, "edition": None, "role": "作者说明的数据整理来源", "metadata_basis": "书名、编著者与出版社由作者提供；年份及版次未猜补"},
            {"title": "缩略语词典", "author": "项布旺杰", "publisher": "四川民族出版社", "publication_year": None, "edition": None, "role": "作者补充的参考来源", "metadata_basis": "书名、作者与出版社由作者提供；保留作者姓名原字，年份及版次未猜补"},
        ],
        "metric_definitions": {
            "micro_character_accuracy": {
                "label": "字符编辑准确率（Character Accuracy，CA）",
                "formula": "1 - micro_cer",
                "display": "multiply by100 for percent",
                "scope": "edit-distance complement; may be negative; not character classification accuracy"
            },
            "exact_match": {
                "label": "整段还原准确率（完全匹配率，EM）",
                "formula": "fully_correct_segments / samples",
                "comparison": "prediction and reference equal codepoint by codepoint"
            },
            "micro_cer": {
                "label": "字符错误率CER",
                "formula": "sum(character_edit_distance) / sum(reference_codepoints)"
            },
            "line_error_rate": {
                "label": "片段错误率",
                "formula": "1 - exact_match"
            }
        },
        "models": {name: numeric_metrics(metrics) for name, metrics in evaluation["test_metrics"].items()},
        "selector_effect": evaluation["selector_effect"],
        "limitations": [
            "Parent-line groups do not establish dictionary-source or manuscript-level independence; publication years, editions, per-record source mapping and transcription/review records remain incomplete despite the author-provided titles, authors and publishers.",
            *evaluation["limitations"][1:],
        ],
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "records_sha256": manifest["records_sha256"],
        "source_file_sha256": manifest["source_file_sha256"],
        "restore_file_sha256": manifest["restore_file_sha256"],
        "sensitivity_runs": [],
    }
    baseline = run / "seq2seq/metrics.json"
    if baseline.exists():
        result = read(baseline)
        output["models"]["seq2seq"] = numeric_metrics(result["test"])
        output["seq2seq_checkpoint_epoch"] = result["checkpoint_epoch"]
        output["seq2seq_inference"] = "TF32 disabled, batched greedy; fixed dev-selected checkpoint"
    external = run / "tibnorm_external/metrics.json"
    if external.exists():
        result = read(external)
        output["models"]["tibnorm_external"] = numeric_metrics(result["test_metrics"])
        output["tibnorm_external"] = {key: result[key] for key in ["upstream_url", "upstream_commit", "upstream_doi", "config_flag", "tables_loaded_rows", "effective_abbreviation_keys", "pipeline_order", "resource_condition", "total_reference_characters", "total_character_distance", "caveats"]}
    for directory in args.sensitivity_dir:
        sensitivity_manifest = read(directory / 'corpus/split_manifest.json')
        if sensitivity_manifest['records_sha256'] != manifest['records_sha256'] or sensitivity_manifest['splits'] != manifest['splits']:
            raise ValueError('Sensitivity runs do not use the same frozen corpus and split.')
        result = read(directory / "evaluation/test_metrics.json")
        tagger = read(directory / "edit_tagger/metrics.json")
        output["sensitivity_runs"].append({
            "model_seed": tagger.get("training_args", {}).get("seed", directory.name),
            "same_split_seed": manifest["seed"],
            "models": {name: numeric_metrics(metrics) for name, metrics in result["test_metrics"].items() if name in ["contextual", "edit_tagger", "hybrid"]},
            "selector_effect": result["selector_effect"],
        })
    diagnostic = run / 'diagnostics/candidate_diagnostics.json'
    multiseed = run / 'multiseed_summary.json'
    if multiseed.exists():
        output['multiseed_summary'] = read(multiseed)
    if diagnostic.exists():
        result = read(diagnostic)
        output['posthoc_candidate_diagnostics'] = {
            'scope': result['scope'],
            'fallback_fraction': result['fallback_fraction'],
            'source_codepoints': result['source_codepoints'],
            'parameter_count': result['parameter_count'],
            'controls': {name: numeric_metrics(result[name]) for name in ['reference_driven_two_candidate_oracle_metrics', 'pure_neural_same_checkpoint_postprocessed_metrics', 'pure_rules_edit_alignment_postprocessed_metrics']},
            'constraints': result['constraints'],
        }
    ablation = run / 'no_neural_ablation/evaluation/test_metrics.json'
    if ablation.exists():
        result = read(ablation)
        output['posthoc_no_neural_ablation'] = {
            'scope': result['scope'], 'prior_test_observation': result['prior_test_observation'],
            'test_metrics': numeric_metrics(result['test_metrics']),
            'total_edit_distance': result['total_edit_distance'],
            'total_reference_codepoints': result['total_reference_codepoints'],
            'C': result['C'], 'threshold': result['threshold'],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"models": len(output["models"]), "sensitivity_runs": len(output["sensitivity_runs"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
