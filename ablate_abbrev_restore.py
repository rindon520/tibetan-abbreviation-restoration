from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from abbrev_restore import (
    AbbreviationDetector,
    build_examples,
    candidate_thresholds,
    evaluate_restoration,
    restore_unit,
)
from segment_tibetan import DEFAULT_SEGMENTER_DIR, segment_lines
from tibetan_syllable import build_syllable_vocabulary
from train_abbrev_restore import split_records, summarize_labels
from train_restore import configure_utf8_stdout, read_parallel_data
from train_syllable_corrector import terminal_shad
from train_transducer import apply_mapping


DetectionPolicy = Callable[[dict, dict, int], int]
RestorationPolicy = Callable[[str, dict], str]


def write_json(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_model(model_dir: Path) -> dict:
    model_path = model_dir / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return json.loads(model_path.read_text(encoding="utf-8"))


def detector_from_model(model: dict) -> AbbreviationDetector:
    detector_payload = model["detector"]
    return AbbreviationDetector(
        weights=dict(detector_payload["weights"]),
        restoration_table=dict(detector_payload.get("restoration_table", {})),
    )


def model_arg(model: dict, name: str, default: Any = None) -> Any:
    return model.get("args", {}).get(name, default)


def prepare_examples(data_dir: Path, model: dict, segmenter_dir: Path) -> dict[str, Any]:
    args = model.get("args", {})
    pairs = read_parallel_data(data_dir / "Source.txt", data_dir / "Restore.txt")
    max_samples = int(args.get("max_samples", 0) or 0)
    if max_samples > 0:
        pairs = pairs[:max_samples]

    prefer_external = bool(args.get("prefer_external_segmenter", True))
    allow_fallback = bool(args.get("allow_fallback_segmenter", True))
    timeout = float(args.get("segmenter_timeout", 10.0))

    source_segmented, source_status = segment_lines(
        [source for source, _ in pairs],
        segmenter_dir=segmenter_dir,
        prefer_external=prefer_external,
        allow_fallback=allow_fallback,
        timeout_seconds=timeout,
    )
    prefer_reference_external = prefer_external and source_status["mode"] == "external"
    reference_segmented, reference_status = segment_lines(
        [reference for _, reference in pairs],
        segmenter_dir=segmenter_dir,
        prefer_external=prefer_reference_external,
        allow_fallback=allow_fallback,
        timeout_seconds=timeout,
    )

    records = [
        {
            "source": source,
            "reference": reference,
            "source_segmented": source_seg,
            "reference_segmented": reference_seg,
        }
        for (source, reference), source_seg, reference_seg in zip(
            pairs, source_segmented, reference_segmented
        )
    ]
    train_records, valid_records, test_records = split_records(
        records,
        valid_ratio=float(args.get("valid_ratio", 0.1)),
        test_ratio=float(args.get("test_ratio", 0.05)),
        seed=int(args.get("seed", 42)),
    )
    max_ref_units_per_source = int(args.get("max_ref_units_per_source", 4))
    train_examples = build_examples(
        pairs=[(record["source"], record["reference"]) for record in train_records],
        source_segmented=[record["source_segmented"] for record in train_records],
        reference_segmented=[record["reference_segmented"] for record in train_records],
        max_ref_units_per_source=max_ref_units_per_source,
    )
    valid_examples = build_examples(
        pairs=[(record["source"], record["reference"]) for record in valid_records],
        source_segmented=[record["source_segmented"] for record in valid_records],
        reference_segmented=[record["reference_segmented"] for record in valid_records],
        max_ref_units_per_source=max_ref_units_per_source,
    )
    test_examples = build_examples(
        pairs=[(record["source"], record["reference"]) for record in test_records],
        source_segmented=[record["source_segmented"] for record in test_records],
        reference_segmented=[record["reference_segmented"] for record in test_records],
        max_ref_units_per_source=max_ref_units_per_source,
    )

    train_pairs = [(record["source"], record["reference"]) for record in train_records]
    return {
        "pairs": pairs,
        "records": records,
        "train_pairs": train_pairs,
        "examples": {
            "train": train_examples,
            "valid": valid_examples,
            "test": test_examples,
        },
        "segmenter": {
            "source": source_status,
            "reference": reference_status,
        },
        "label_summary": {
            "train": summarize_labels(train_examples),
            "valid": summarize_labels(valid_examples),
            "test": summarize_labels(test_examples),
        },
    }


def detection_metrics_from_predictions(
    examples: list[dict],
    predictions: list[list[int]],
) -> dict:
    tp = fp = tn = fn = 0
    for example, example_predictions in zip(examples, predictions):
        labels = [int(item["label"]) for item in example["alignment"]]
        if len(labels) != len(example_predictions):
            raise ValueError(
                "Prediction count does not match alignment count for example "
                f"{example.get('id', '<unknown>')}: {len(example_predictions)} vs {len(labels)}"
            )
        for prediction, label in zip(example_predictions, labels):
            prediction = int(prediction)
            if prediction and label:
                tp += 1
            elif prediction and not label:
                fp += 1
            elif not prediction and label:
                fn += 1
            else:
                tn += 1
    total = tp + fp + fn + tn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tokens": total,
        "positive_support": tp + fn,
        "predicted_positive": tp + fp,
        "accuracy": (tp + tn) / max(1, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def predict_with_policies(
    examples: list[dict],
    detect: DetectionPolicy,
    restore: RestorationPolicy,
    apply_terminal_shad: bool = True,
) -> tuple[list[dict], dict]:
    rows = []
    detection_predictions: list[list[int]] = []
    for example in examples:
        pieces = []
        detected = []
        example_predictions = []
        for index, unit in enumerate(example["source_units"]):
            item = example["alignment"][index]
            is_abbrev = int(detect(example, item, index))
            example_predictions.append(is_abbrev)
            if is_abbrev:
                output = restore(unit, item)
                detected.append({"index": index, "source_unit": unit, "prediction": output})
            else:
                output = unit
            pieces.append(output)
        prediction = "".join(pieces)
        if apply_terminal_shad:
            prediction = terminal_shad(prediction)
        rows.append(
            {
                "source": example["source"],
                "reference": example["reference"],
                "prediction": prediction,
                "detected": detected,
            }
        )
        detection_predictions.append(example_predictions)
    return rows, detection_metrics_from_predictions(examples, detection_predictions)


def source_rows(examples: list[dict], apply_terminal: bool = False) -> list[dict]:
    rows = []
    for example in examples:
        prediction = example["source"]
        if apply_terminal:
            prediction = terminal_shad("".join(example["source_units"]))
        rows.append(
            {
                "source": example["source"],
                "reference": example["reference"],
                "prediction": prediction,
                "detected": [],
            }
        )
    return rows


def flatten_variant_metrics(
    variant: str,
    split: str,
    description: str,
    restoration: dict,
    detection: dict | None,
    threshold: float | None = None,
    max_edit_distance: int | None = None,
) -> dict:
    row = {
        "variant": variant,
        "split": split,
        "description": description,
        "threshold": threshold,
        "max_edit_distance": max_edit_distance,
        "detection_accuracy": None,
        "detection_precision": None,
        "detection_recall": None,
        "detection_f1": None,
        "detection_predicted_positive": None,
        "restore_exact_match": restoration["exact_match"],
        "restore_micro_cer": restoration["micro_cer"],
        "restore_micro_character_accuracy": restoration["micro_character_accuracy"],
        "restore_micro_tibetan_unit_error_rate": restoration[
            "micro_tibetan_unit_error_rate"
        ],
        "restore_normalized_edit_similarity_mean": restoration[
            "normalized_edit_similarity_mean"
        ],
        "restore_length_mae": restoration["length_mae"],
        "edit_substitutions": restoration["edit_substitutions"],
        "edit_insertions": restoration["edit_insertions"],
        "edit_deletions": restoration["edit_deletions"],
    }
    if detection is not None:
        row.update(
            {
                "detection_accuracy": detection["accuracy"],
                "detection_precision": detection["precision"],
                "detection_recall": detection["recall"],
                "detection_f1": detection["f1"],
                "detection_predicted_positive": detection["predicted_positive"],
            }
        )
    return row


def add_comparison_fields(rows: list[dict]) -> list[dict]:
    by_split: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_split.setdefault(row["split"], {})[row["variant"]] = row

    enriched = []
    for row in rows:
        split_rows = by_split[row["split"]]
        full = split_rows.get("full_two_stage")
        identity = split_rows.get("identity_source")
        output = dict(row)
        if full is not None:
            output["cer_delta_vs_full"] = (
                row["restore_micro_cer"] - full["restore_micro_cer"]
            )
            output["exact_delta_vs_full"] = (
                row["restore_exact_match"] - full["restore_exact_match"]
            )
        else:
            output["cer_delta_vs_full"] = None
            output["exact_delta_vs_full"] = None
        if identity is not None and identity["restore_micro_cer"]:
            output["cer_delta_vs_identity"] = (
                row["restore_micro_cer"] - identity["restore_micro_cer"]
            )
            output["cer_relative_reduction_vs_identity"] = (
                identity["restore_micro_cer"] - row["restore_micro_cer"]
            ) / identity["restore_micro_cer"]
        else:
            output["cer_delta_vs_identity"] = None
            output["cer_relative_reduction_vs_identity"] = None
        enriched.append(output)
    return enriched


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    separator = ["---"] * len(header)
    all_rows = [header, separator] + rows[1:]
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows)


def build_ablation_report(
    rows: list[dict],
    threshold_rows: list[dict],
    metadata: dict,
) -> str:
    test_rows = [row for row in rows if row["split"] == "test"]
    valid_rows = [row for row in rows if row["split"] == "valid"]

    def main_table(selected_rows: list[dict]) -> str:
        table_rows = [
            [
                "变体",
                "说明",
                "Exact",
                "CER",
                "CharAcc",
                "Det-F1",
                "ΔCER(相对完整)",
                "相对源句降幅",
            ]
        ]
        for row in selected_rows:
            table_rows.append(
                [
                    row["variant"],
                    row["description"],
                    format_percent(row["restore_exact_match"]),
                    format_percent(row["restore_micro_cer"]),
                    format_percent(row["restore_micro_character_accuracy"]),
                    format_percent(row["detection_f1"]),
                    format_float(row.get("cer_delta_vs_full"), 4),
                    format_percent(row.get("cer_relative_reduction_vs_identity")),
                ]
            )
        return markdown_table(table_rows)

    threshold_test = [row for row in threshold_rows if row["split"] == "test"]
    threshold_table_rows = [
        [
            "阈值",
            "编辑距离",
            "Det-P",
            "Det-R",
            "Det-F1",
            "Exact",
            "CER",
            "CharAcc",
        ]
    ]
    for row in threshold_test:
        threshold_table_rows.append(
            [
                format_float(row["threshold"], 4),
                str(row["max_edit_distance"]),
                format_percent(row["detection_precision"]),
                format_percent(row["detection_recall"]),
                format_percent(row["detection_f1"]),
                format_percent(row["restore_exact_match"]),
                format_percent(row["restore_micro_cer"]),
                format_percent(row["restore_micro_character_accuracy"]),
            ]
        )

    lines = [
        "# 藏文缩略词识别与还原消融实验报告",
        "",
        "## 实验设置",
        "",
        f"- 数据目录：`{metadata.get('data_dir')}`",
        f"- 模型目录：`{metadata.get('model_dir')}`",
        f"- 训练/验证/测试样本数：{metadata.get('train_size')} / {metadata.get('valid_size')} / {metadata.get('test_size')}",
        f"- 最优阈值：{format_float(metadata.get('best_threshold'), 6)}",
        f"- 最优音节编辑距离：{metadata.get('best_max_edit_distance')}",
        f"- 分词模式：source={metadata.get('source_segmenter_mode')}，reference={metadata.get('reference_segmenter_mode')}",
        "",
        "说明：检测指标基于自动对齐产生的伪标签；还原指标以完整句级预测与参考还原文本计算。",
        "",
        "## 测试集主消融结果",
        "",
        main_table(test_rows),
        "",
        "## 验证集主消融结果",
        "",
        main_table(valid_rows),
        "",
        "## 阈值敏感性（测试集）",
        "",
        markdown_table(threshold_table_rows),
        "",
        "## 读数建议",
        "",
        "- `identity_source` 表示不进行缩略词识别与还原，是源句复制基线。",
        "- `full_two_stage` 表示检测器、还原表、字符映射、音节词表纠错和句末规范化全部启用。",
        "- `oracle_detector_*` 使用自动对齐标签作为检测上界，用来估计检测误差对最终还原的影响。",
        "- `oracle_restore` 使用自动对齐参考片段替换阳性单元，是当前伪标签对齐框架下的近似上界。",
    ]
    return "\n".join(lines) + "\n"


def build_variant_specs(
    detector: AbbreviationDetector,
    threshold: float,
    restoration_table: dict[str, str],
    char_mapping: dict[str, str],
    syllable_vocab: dict,
    max_edit_distance: int,
    max_syllable_chars: int,
) -> list[dict]:
    def full_restore(unit: str, _item: dict) -> str:
        return restore_unit(
            unit,
            restoration_table=restoration_table,
            char_mapping=char_mapping,
            syllable_vocab=syllable_vocab,
            max_edit_distance=max_edit_distance,
            max_syllable_chars=max_syllable_chars,
        )

    def edit0_restore(unit: str, _item: dict) -> str:
        return restore_unit(
            unit,
            restoration_table=restoration_table,
            char_mapping=char_mapping,
            syllable_vocab=syllable_vocab,
            max_edit_distance=0,
            max_syllable_chars=max_syllable_chars,
        )

    def table_only_restore(unit: str, _item: dict) -> str:
        return restoration_table.get(unit, unit)

    def no_table_restore(unit: str, _item: dict) -> str:
        return restore_unit(
            unit,
            restoration_table={},
            char_mapping=char_mapping,
            syllable_vocab=syllable_vocab,
            max_edit_distance=max_edit_distance,
            max_syllable_chars=max_syllable_chars,
        )

    def raw_mapping_restore(unit: str, _item: dict) -> str:
        if unit in restoration_table:
            return restoration_table[unit]
        return apply_mapping(unit, char_mapping)

    def oracle_restore(unit: str, item: dict) -> str:
        return item["reference_text"] or unit

    detector_policy = lambda _example, item, _index: detector.predict_unit(
        item["source_unit"], threshold=threshold
    )
    all_positive = lambda _example, _item, _index: 1
    all_negative = lambda _example, _item, _index: 0
    oracle_detector = lambda _example, item, _index: int(item["label"])

    return [
        {
            "variant": "no_detection_with_terminal_shad",
            "description": "不检测不还原，仅执行句末 ། 规范化",
            "detect": all_negative,
            "restore": full_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "restore_all_units",
            "description": "取消检测器，所有音节单元均尝试还原",
            "detect": all_positive,
            "restore": full_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "detector_table_only",
            "description": "保留检测器，仅使用训练集还原表，未登录项复制",
            "detect": detector_policy,
            "restore": table_only_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "detector_no_table",
            "description": "保留检测器，移除还原表，仅用字符映射与音节纠错",
            "detect": detector_policy,
            "restore": no_table_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "detector_no_syllable_correction",
            "description": "保留检测器与还原表，未登录项仅用原始字符映射",
            "detect": detector_policy,
            "restore": raw_mapping_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "detector_edit_distance_0",
            "description": "保留完整结构，但音节词表纠错编辑距离设为 0",
            "detect": detector_policy,
            "restore": edit0_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "full_two_stage_no_terminal_shad",
            "description": "完整两阶段模型，但去掉句末 ། 规范化",
            "detect": detector_policy,
            "restore": full_restore,
            "apply_terminal_shad": False,
        },
        {
            "variant": "full_two_stage",
            "description": "检测器+还原表+字符映射+音节纠错+句末规范化",
            "detect": detector_policy,
            "restore": full_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "oracle_detector_full_restorer",
            "description": "使用伪标签作检测上界，还原器保持完整",
            "detect": oracle_detector,
            "restore": full_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "oracle_detector_table_only",
            "description": "使用伪标签作检测上界，仅使用还原表",
            "detect": oracle_detector,
            "restore": table_only_restore,
            "apply_terminal_shad": True,
        },
        {
            "variant": "oracle_restore",
            "description": "使用伪标签检测，并直接替换为对齐参考片段",
            "detect": oracle_detector,
            "restore": oracle_restore,
            "apply_terminal_shad": True,
        },
    ]


def evaluate_identity(examples: list[dict], split: str) -> tuple[list[dict], dict]:
    rows = source_rows(examples, apply_terminal=False)
    return rows, flatten_variant_metrics(
        variant="identity_source",
        split=split,
        description="源句复制基线，不执行检测、还原或句末规范化",
        restoration=evaluate_restoration(rows),
        detection=None,
    )


def evaluate_variants(
    examples_by_split: dict[str, list[dict]],
    splits: list[str],
    variant_specs: list[dict],
    threshold: float,
    max_edit_distance: int,
) -> tuple[list[dict], dict[str, list[dict]]]:
    metric_rows = []
    prediction_rows: dict[str, list[dict]] = {}
    for split in splits:
        examples = examples_by_split[split]
        rows, metric = evaluate_identity(examples, split=split)
        metric_rows.append(metric)
        prediction_rows[f"{split}:identity_source"] = rows
        for spec in variant_specs:
            rows, detection = predict_with_policies(
                examples,
                detect=spec["detect"],
                restore=spec["restore"],
                apply_terminal_shad=bool(spec.get("apply_terminal_shad", True)),
            )
            restoration = evaluate_restoration(rows)
            metric_rows.append(
                flatten_variant_metrics(
                    variant=spec["variant"],
                    split=split,
                    description=spec["description"],
                    restoration=restoration,
                    detection=detection,
                    threshold=threshold,
                    max_edit_distance=max_edit_distance,
                )
            )
            prediction_rows[f"{split}:{spec['variant']}"] = rows
    return add_comparison_fields(metric_rows), prediction_rows


def evaluate_threshold_sweep(
    examples_by_split: dict[str, list[dict]],
    splits: list[str],
    detector: AbbreviationDetector,
    restoration_table: dict[str, str],
    char_mapping: dict[str, str],
    syllable_vocab: dict,
    thresholds: list[float],
    max_edit_distance: int,
    max_syllable_chars: int,
) -> list[dict]:
    rows = []
    for split in splits:
        examples = examples_by_split[split]
        for threshold in thresholds:
            detect = lambda _example, item, _index, threshold=threshold: detector.predict_unit(
                item["source_unit"], threshold=threshold
            )

            def restore(unit: str, _item: dict) -> str:
                return restore_unit(
                    unit,
                    restoration_table=restoration_table,
                    char_mapping=char_mapping,
                    syllable_vocab=syllable_vocab,
                    max_edit_distance=max_edit_distance,
                    max_syllable_chars=max_syllable_chars,
                )

            pred_rows, detection = predict_with_policies(examples, detect, restore)
            restoration = evaluate_restoration(pred_rows)
            rows.append(
                flatten_variant_metrics(
                    variant="threshold_sweep",
                    split=split,
                    description="固定还原器，扫描检测阈值",
                    restoration=restoration,
                    detection=detection,
                    threshold=threshold,
                    max_edit_distance=max_edit_distance,
                )
            )
    return rows


def evaluate_edit_distance_sweep(
    examples_by_split: dict[str, list[dict]],
    splits: list[str],
    detector: AbbreviationDetector,
    threshold: float,
    restoration_table: dict[str, str],
    char_mapping: dict[str, str],
    syllable_vocab: dict,
    edit_distances: list[int],
    max_syllable_chars: int,
) -> list[dict]:
    rows = []
    for split in splits:
        examples = examples_by_split[split]
        for max_edit_distance in edit_distances:
            detect = lambda _example, item, _index: detector.predict_unit(
                item["source_unit"], threshold=threshold
            )

            def restore(unit: str, _item: dict, max_edit_distance=max_edit_distance) -> str:
                return restore_unit(
                    unit,
                    restoration_table=restoration_table,
                    char_mapping=char_mapping,
                    syllable_vocab=syllable_vocab,
                    max_edit_distance=max_edit_distance,
                    max_syllable_chars=max_syllable_chars,
                )

            pred_rows, detection = predict_with_policies(examples, detect, restore)
            restoration = evaluate_restoration(pred_rows)
            rows.append(
                flatten_variant_metrics(
                    variant="edit_distance_sweep",
                    split=split,
                    description="固定检测器，扫描音节词表纠错编辑距离",
                    restoration=restoration,
                    detection=detection,
                    threshold=threshold,
                    max_edit_distance=max_edit_distance,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_thresholds(
    detector: AbbreviationDetector,
    valid_examples: list[dict],
    model: dict,
    max_threshold_candidates: int,
) -> list[float]:
    thresholds = {float(model["best_threshold"])}
    for item in model.get("tuning", []):
        thresholds.add(float(item["threshold"]))
    if max_threshold_candidates <= 0:
        thresholds.update(candidate_thresholds(detector, valid_examples))
    elif len(thresholds) < max_threshold_candidates:
        candidates = candidate_thresholds(detector, valid_examples)
        if candidates:
            step = max(1, round(len(candidates) / max_threshold_candidates))
            thresholds.update(candidates[::step])
    return sorted(thresholds)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run detailed ablation experiments for Tibetan abbreviation restoration."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_segments_trusted"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("runs/tibetan_abbrev_restore_segments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/tibetan_abbrev_restore_ablation"),
    )
    parser.add_argument("--segmenter-dir", type=Path, default=None)
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--max-threshold-candidates", type=int, default=12)
    parser.add_argument("--edit-distance-sweep", default="0,1,2")
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def main():
    configure_utf8_stdout()
    args = parse_args()
    model = load_model(args.model_dir)
    segmenter_dir = args.segmenter_dir or Path(
        model_arg(model, "segmenter_dir", DEFAULT_SEGMENTER_DIR)
    )
    print(
        json.dumps(
            {
                "stage": "load",
                "data_dir": str(args.data_dir),
                "model_dir": str(args.model_dir),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    prepared = prepare_examples(args.data_dir, model=model, segmenter_dir=segmenter_dir)
    examples_by_split = prepared["examples"]
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    for split in splits:
        if split not in examples_by_split:
            raise ValueError(f"Unknown split: {split}")

    detector = detector_from_model(model)
    restoration_table = dict(model["restoration_table"])
    char_mapping = dict(model["char_mapping"])
    syllable_vocab = build_syllable_vocabulary(prepared["train_pairs"])
    threshold = float(model["best_threshold"])
    max_edit_distance = int(model["best_max_edit_distance"])
    max_syllable_chars = int(model_arg(model, "max_syllable_chars", 8))

    print(
        json.dumps(
            {
                "stage": "prepared",
                "segmenter": prepared["segmenter"],
                "labels": prepared["label_summary"],
                "restoration_table_size": len(restoration_table),
                "char_mapping_size": len(char_mapping),
                "syllable_vocab_size": len(syllable_vocab["counts"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    variant_specs = build_variant_specs(
        detector=detector,
        threshold=threshold,
        restoration_table=restoration_table,
        char_mapping=char_mapping,
        syllable_vocab=syllable_vocab,
        max_edit_distance=max_edit_distance,
        max_syllable_chars=max_syllable_chars,
    )
    metric_rows, prediction_rows = evaluate_variants(
        examples_by_split=examples_by_split,
        splits=splits,
        variant_specs=variant_specs,
        threshold=threshold,
        max_edit_distance=max_edit_distance,
    )
    print(json.dumps({"stage": "variants_done", "rows": len(metric_rows)}, ensure_ascii=False), flush=True)

    thresholds = select_thresholds(
        detector,
        valid_examples=examples_by_split["valid"],
        model=model,
        max_threshold_candidates=args.max_threshold_candidates,
    )
    threshold_rows = evaluate_threshold_sweep(
        examples_by_split=examples_by_split,
        splits=splits,
        detector=detector,
        restoration_table=restoration_table,
        char_mapping=char_mapping,
        syllable_vocab=syllable_vocab,
        thresholds=thresholds,
        max_edit_distance=max_edit_distance,
        max_syllable_chars=max_syllable_chars,
    )
    print(
        json.dumps(
            {"stage": "threshold_sweep_done", "thresholds": len(thresholds)},
            ensure_ascii=False,
        ),
        flush=True,
    )

    edit_distance_rows = evaluate_edit_distance_sweep(
        examples_by_split=examples_by_split,
        splits=splits,
        detector=detector,
        threshold=threshold,
        restoration_table=restoration_table,
        char_mapping=char_mapping,
        syllable_vocab=syllable_vocab,
        edit_distances=parse_int_list(args.edit_distance_sweep),
        max_syllable_chars=max_syllable_chars,
    )
    print(
        json.dumps(
            {"stage": "edit_distance_sweep_done", "rows": len(edit_distance_rows)},
            ensure_ascii=False,
        ),
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data_dir": str(args.data_dir),
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "best_threshold": threshold,
        "best_max_edit_distance": max_edit_distance,
        "train_size": len(examples_by_split["train"]),
        "valid_size": len(examples_by_split["valid"]),
        "test_size": len(examples_by_split["test"]),
        "source_segmenter_mode": prepared["segmenter"]["source"]["mode"],
        "reference_segmenter_mode": prepared["segmenter"]["reference"]["mode"],
        "label_summary": prepared["label_summary"],
        "thresholds": thresholds,
    }
    write_json(
        args.output_dir / "ablation_metrics.json",
        {
            "metadata": metadata,
            "variants": metric_rows,
            "threshold_sweep": threshold_rows,
            "edit_distance_sweep": edit_distance_rows,
        },
    )
    write_csv(args.output_dir / "ablation_metrics.csv", metric_rows)
    write_csv(args.output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(args.output_dir / "edit_distance_sweep.csv", edit_distance_rows)
    report = build_ablation_report(metric_rows, threshold_rows, metadata=metadata)
    (args.output_dir / "ablation_report.md").write_text(report, encoding="utf-8")

    for key, rows in prediction_rows.items():
        split, variant = key.split(":", maxsplit=1)
        if split == "test" and variant in {
            "identity_source",
            "full_two_stage",
            "oracle_detector_full_restorer",
            "oracle_restore",
        }:
            write_jsonl(args.output_dir / f"{split}_{variant}_predictions.jsonl", rows)
            write_json(
                args.output_dir / f"{split}_{variant}_samples.json",
                rows[: args.sample_count],
            )

    test_summary = [
        row
        for row in metric_rows
        if row["split"] == "test"
        and row["variant"]
        in {
            "identity_source",
            "full_two_stage",
            "detector_table_only",
            "detector_no_syllable_correction",
            "oracle_detector_full_restorer",
            "oracle_restore",
        }
    ]
    print(
        json.dumps(
            {
                "stage": "done",
                "output_dir": str(args.output_dir),
                "test_summary": test_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
