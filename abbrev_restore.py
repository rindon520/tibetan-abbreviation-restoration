from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from evaluate_restore import compute_metrics
from tibetan_syllable import (
    apply_syllable_vocabulary_correction,
    is_likely_tibetan_syllable,
    is_subjoined_consonant,
    split_syllable_units,
)
from train_restore import levenshtein
from train_syllable_corrector import terminal_shad
from train_transducer import apply_mapping


def normalize_units_text(units: list[str]) -> str:
    return "".join(units).replace(" ", "")


def tokenize_segmented_line(line: str) -> list[str]:
    tokens = [part for part in line.split() if part]
    if tokens:
        return tokens
    return [unit.core + unit.delimiter for unit in split_syllable_units(line)]


def _alignment_cost(source_unit: str, reference_units: list[str]) -> float:
    reference_text = normalize_units_text(reference_units)
    source_text = source_unit.replace(" ", "")
    if reference_text == source_text:
        return 0.0
    if not reference_text:
        return 2.0 + len(source_text) * 0.1
    distance = levenshtein(source_text, reference_text)
    length_penalty = abs(len(reference_text) - len(source_text)) * 0.05
    expansion_bonus = -0.15 if len(reference_text) > len(source_text) else 0.0
    return 0.75 + distance * 0.18 + length_penalty + expansion_bonus


def align_segment_units(
    source_units: list[str],
    reference_units: list[str],
    max_ref_units_per_source: int = 4,
) -> list[dict]:
    source_count = len(source_units)
    reference_count = len(reference_units)
    inf = 10**9
    dp: list[list[tuple[float, int | None]]] = [
        [(inf, None) for _ in range(reference_count + 1)] for __ in range(source_count + 1)
    ]
    dp[0][0] = (0.0, None)

    for source_index, source_unit in enumerate(source_units):
        for ref_index in range(reference_count + 1):
            base_cost, _ = dp[source_index][ref_index]
            if base_cost >= inf:
                continue
            max_take = min(max_ref_units_per_source, reference_count - ref_index)
            for take in range(0, max_take + 1):
                ref_slice = reference_units[ref_index : ref_index + take]
                cost = _alignment_cost(source_unit, ref_slice)
                next_ref = ref_index + take
                new_cost = base_cost + cost
                if new_cost < dp[source_index + 1][next_ref][0]:
                    dp[source_index + 1][next_ref] = (new_cost, take)

    if dp[source_count][reference_count][0] >= inf:
        return [
            {
                "source_index": index,
                "source_unit": source_unit,
                "reference_units": [],
                "reference_text": "",
                "label": 1,
            }
            for index, source_unit in enumerate(source_units)
        ]

    alignment = []
    source_index = source_count
    ref_index = reference_count
    while source_index > 0:
        take = dp[source_index][ref_index][1]
        if take is None:
            take = 0
        start = ref_index - take
        source_unit = source_units[source_index - 1]
        ref_units = reference_units[start:ref_index]
        ref_text = normalize_units_text(ref_units)
        src_text = source_unit.replace(" ", "")
        alignment.append(
            {
                "source_index": source_index - 1,
                "source_unit": source_unit,
                "reference_units": ref_units,
                "reference_text": ref_text,
                "label": int(ref_text != src_text),
            }
        )
        ref_index = start
        source_index -= 1
    alignment.reverse()
    return alignment


def build_examples(
    pairs: list[tuple[str, str]],
    source_segmented: list[str],
    reference_segmented: list[str],
    max_ref_units_per_source: int = 4,
) -> list[dict]:
    examples = []
    for row_id, ((source, reference), source_seg, reference_seg) in enumerate(
        zip(pairs, source_segmented, reference_segmented)
    ):
        source_units = tokenize_segmented_line(source_seg)
        reference_units = tokenize_segmented_line(reference_seg)
        alignment = align_segment_units(
            source_units,
            reference_units,
            max_ref_units_per_source=max_ref_units_per_source,
        )
        labels = [item["label"] for item in alignment]
        examples.append(
            {
                "id": row_id,
                "source": source,
                "reference": reference,
                "source_segmented": source_seg,
                "reference_segmented": reference_seg,
                "source_units": source_units,
                "reference_units": reference_units,
                "alignment": alignment,
                "labels": labels,
            }
        )
    return examples


def build_restoration_table(examples: list[dict], min_count: int = 1) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for example in examples:
        for item in example["alignment"]:
            source_unit = item["source_unit"]
            reference_text = item["reference_text"]
            if item["label"] and reference_text:
                counts[source_unit][reference_text] += 1
    return {
        source_unit: outputs.most_common(1)[0][0]
        for source_unit, outputs in counts.items()
        if outputs.most_common(1)[0][1] >= min_count
    }


def _shape_features(unit: str, restoration_table: dict[str, str] | None = None) -> list[str]:
    restoration_table = restoration_table or {}
    core = unit.rstrip("\u0f0b\u0f0d\u0f0e")
    features = [
        "bias",
        f"unit={unit}",
        f"core={core}",
        f"len={min(len(unit), 12)}",
        f"has_table={int(unit in restoration_table)}",
        f"likely_syllable={int(is_likely_tibetan_syllable(core))}",
        f"has_subjoined={int(any(is_subjoined_consonant(char) for char in unit))}",
        f"ends_tsheg={int(unit.endswith(chr(0x0F0B)))}",
        f"ends_shad={int(unit.endswith(chr(0x0F0D)) or unit.endswith(chr(0x0F0E)))}",
    ]
    if unit:
        features.append(f"prefix={unit[:1]}")
        features.append(f"suffix={unit[-1:]}")
    if len(unit) >= 2:
        features.append(f"prefix2={unit[:2]}")
        features.append(f"suffix2={unit[-2:]}")
    return features


@dataclass
class AbbreviationDetector:
    weights: dict[str, float]
    restoration_table: dict[str, str]

    def score_unit(self, unit: str) -> float:
        score = sum(self.weights.get(feature, 0.0) for feature in self.features(unit))
        if unit in self.restoration_table:
            score += 2.0
        else:
            score -= 0.25
        return score

    def predict_unit(self, unit: str, threshold: float = 0.0) -> int:
        return int(self.score_unit(unit) >= threshold)

    def features(self, unit: str) -> list[str]:
        return _shape_features(unit, self.restoration_table)

    def to_json(self) -> dict:
        return {
            "model_type": "averaged_perceptron_abbreviation_detector",
            "weights": self.weights,
            "restoration_table": self.restoration_table,
        }


def _iter_labeled_units(examples: list[dict]):
    for example in examples:
        for item in example["alignment"]:
            yield item["source_unit"], int(item["label"])


def train_detector(
    examples: list[dict],
    restoration_table: dict[str, str] | None = None,
    epochs: int = 5,
    seed: int = 42,
) -> AbbreviationDetector:
    restoration_table = restoration_table or {}
    items = list(_iter_labeled_units(examples))
    weights: dict[str, float] = defaultdict(float)
    totals: dict[str, float] = defaultdict(float)
    timestamps: dict[str, int] = defaultdict(int)
    step = 0
    rng = random.Random(seed)

    def update_total(feature: str):
        totals[feature] += (step - timestamps[feature]) * weights[feature]
        timestamps[feature] = step

    for _ in range(epochs):
        rng.shuffle(items)
        for unit, label in items:
            step += 1
            features = _shape_features(unit, restoration_table)
            score = sum(weights[feature] for feature in features)
            prediction = int(score >= 0.0)
            if prediction == label:
                continue
            delta = 1.0 if label else -1.0
            for feature in features:
                update_total(feature)
                weights[feature] += delta

    averaged = {}
    for feature in list(weights):
        update_total(feature)
        averaged_weight = weights[feature] - totals[feature] / max(1, step)
        if abs(averaged_weight) > 1e-9:
            averaged[feature] = averaged_weight
    return AbbreviationDetector(dict(sorted(averaged.items())), dict(restoration_table))


def evaluate_detection(
    examples: list[dict],
    detector: AbbreviationDetector,
    threshold: float = 0.0,
) -> dict:
    tp = fp = tn = fn = 0
    for unit, label in _iter_labeled_units(examples):
        prediction = detector.predict_unit(unit, threshold=threshold)
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


def candidate_thresholds(detector: AbbreviationDetector, examples: list[dict]) -> list[float]:
    scores = sorted({detector.score_unit(unit) for unit, _ in _iter_labeled_units(examples)})
    if not scores:
        return [0.0]
    thresholds = {0.0, scores[0] - 1e-6, scores[-1] + 1e-6}
    thresholds.update(scores)
    for left, right in zip(scores, scores[1:]):
        thresholds.add((left + right) / 2)
    return sorted(thresholds)


def restore_unit(
    unit: str,
    restoration_table: dict[str, str],
    char_mapping: dict[str, str],
    syllable_vocab: dict,
    max_edit_distance: int,
    max_syllable_chars: int,
) -> str:
    if unit in restoration_table:
        return restoration_table[unit]
    mapped = apply_mapping(unit, char_mapping)
    return apply_syllable_vocabulary_correction(
        mapped,
        syllable_vocab,
        max_edit_distance=max_edit_distance,
        max_syllable_chars=max_syllable_chars,
    )


def predict_restoration_rows(
    examples: list[dict],
    detector: AbbreviationDetector,
    threshold: float,
    restoration_table: dict[str, str],
    char_mapping: dict[str, str],
    syllable_vocab: dict,
    max_edit_distance: int = 1,
    max_syllable_chars: int = 8,
) -> list[dict]:
    rows = []
    for example in examples:
        pieces = []
        detected = []
        for index, unit in enumerate(example["source_units"]):
            is_abbrev = detector.predict_unit(unit, threshold=threshold)
            if is_abbrev:
                output = restore_unit(
                    unit,
                    restoration_table=restoration_table,
                    char_mapping=char_mapping,
                    syllable_vocab=syllable_vocab,
                    max_edit_distance=max_edit_distance,
                    max_syllable_chars=max_syllable_chars,
                )
                detected.append({"index": index, "source_unit": unit, "prediction": output})
            else:
                output = unit
            pieces.append(output)
        prediction = terminal_shad("".join(pieces))
        rows.append(
            {
                "source": example["source"],
                "reference": example["reference"],
                "prediction": prediction,
                "detected": detected,
            }
        )
    return rows


def evaluate_restoration(rows: list[dict]) -> dict:
    return compute_metrics(rows)


def detector_payload(detector: AbbreviationDetector) -> dict:
    payload = detector.to_json()
    payload["weights"] = dict(sorted(payload["weights"].items()))
    return payload
