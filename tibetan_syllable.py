from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

from train_restore import levenshtein


TSHEG = "\u0f0b"
SHAD = "\u0f0d"
NYIS_SHAD = "\u0f0e"
TERMINAL_MARKS = {SHAD, NYIS_SHAD}
SYLLABLE_SEPARATORS = {TSHEG, SHAD, NYIS_SHAD, " ", "\t", "\r", "\n"}

VOWEL_SIGNS = {
    "\u0f71",
    "\u0f72",
    "\u0f73",
    "\u0f74",
    "\u0f75",
    "\u0f76",
    "\u0f77",
    "\u0f78",
    "\u0f79",
    "\u0f7a",
    "\u0f7b",
    "\u0f7c",
    "\u0f7d",
    "\u0f80",
}


@dataclass(frozen=True)
class SyllableUnit:
    core: str
    delimiter: str = ""


def is_base_consonant(char: str) -> bool:
    return "\u0f40" <= char <= "\u0f6c"


def is_subjoined_consonant(char: str) -> bool:
    return "\u0f90" <= char <= "\u0fbc"


def is_tibetan_digit(char: str) -> bool:
    return "\u0f20" <= char <= "\u0f29"


def is_combining_mark(char: str) -> bool:
    return char in VOWEL_SIGNS or "\u0f71" <= char <= "\u0f84"


def is_syllable_separator(char: str) -> bool:
    return char in SYLLABLE_SEPARATORS


def split_syllable_units(text: str) -> list[SyllableUnit]:
    units: list[SyllableUnit] = []
    current: list[str] = []
    pending_terminal = ""
    for char in text:
        if char == TSHEG:
            if current:
                units.append(SyllableUnit("".join(current), TSHEG))
                current = []
            continue
        if char in TERMINAL_MARKS:
            if current:
                units.append(SyllableUnit("".join(current), pending_terminal + char))
                current = []
                pending_terminal = ""
            elif units:
                previous = units[-1]
                units[-1] = SyllableUnit(previous.core, previous.delimiter + char)
            else:
                pending_terminal += char
            continue
        if char.isspace():
            if current:
                units.append(SyllableUnit("".join(current), " "))
                current = []
            continue
        current.append(char)
    if current:
        units.append(SyllableUnit("".join(current), pending_terminal))
    return units


def strip_syllable_separators(text: str) -> str:
    return "".join(char for char in text if not is_syllable_separator(char))


def is_likely_tibetan_syllable(core: str) -> bool:
    if not core:
        return False
    if core[0] in VOWEL_SIGNS or is_subjoined_consonant(core[0]):
        return False
    consonants = 0
    vowels = 0
    has_tibetan = False
    for char in core:
        if is_base_consonant(char) or is_subjoined_consonant(char):
            consonants += 1
            has_tibetan = True
        elif is_tibetan_digit(char):
            has_tibetan = True
        elif char in VOWEL_SIGNS:
            vowels += 1
            has_tibetan = True
        elif "\u0f00" <= char <= "\u0fff":
            has_tibetan = True
    return has_tibetan and 1 <= consonants <= 6 and vowels <= 3


def build_syllable_vocabulary(pairs) -> dict:
    counts = Counter()
    delimiters = defaultdict(Counter)
    for _, reference in pairs:
        for unit in split_syllable_units(reference):
            if not unit.core:
                continue
            counts[unit.core] += 1
            delimiters[unit.core][unit.delimiter or TSHEG] += 1
    by_length = defaultdict(list)
    for core in counts:
        by_length[len(core)].append(core)
    return {
        "counts": counts,
        "delimiters": delimiters,
        "by_length": dict(by_length),
        "total": sum(counts.values()),
    }


def _candidate_cores(substring: str, vocab: dict, max_edit_distance: int) -> list[tuple[str, int]]:
    cache = vocab.setdefault("_candidate_cache", {})
    cache_key = (substring, max_edit_distance)
    if cache_key in cache:
        return cache[cache_key]

    counts: Counter = vocab["counts"]
    if substring in counts:
        cache[cache_key] = [(substring, 0)]
        return cache[cache_key]

    candidates: list[tuple[str, int]] = []
    if max_edit_distance <= 0:
        cache[cache_key] = candidates
        return candidates

    by_length = vocab["by_length"]
    min_len = len(substring)
    max_len = len(substring) + max_edit_distance
    for length in range(min_len, max_len + 1):
        for core in by_length.get(length, []):
            distance = levenshtein(substring, core)
            if distance <= max_edit_distance:
                candidates.append((core, distance))
    candidates.sort(key=lambda item: (item[1], -counts[item[0]], item[0]))
    cache[cache_key] = candidates[:12]
    return cache[cache_key]


def _core_cost(core: str, distance: int, vocab: dict) -> float:
    counts: Counter = vocab["counts"]
    frequency = counts.get(core, 0)
    short_penalty = 0.35 if len(core) == 1 and not is_tibetan_digit(core) else 0.0
    structure_penalty = 0.0 if is_likely_tibetan_syllable(core) else 0.25
    unknown_penalty = 2.5 + len(core) * 0.08 if frequency == 0 else 0.0
    frequency_bonus = 0.04 * log(frequency + 1)
    return (
        0.12
        + short_penalty
        + structure_penalty
        + unknown_penalty
        + distance * 0.55
        - frequency_bonus
    )


def segment_core_with_vocabulary(
    core_text: str,
    vocab: dict,
    max_edit_distance: int = 1,
    max_syllable_chars: int = 8,
) -> list[str]:
    if not core_text:
        return []

    n_chars = len(core_text)
    inf = 10**9
    dp: list[tuple[float, int, str] | None] = [None] * (n_chars + 1)
    dp[0] = (0.0, -1, "")

    for index in range(n_chars):
        if dp[index] is None:
            continue
        base_cost = dp[index][0]
        for end in range(index + 1, min(n_chars, index + max_syllable_chars) + 1):
            substring = core_text[index:end]
            candidates = _candidate_cores(substring, vocab, max_edit_distance)
            if not candidates and is_likely_tibetan_syllable(substring):
                candidates = [(substring, 0)]
            for core, distance in candidates:
                cost = base_cost + _core_cost(core, distance, vocab)
                current = dp[end]
                if current is None or cost < current[0]:
                    dp[end] = (cost, index, core)

    if dp[n_chars] is None:
        return [core_text]

    result: list[str] = []
    index = n_chars
    while index > 0:
        cell = dp[index]
        if cell is None:
            return [core_text]
        _, previous, core = cell
        result.append(core)
        index = previous
    result.reverse()
    return result


def apply_syllable_vocabulary_correction(
    prediction: str,
    vocab: dict,
    max_edit_distance: int = 1,
    max_syllable_chars: int = 8,
) -> str:
    units = split_syllable_units(prediction)
    if not units:
        return prediction

    corrected: list[str] = []
    counts: Counter = vocab["counts"]
    for unit in units:
        if (
            unit.core in counts
            or (unit.delimiter and len(unit.core) <= 3 and is_likely_tibetan_syllable(unit.core))
        ):
            corrected.append(unit.core + unit.delimiter)
            continue

        cores = segment_core_with_vocabulary(
            unit.core,
            vocab,
            max_edit_distance=max_edit_distance,
            max_syllable_chars=max_syllable_chars,
        )
        if not cores:
            corrected.append(unit.core + unit.delimiter)
        elif len(cores) == 1:
            corrected.append(cores[0] + unit.delimiter)
        else:
            corrected.append(TSHEG.join(cores[:-1]) + TSHEG + cores[-1] + unit.delimiter)

    return "".join(corrected)
