"""Portable, complete JSON state for statistical restoration models."""

from collections import Counter, defaultdict


def serialize_contextual_state(base_mapping, context_counts, vocabulary):
    return {
        "format_version": 1,
        "base_mapping": dict(base_mapping),
        "context_counts": [
            {"key": list(key), "outputs": dict(counts)}
            for key, counts in sorted(context_counts.items())
        ],
        "vocabulary": {
            "counts": dict(vocabulary["counts"]),
            "delimiters": {
                core: dict(counts)
                for core, counts in vocabulary["delimiters"].items()
            },
            "total": vocabulary["total"],
        },
    }


def deserialize_contextual_state(state):
    if state.get("format_version") != 1:
        raise ValueError("Unsupported contextual state format.")
    vocabulary = state["vocabulary"]
    counts = Counter(vocabulary["counts"])
    delimiters = {
        core: Counter(counter) for core, counter in vocabulary["delimiters"].items()
    }
    by_length = defaultdict(list)
    for core in counts:
        by_length[len(core)].append(core)
    restored = {
        "counts": counts,
        "delimiters": delimiters,
        "by_length": dict(by_length),
        "total": vocabulary["total"],
    }
    return (
        dict(state["base_mapping"]),
        {tuple(item["key"]): Counter(item["outputs"]) for item in state["context_counts"]},
        restored,
    )


def validate_candidate_alignment(primary_rows, auxiliary_rows, expected_pairs=None):
    """Reject silent zip truncation and row permutation before training/evaluation."""
    if len(primary_rows) != len(auxiliary_rows):
        raise ValueError("Candidate files have different row counts.")
    if expected_pairs is not None and len(primary_rows) != len(expected_pairs):
        raise ValueError("Candidate row count does not match the frozen split.")
    for index, (primary, auxiliary) in enumerate(zip(primary_rows, auxiliary_rows)):
        if primary.get("source") != auxiliary.get("source"):
            raise ValueError(f"Candidate source mismatch at row {index + 1}.")
        if primary.get("reference") != auxiliary.get("reference"):
            raise ValueError(f"Candidate reference mismatch at row {index + 1}.")
        if expected_pairs is not None:
            expected_source, expected_reference = expected_pairs[index]
            if (primary.get("source"), primary.get("reference")) != (
                expected_source, expected_reference
            ):
                raise ValueError(f"Candidate order differs from frozen split at row {index + 1}.")
