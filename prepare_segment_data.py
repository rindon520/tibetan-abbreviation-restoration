import argparse
import json
import re
from pathlib import Path

from train_restore import read_parallel_data


SHAD = "\u0f0d"


def split_restore_segments(text):
    text = text.strip()
    if not text:
        return []
    if SHAD not in text:
        return [text]
    segments = []
    for part in text.split(SHAD):
        part = part.strip()
        if part:
            segments.append(part + SHAD)
    return segments


def split_source_segments(text):
    return [part.strip() for part in re.split(r"\s{2,}", text.strip()) if part.strip()]


def extract_trusted_segments(pairs):
    segments = []
    stats = {
        "input_lines": 0,
        "accepted_lines": 0,
        "rejected_lines": 0,
        "segments": 0,
    }
    mismatch_counts = {}
    for source, restore in pairs:
        stats["input_lines"] += 1
        source_segments = split_source_segments(source)
        restore_segments = split_restore_segments(restore)
        key = f"{len(source_segments)}:{len(restore_segments)}"
        mismatch_counts[key] = mismatch_counts.get(key, 0) + 1

        if source_segments and len(source_segments) == len(restore_segments):
            stats["accepted_lines"] += 1
            for source_segment, restore_segment in zip(source_segments, restore_segments):
                segments.append((source_segment, restore_segment))
        else:
            stats["rejected_lines"] += 1

    stats["segments"] = len(segments)
    stats["source_restore_segment_count_pairs"] = mismatch_counts
    return segments, stats


def write_parallel_data(pairs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "Source.txt").write_text(
        "\n".join(source for source, _ in pairs) + "\n", encoding="utf-8"
    )
    (output_dir / "Restore.txt").write_text(
        "\n".join(restore for _, restore in pairs) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare trusted segment-level pairs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_segments_trusted"))
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = read_parallel_data(args.data_dir / "Source.txt", args.data_dir / "Restore.txt")
    segments, stats = extract_trusted_segments(pairs)
    write_parallel_data(segments, args.output_dir)
    vocab_src = args.data_dir / "tibetan_VCC_7387.txt"
    vocab_dst = args.output_dir / "tibetan_VCC_7387.txt"
    if vocab_src.exists():
        vocab_dst.write_text(vocab_src.read_text(encoding="utf-8"), encoding="utf-8")
    (args.output_dir / "segment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
