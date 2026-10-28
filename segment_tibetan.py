from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from tibetan_syllable import split_syllable_units


DEFAULT_SEGMENTER_DIR = Path(__file__).resolve().parent / "segment-ti"
DEFAULT_SEGMENTER_NAME = "./tiSegmenter"


def fallback_segment_line(line: str) -> str:
    units = split_syllable_units(line)
    if not units:
        return " ".join(part for part in line.split() if part)
    return " ".join(unit.core + unit.delimiter for unit in units if unit.core or unit.delimiter)


def fallback_segment_lines(lines: list[str]) -> list[str]:
    return [fallback_segment_line(line) for line in lines]


def _raise_if_unsupported_binary(segmenter_dir: Path):
    binary_path = segmenter_dir / "tiSegmenter"
    if os.name != "nt" or not binary_path.exists():
        return
    with binary_path.open("rb") as file:
        magic = file.read(4)
    if magic == b"\x7fELF":
        raise RuntimeError(
            "tiSegmenter is an ELF binary and cannot run directly from native Windows; "
            "use WSL/Linux or provide a Windows executable."
        )


def _run_external_segmenter(
    lines: list[str],
    segmenter_dir: Path,
    timeout_seconds: float = 10.0,
) -> tuple[list[str], dict]:
    segmenter_dir = Path(segmenter_dir).resolve()
    if not segmenter_dir.exists():
        raise FileNotFoundError(f"Segmenter directory does not exist: {segmenter_dir}")
    _raise_if_unsupported_binary(segmenter_dir)

    with tempfile.TemporaryDirectory(prefix="_codex_seg_", dir=segmenter_dir) as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.txt"
        output_path = tmp_dir / "output.seg"
        input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        command = [
            DEFAULT_SEGMENTER_NAME,
            "-s",
            str(input_path),
            "-t",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=segmenter_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "tiSegmenter failed with exit code "
                f"{completed.returncode}: {completed.stderr or completed.stdout}"
            )
        if not output_path.exists():
            raise RuntimeError("tiSegmenter did not create the requested .seg output file.")

        segmented = output_path.read_text(encoding="utf-8").splitlines()
        if len(segmented) != len(lines):
            raise RuntimeError(
                "tiSegmenter output line count mismatch: "
                f"expected {len(lines)}, got {len(segmented)}. "
                "Check its model and resource files."
            )
        status = {
            "mode": "external",
            "segmenter": "tiSegmenter",
            "cwd": str(segmenter_dir),
            "command": " ".join(command),
            "timeout_seconds": timeout_seconds,
            "lines": len(lines),
        }
        return segmented, status


def segment_lines(
    lines: list[str],
    segmenter_dir: Path | str = DEFAULT_SEGMENTER_DIR,
    prefer_external: bool = True,
    allow_fallback: bool = True,
    timeout_seconds: float = 10.0,
) -> tuple[list[str], dict]:
    segmenter_dir = Path(segmenter_dir)
    if prefer_external:
        try:
            return _run_external_segmenter(
                lines,
                segmenter_dir,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            segmented = fallback_segment_lines(lines)
            return segmented, {
                "mode": "fallback",
                "segmenter": "syllable_units",
                "requested_segmenter": "tiSegmenter",
                "cwd": str(segmenter_dir),
                "reason": str(exc),
                "timeout_seconds": timeout_seconds,
                "lines": len(lines),
            }

    return fallback_segment_lines(lines), {
        "mode": "fallback",
        "segmenter": "syllable_units",
        "requested_segmenter": None,
        "cwd": str(segmenter_dir),
        "reason": "external segmenter disabled",
        "timeout_seconds": timeout_seconds,
        "lines": len(lines),
    }
