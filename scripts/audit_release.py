"""Audit the intended upload files; optionally create an archive without local runs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", ".validation-venv", "venv", "__pycache__", ".pytest_cache",
    "data_segments_trusted", "runs", "output", "checkpoints", "final_model",
    "experiments", "papers", "submission", "archive", "segment-ti", "third_party", "generated",
}
DISALLOWED_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".whl", ".log", ".docx", ".pdf", ".png", ".svg", ".exe", ".dll", ".so", ".o", ".zip", ".pyc"}
INCLUDED_DATA_FILES = {
    "data/README.md", "data/Source.txt", "data/Restore.txt", "data/tibetan_VCC_7387.txt",
}


def segmenter_upload_files():
    manifest_path = ROOT / "segment-ti/archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {"segment-ti/README.md", "segment-ti/archive_manifest.json"}
    violations = []
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]:
            raise ValueError("Invalid segmenter manifest path.")
        name = (PurePosixPath("segment-ti") / relative).as_posix()
        if name in allowed:
            raise ValueError("Duplicate segmenter manifest path.")
        allowed.add(name)
        path = ROOT / name
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
            violations.append(name)
    for name in allowed:
        if not (ROOT / name).is_file():
            violations.append(name)
    return allowed, violations


def upload_files():
    files = []
    segmenter_files, violations = segmenter_upload_files()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "segment-ti":
            if path.is_file() and relative.as_posix() in segmenter_files:
                files.append(path)
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts) or not path.is_file():
            continue
        if "data" in relative.parts and relative.as_posix() not in INCLUDED_DATA_FILES:
            continue
        if path.name == "release_audit.local.json":
            continue
        if path.suffix.lower() in DISALLOWED_SUFFIXES or path.name.startswith(".env"):
            violations.append(relative.as_posix())
        if (path.name in {"Source.txt", "Restore.txt"}
                and relative.parts[:2] != ("examples", "toy")
                and relative.as_posix() not in INCLUDED_DATA_FILES):
            violations.append(relative.as_posix())
        if path.name in {"model.json", "selector.json"} or path.name.endswith(("_predictions.jsonl", "_samples.json")):
            violations.append(relative.as_posix())
        if path.stat().st_size > 10 * 1024 * 1024:
            violations.append(relative.as_posix())
        files.append(path)
    return sorted(files), sorted(set(violations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    files, violations = upload_files()
    report = {"file_count": len(files), "bytes": sum(path.stat().st_size for path in files), "violations": violations}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)
    if args.archive:
        destination = args.archive.resolve()
        if destination.is_relative_to(ROOT):
            raise ValueError("Archive must be outside the repository to avoid self-inclusion.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(ROOT).as_posix())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(json.dumps({"archive": str(destination), "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
