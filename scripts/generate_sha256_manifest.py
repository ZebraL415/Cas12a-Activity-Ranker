#!/usr/bin/env python3
"""Generate the repository-wide SHA-256 and size manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root_default)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    destination = root / "data" / "metadata" / "sha256_manifest.tsv"
    excluded_top_level = {"_local_only", ".git", ".venv", "reproduced_run", "build", "dist"}
    excluded_relative: set[str] = set()
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (
            not path.is_file()
            or path == destination
            or path.parts[len(root.parts)] in excluded_top_level
            or "__pycache__" in path.parts
            or path.name == ".DS_Store"
            or path.name.endswith((".pyc", ".pyo"))
            or any(relative == item or relative.startswith(item + "/") for item in excluded_relative)
            or relative.endswith(".egg-info")
            or ".egg-info/" in relative
        ):
            continue
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(destination, sep="\t", index=False)
    print(f"Wrote {len(rows)} file checksums -> {destination}")


if __name__ == "__main__":
    main()
