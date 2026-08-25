#!/usr/bin/env python3
"""Backward-compatible wrapper for table-based Cas12a activity prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas12a_ml import predict_file, read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--model", choices=["d", "xgboost-legacy"], default="d")
    parser.add_argument("--on-invalid", choices=["error", "keep"], default="error")
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    destination = predict_file(
        args.input,
        args.output,
        repository_root=ROOT,
        primary_model=args.model,
        sheet=sheet,
        on_invalid=args.on_invalid,
        overwrite_results=args.overwrite_results,
    )
    print(f"Predicted {len(read_table(destination))} sequence pairs -> {destination.resolve()}")


if __name__ == "__main__":
    main()
