#!/usr/bin/env python3
"""Predict fluorescence-derived Cas12a diagnostic activity for sequence pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cas12a_ml import Cas12aPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV containing crRNA_sequence and target_aligned_25")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    args = parser.parse_args()
    pairs = pd.read_csv(args.input, dtype=str)
    predictions = Cas12aPredictor(ROOT).predict(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predicted {len(predictions)} sequence pairs -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
