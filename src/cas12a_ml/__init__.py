"""Utilities for frozen Cas12a diagnostic-activity models."""

from .features import build_feature_frame
from .io import InputValidationError, predict_file, read_table, write_table
from .predict import Cas12aPredictor

__all__ = [
    "Cas12aPredictor",
    "InputValidationError",
    "build_feature_frame",
    "predict_file",
    "read_table",
    "write_table",
]
__version__ = "2.0.0"
