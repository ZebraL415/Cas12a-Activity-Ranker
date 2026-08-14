"""Utilities for the frozen Cas12a diagnostic-activity models."""

from .features import build_feature_frame
from .predict import Cas12aPredictor

__all__ = ["Cas12aPredictor", "build_feature_frame"]
__version__ = "1.0.0"
