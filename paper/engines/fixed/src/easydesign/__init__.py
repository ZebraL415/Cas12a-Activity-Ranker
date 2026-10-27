"""Fold-local adapter for the released EasyDesign regressor.

TensorFlow is loaded lazily by the training/parity functions so the core
revision environment can import this package without importing TensorFlow.
"""

from .adapter import (
    encode_sequences,
    fit_predict_fold,
    prepare_partition,
)
from .context_audit import audit_common_context_partitions
from .parity import check_frozen_runtime, check_vectorization_parity

__all__ = [
    "audit_common_context_partitions",
    "check_frozen_runtime",
    "check_vectorization_parity",
    "encode_sequences",
    "fit_predict_fold",
    "prepare_partition",
]
