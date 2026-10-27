"""Protocol-aware retraining adapter for the released EasyDesign regressor.

The released SavedModel is loaded only inside the dedicated TensorFlow
environment.  The model is reset from its released graph for every stopping
and final-refit stage; no author training augmentation or deployment gate is
used here.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOLS = ("target", "guide", "atomic_template", "dual_atomic")
FOLD_COLUMNS = {protocol: f"{protocol}_fold" for protocol in PROTOCOLS}
GROUP_COLUMNS = {protocol: f"{protocol}_group" for protocol in PROTOCOLS}
LABEL_COLUMNS = ("label_normalized", "y_true")
QUERY_LABEL_COLUMNS = {"label_normalized", "y_true", "label_raw_value", "label_raw_name"}
CONTEXT_COLUMN = "target_context_45"
GUIDE_COLUMNS = ("guide", "crRNA_sequence")
MODEL_PB_SHA256 = "f012985be29b90e86e45b351cf826de1f1f712709611ec0b3c81f07f24900309"
BASE_INDEX = {base: index for index, base in enumerate("ACGT-")}
IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "-": "-",
    "K": "GT",
    "M": "AC",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "B": "CGT",
    "V": "ACG",
    "H": "ACT",
    "D": "AGT",
    "N": "ACGT",
}

FROZEN_RECIPE = {
    "learning_rate": 0.003684121184051037,
    "batch_size": 229,
    "max_epochs": 450,
    "early_stopping_patience": 25,
    "min_delta": 0.0001,
    "seed": 42,
    "optimizer": "Adam",
    "loss": "mse",
}


@dataclass(frozen=True)
class FoldPartition:
    """Validated ID sets and masks for one protocol/outer-fold fit."""

    protocol_id: str
    outer_fold: int
    expected_stop_fold: int
    fit_ids: tuple[str, ...]
    inner_fit_ids: tuple[str, ...]
    stop_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    inner_fit_mask: np.ndarray
    stop_mask: np.ndarray
    fold_column: str | None
    group_column: str | None
    assignment_check: str


@dataclass(frozen=True)
class EasyDesignRecipe:
    model_dir: Path
    model_pb_sha256: str
    training_config_path: Path | None
    training_config_sha256: str
    training: Mapping[str, Any]
    label_column: str | None
    main_config: Mapping[str, Any]


def _frame_columns(frame: Any) -> set[str]:
    if not hasattr(frame, "columns"):
        raise TypeError("fit_frame and query_frame must be tabular objects with columns")
    return set(str(column) for column in frame.columns)


def _column_values(frame: Any, name: str) -> list[Any]:
    if name not in _frame_columns(frame):
        raise ValueError(f"Frame is missing required column {name!r}")
    values = frame[name]
    return values.tolist() if hasattr(values, "tolist") else list(values)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    # pandas.NA deliberately stays an optional dependency here.
    try:
        unequal_to_self = value != value
        return bool(unequal_to_self) if isinstance(unequal_to_self, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _record_ids(frame: Any, name: str) -> tuple[str, ...]:
    raw = _column_values(frame, "record_id")
    if any(_is_missing(value) for value in raw):
        raise ValueError(f"{name} contains a missing record_id")
    ids = tuple(str(value).strip() for value in raw)
    if any(not record_id for record_id in ids):
        raise ValueError(f"{name} contains an empty record_id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate record_id values")
    return ids


def prepare_partition(
    fit_frame: Any,
    stop_fold_ids: Iterable[Any],
    query_frame: Any,
    *,
    outer_fold: int,
    protocol_id: str,
) -> FoldPartition:
    """Check ID isolation and derive inner-fit/stop masks without reading labels."""

    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unknown protocol_id {protocol_id!r}; expected one of {PROTOCOLS}")
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, (int, np.integer)):
        raise TypeError("outer_fold must be an integer in [0, 4]")
    outer_fold = int(outer_fold)
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in [0, 4]")

    fit_ids = _record_ids(fit_frame, "fit_frame")
    query_ids = _record_ids(query_frame, "query_frame")
    stop_ids_input = tuple(str(value).strip() for value in stop_fold_ids)
    if not stop_ids_input or any(not value for value in stop_ids_input):
        raise ValueError("stop_fold_ids must contain at least one non-empty record_id")
    if len(set(stop_ids_input)) != len(stop_ids_input):
        raise ValueError("stop_fold_ids contains duplicate record IDs")
    stop_id_set = set(stop_ids_input)
    fit_id_set = set(fit_ids)
    query_id_set = set(query_ids)
    if not stop_id_set.issubset(fit_id_set):
        missing = sorted(stop_id_set - fit_id_set)[:5]
        raise ValueError(f"Stopping IDs are not contained in outer-fit IDs: {missing}")
    overlap = fit_id_set & query_id_set
    if overlap:
        raise ValueError(f"Outer fit and query share record IDs: {sorted(overlap)[:5]}")
    if stop_id_set & query_id_set:
        raise ValueError("Stopping IDs overlap outer-query IDs")
    if len(fit_ids) <= len(stop_id_set):
        raise ValueError("At least one non-stopping record is required for inner fitting")
    if not query_ids:
        raise ValueError("Outer query must not be empty")

    forbidden = sorted(_frame_columns(query_frame) & QUERY_LABEL_COLUMNS)
    if forbidden:
        raise ValueError(
            "query_frame must be label-free; remove query target columns before prediction: "
            f"{forbidden}"
        )

    expected_stop_fold = (outer_fold + 1) % 5
    fold_column = FOLD_COLUMNS[protocol_id]
    group_column = GROUP_COLUMNS[protocol_id]
    fold_present = fold_column in _frame_columns(fit_frame) and fold_column in _frame_columns(
        query_frame
    )
    assignment_check = "explicit ID sets; protocol fold column not supplied"
    if fold_present:
        fit_folds = _fold_values(_column_values(fit_frame, fold_column), "fit_frame", fold_column)
        query_folds = _fold_values(
            _column_values(query_frame, fold_column), "query_frame", fold_column
        )
        if any(value == outer_fold for value in fit_folds):
            raise ValueError("fit_frame contains records assigned to the outer query fold")
        if any(value != outer_fold for value in query_folds):
            raise ValueError("query_frame contains records outside the requested outer fold")
        fit_id_by_fold = dict(zip(fit_ids, fit_folds))
        stop_folds = {fit_id_by_fold[record_id] for record_id in stop_id_set}
        if stop_folds != {expected_stop_fold}:
            raise ValueError(
                "stop_fold_ids must be the next protocol outer fold within fit: "
                f"expected {expected_stop_fold}, observed {sorted(stop_folds)}"
            )
        assignment_check = f"verified {fold_column}: query={outer_fold}, stop={expected_stop_fold}"
    group_column = (
        GROUP_COLUMNS[protocol_id]
        if GROUP_COLUMNS[protocol_id] in _frame_columns(fit_frame)
        and GROUP_COLUMNS[protocol_id] in _frame_columns(query_frame)
        else None
    )

    stop_mask = np.fromiter((record_id in stop_id_set for record_id in fit_ids), dtype=bool)
    inner_fit_mask = ~stop_mask
    inner_fit_ids = tuple(record_id for record_id, is_stop in zip(fit_ids, stop_mask) if not is_stop)
    ordered_stop_ids = tuple(record_id for record_id, is_stop in zip(fit_ids, stop_mask) if is_stop)
    if set(ordered_stop_ids) != stop_id_set:
        raise AssertionError("Internal error: stopping ID partition changed")

    return FoldPartition(
        protocol_id=protocol_id,
        outer_fold=outer_fold,
        expected_stop_fold=expected_stop_fold,
        fit_ids=fit_ids,
        inner_fit_ids=inner_fit_ids,
        stop_ids=ordered_stop_ids,
        query_ids=query_ids,
        inner_fit_mask=inner_fit_mask,
        stop_mask=stop_mask,
        fold_column=fold_column if fold_present else None,
        group_column=group_column,
        assignment_check=assignment_check,
    )


def _fold_values(values: list[Any], frame_name: str, column: str) -> tuple[int, ...]:
    folds: list[int] = []
    for value in values:
        if _is_missing(value):
            raise ValueError(f"{frame_name}.{column} contains a missing fold assignment")
        number = float(value)
        if not number.is_integer() or int(number) not in range(5):
            raise ValueError(f"{frame_name}.{column} contains invalid fold value {value!r}")
        folds.append(int(number))
    return tuple(folds)


def _get_guide_column(frame: Any) -> str:
    columns = _frame_columns(frame)
    for candidate in GUIDE_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(f"Frame needs one of the EasyDesign guide columns: {GUIDE_COLUMNS}")


def _normalize_sequence(value: Any, *, row: int, field: str, expected_length: int) -> str:
    if _is_missing(value):
        raise ValueError(f"Missing {field} at row {row}")
    sequence = str(value).strip().upper().replace("U", "T")
    if len(sequence) != expected_length:
        raise ValueError(
            f"{field} at row {row} must have length {expected_length}; got {len(sequence)}"
        )
    invalid = sorted(set(sequence) - set(IUPAC))
    if invalid:
        raise ValueError(f"Invalid EasyDesign IUPAC symbol(s) in {field} at row {row}: {invalid}")
    if field == CONTEXT_COLUMN and set(sequence) == {"-"}:
        raise ValueError(f"{field} at row {row} cannot consist only of gaps")
    return sequence


def encode_sequences(frame: Any) -> np.ndarray:
    """Encode 45-nt context and aligned 25-nt guide as the frozen 45×10 tensor.

    The fifth target/guide alphabet channel represents a gap.  Ambiguous IUPAC
    symbols are represented by the uniform fractional code used in the source
    adapter; no sequence feature is inferred from labels.
    """

    contexts = _column_values(frame, CONTEXT_COLUMN)
    guide_column = _get_guide_column(frame)
    guides = _column_values(frame, guide_column)
    if len(contexts) != len(guides):
        raise ValueError("Context and guide columns have different lengths")

    encoded = np.zeros((len(contexts), 45, 10), dtype=np.float32)
    for row, (raw_context, raw_guide) in enumerate(zip(contexts, guides)):
        context = _normalize_sequence(raw_context, row=row, field=CONTEXT_COLUMN, expected_length=45)
        guide = _normalize_sequence(raw_guide, row=row, field=guide_column, expected_length=25)
        for position, symbol in enumerate(context):
            bases = IUPAC[symbol]
            probability = np.float32(1.0 / len(bases))
            for base in bases:
                encoded[row, position, BASE_INDEX[base]] = probability
        for offset, symbol in enumerate(guide, start=10):
            bases = IUPAC[symbol]
            probability = np.float32(1.0 / len(bases))
            for base in bases:
                encoded[row, offset, 5 + BASE_INDEX[base]] = probability
    return encoded


def _get_labels(frame: Any, recipe: EasyDesignRecipe) -> tuple[np.ndarray, str]:
    columns = _frame_columns(frame)
    requested = recipe.label_column
    if requested:
        if requested not in columns:
            raise ValueError(f"Configured EasyDesign label column {requested!r} is absent from fit")
        label_column = requested
    else:
        present = [column for column in LABEL_COLUMNS if column in columns]
        if not present:
            raise ValueError(f"fit_frame must contain one of {LABEL_COLUMNS}")
        label_column = present[0]

    raw = _column_values(frame, label_column)
    labels = np.asarray(raw, dtype=np.float32)
    if labels.ndim != 1 or labels.size != len(frame):
        raise ValueError("Fit labels must be one-dimensional and match fit_frame rows")
    if not np.isfinite(labels).all():
        raise ValueError("Fit labels must all be finite on the supplied processed-label scale")

    other_columns = [column for column in LABEL_COLUMNS if column != label_column and column in columns]
    for other in other_columns:
        other_values = np.asarray(_column_values(frame, other), dtype=np.float32)
        if not np.isfinite(other_values).all() or not np.array_equal(labels, other_values):
            raise ValueError(
                f"fit_frame label columns {label_column!r} and {other!r} disagree; "
                "resolve the processed-label source before EasyDesign fitting"
            )
    return labels, label_column


def _config_mapping(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config).expanduser()
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_recipe(config: Mapping[str, Any] | str | Path) -> EasyDesignRecipe:
    main = _config_mapping(config)
    paths = main.get("source_paths", main.get("inputs", {}))
    if not isinstance(paths, Mapping):
        paths = {}
    project_root = Path(str(paths.get("project_root", Path.cwd()))).expanduser()

    training_raw = main.get("easydesign_training_config")
    training_config_path: Path | None = None
    if isinstance(training_raw, Mapping):
        training = dict(training_raw)
        training_hash = hashlib.sha256(
            json.dumps(training, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    else:
        config_path_raw = paths.get("easydesign_config")
        if config_path_raw is not None:
            config_path = Path(str(config_path_raw)).expanduser()
            if not config_path.is_absolute():
                config_path = project_root / config_path
            if not config_path.is_file():
                raise FileNotFoundError(f"Configured EasyDesign recipe file is absent: {config_path}")
            training_config_path = config_path
            training = json.loads(config_path.read_text(encoding="utf-8"))
            training_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        elif all(key in main for key in FROZEN_RECIPE):
            training = main
            training_hash = hashlib.sha256(
                json.dumps(training, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError(
                "config must provide source_paths.easydesign_config or a complete "
                "easydesign_training_config"
            )

    for key, expected in FROZEN_RECIPE.items():
        observed = training.get(key)
        resolved_observed = (
            str(observed).strip().lower().split()[0]
            if key == "loss" and observed is not None
            else observed
        )
        if resolved_observed != expected:
            raise ValueError(
                f"Frozen EasyDesign recipe mismatch for {key}: expected {expected!r}, "
                f"observed {observed!r}"
            )

    model_path_raw = paths.get("easydesign_model", main.get("model_path"))
    if model_path_raw is None:
        raise ValueError("config.source_paths.easydesign_model is required")
    model_dir = Path(str(model_path_raw)).expanduser()
    if not model_dir.is_absolute():
        model_dir = project_root / model_dir
    saved_model = model_dir / "saved_model.pb"
    if not saved_model.is_file():
        raise FileNotFoundError(f"Released EasyDesign SavedModel is missing: {saved_model}")
    model_hash = hashlib.sha256(saved_model.read_bytes()).hexdigest()
    if model_hash != MODEL_PB_SHA256:
        raise ValueError(
            "Released EasyDesign SavedModel hash differs from the frozen source: "
            f"expected {MODEL_PB_SHA256}, observed {model_hash}"
        )

    label_column = main.get("easydesign_label_column")
    return EasyDesignRecipe(
        model_dir=model_dir,
        model_pb_sha256=model_hash,
        training_config_path=training_config_path,
        training_config_sha256=training_hash,
        training=training,
        label_column=str(label_column) if label_column else None,
        main_config=main,
    )


def _load_tensorflow() -> tuple[Any, Any]:
    """Import and verify the frozen legacy Keras runtime only when needed."""

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        raise RuntimeError("TensorFlow must start with TF_USE_LEGACY_KERAS=1")

    # TensorFlow 2.18 on Python 3.12 expects setuptools' vendored distutils.
    import setuptools  # noqa: F401
    import tensorflow as tf
    import tf_keras

    if tf.__version__ != "2.18.0" or getattr(tf_keras, "__version__", None) != "2.18.0":
        raise RuntimeError(
            "EasyDesign requires TensorFlow 2.18.0 + tf-keras 2.18.0; observed "
            f"TensorFlow {tf.__version__}, tf-keras {getattr(tf_keras, '__version__', 'unknown')}"
        )
    return tf, tf_keras


def _prepare_legacy_seed(tf: Any, seed: int) -> None:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        from tf_keras.src import backend as legacy_backend

        # tf-keras 2.18 passes 1e9 as float to Random.randint on the seeded
        # Python 3.12 path; the published compatibility workaround forces its
        # int(1e9) fallback.  All model variables are reset explicitly below.
        legacy_backend._SEED_GENERATOR.generator = None
    except (AttributeError, ImportError):
        pass


def _make_vector_local(tf: Any) -> type:
    class VectorLocal(tf.keras.layers.Layer):
        def __init__(self, source: Any, **kwargs: Any) -> None:
            super().__init__(name=source.name, **kwargs)
            self.width = int(source.kernel_size[0])
            self.filters = int(source.filters)
            self.kernel_shape = tuple(int(value) for value in source.kernel.shape)
            self.kernel_regularizer = source.kernel_regularizer
            self.bias_regularizer = source.bias_regularizer
            self.activation_fn = source.activation

        def build(self, input_shape: Any) -> None:
            self.kernel = self.add_weight(
                name="kernel",
                shape=self.kernel_shape,
                initializer="zeros",
                regularizer=self.kernel_regularizer,
            )
            self.bias = self.add_weight(
                name="bias",
                shape=(self.kernel_shape[0], self.filters),
                initializer="zeros",
                regularizer=self.bias_regularizer,
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            frames = tf.signal.frame(inputs, self.width, 1, axis=1)
            windows = tf.reshape(
                frames,
                [-1, self.kernel_shape[0], self.kernel_shape[1]],
            )
            return self.activation_fn(
                tf.einsum("bpi,pio->bpo", windows, self.kernel) + self.bias
            )

    return VectorLocal


def vectorize_model(source: Any, tf: Any) -> Any:
    """Replace only released LocallyConnected1D ops with a vectorized equivalent."""

    vector_local = _make_vector_local(tf)
    old_layers = {layer.name: layer for layer in source.layers}
    new_layers: dict[str, Any] = {}
    for name, layer in old_layers.items():
        if isinstance(layer, tf.keras.layers.LocallyConnected1D):
            new_layers[name] = vector_local(layer)
        else:
            new_layers[name] = type(layer).from_config(layer.get_config())

    inputs = tf.keras.Input((45, 10))
    x = new_layers["group_w1_conv"](inputs)
    x = new_layers["group_w1_batchnorm"](x)
    x = new_layers["merge_pool"](
        [
            new_layers["group_w1_maxpool"](x),
            new_layers["group_w1_avgpool"](x),
        ]
    )
    x = new_layers["merge_lc"](
        [
            new_layers["group_w1_lc_w1"](x),
            new_layers["group_w1_lc_w2"](x),
        ]
    )
    x = new_layers["flatten"](x)
    x = new_layers["dropout_1"](x)
    x = new_layers["fc_1"](x)
    x = new_layers["dropout_2"](x)
    x = new_layers["fc_2"](x)
    x = new_layers["dropout_2"](x)
    outputs = new_layers["fc_final"](x)
    vectorized = tf.keras.Model(inputs, outputs, name="EasyDesign_released_vectorized")

    for name, layer in old_layers.items():
        new_layers[name].set_weights(layer.get_weights())
    if source.count_params() != vectorized.count_params():
        raise AssertionError(
            "Vectorized EasyDesign graph changed parameter count: "
            f"{source.count_params()} != {vectorized.count_params()}"
        )
    return vectorized


def reset_model(
    recipe: EasyDesignRecipe,
    tf: Any,
    seed: int,
    *,
    vectorized: bool = True,
    compile_for_training: bool = True,
) -> Any:
    """Load the released graph, apply the frozen reset rule, then vectorize."""

    _prepare_legacy_seed(tf, seed)
    model = tf.keras.models.load_model(str(recipe.model_dir), compile=False)
    original_weights = [weight.numpy().copy() for weight in model.weights]
    reset_variables: list[Any] = []
    for index, variable in enumerate(model.weights):
        name = variable.name.split("/")[-1].split(":")[0]
        if name == "kernel":
            value = tf.keras.initializers.GlorotUniform(seed=seed + index)(
                variable.shape, variable.dtype
            )
        elif name in {"gamma", "moving_variance"}:
            value = tf.ones(variable.shape, variable.dtype)
        elif name in {"bias", "beta", "moving_mean"}:
            value = tf.zeros(variable.shape, variable.dtype)
        else:
            raise ValueError(f"Unrecognized released variable during reset: {variable.name}")
        variable.assign(value)
        reset_variables.append(variable)
    kernel_pairs = [
        (old, variable)
        for old, variable in zip(original_weights, reset_variables)
        if "kernel" in variable.name
    ]
    if any(np.array_equal(old, variable.numpy()) for old, variable in kernel_pairs):
        raise AssertionError("At least one released kernel was not reset")

    result = vectorize_model(model, tf) if vectorized else model
    if compile_for_training:
        result.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=FROZEN_RECIPE["learning_rate"]
            ),
            loss="mse",
        )
    return result


def _context_profile(frame: Any) -> dict[str, Any]:
    contexts = _column_values(frame, CONTEXT_COLUMN)
    guide_column = _get_guide_column(frame)
    guides = _column_values(frame, guide_column)
    context_lengths: Counter[str] = Counter()
    guide_lengths: Counter[str] = Counter()
    context_symbols: Counter[str] = Counter()
    guide_symbols: Counter[str] = Counter()
    gap_context_positions: Counter[int] = Counter()
    gap_guide_positions: Counter[int] = Counter()
    for index, (raw_context, raw_guide) in enumerate(zip(contexts, guides)):
        context = _normalize_sequence(raw_context, row=index, field=CONTEXT_COLUMN, expected_length=45)
        guide = _normalize_sequence(raw_guide, row=index, field=guide_column, expected_length=25)
        context_lengths[str(len(context))] += 1
        guide_lengths[str(len(guide))] += 1
        context_symbols.update(context)
        guide_symbols.update(guide)
        gap_context_positions.update(position for position, symbol in enumerate(context) if symbol == "-")
        gap_guide_positions.update(position for position, symbol in enumerate(guide) if symbol == "-")
    return {
        "n_records": len(contexts),
        "guide_column": guide_column,
        "context_length_counts": dict(sorted(context_lengths.items())),
        "guide_length_counts": dict(sorted(guide_lengths.items())),
        "target_context_symbol_counts": dict(sorted(context_symbols.items())),
        "guide_symbol_counts": dict(sorted(guide_symbols.items())),
        "target_context_gap_position_counts_0based": {
            str(key): value for key, value in sorted(gap_context_positions.items())
        },
        "guide_gap_position_counts_0based": {
            str(key): value for key, value in sorted(gap_guide_positions.items())
        },
    }


def _group_count(frame: Any, group_column: str | None) -> int | None:
    if not group_column or group_column not in _frame_columns(frame):
        return None
    values = _column_values(frame, group_column)
    groups = {str(value) for value in values if not _is_missing(value) and str(value) != ""}
    return len(groups)


def _write_id_manifest(
    output_dir: Path,
    *,
    fit_frame: Any,
    query_frame: Any,
    partition: FoldPartition,
) -> Path:
    path = output_dir / "fit_stop_query_ids.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing EasyDesign ID manifest: {path}")

    fields = ["record_id", "role", "protocol_id", "outer_fold", "stop_fold"]
    if partition.fold_column:
        fields.append("assigned_fold")
    if partition.group_column:
        fields.append("isolation_group")

    def add_rows(frame: Any, ids: tuple[str, ...], role: str, rows: list[dict[str, Any]]) -> None:
        frame_ids = _record_ids(frame, f"{role} source frame")
        position_by_id = {record_id: index for index, record_id in enumerate(frame_ids)}
        fold_values = (
            _column_values(frame, partition.fold_column) if partition.fold_column else None
        )
        group_values = (
            _column_values(frame, partition.group_column) if partition.group_column else None
        )
        for record_id in ids:
            if record_id not in position_by_id:
                raise ValueError(f"ID manifest role {role!r} contains ID absent from its source frame: {record_id}")
            index = position_by_id[record_id]
            row: dict[str, Any] = {
                "record_id": record_id,
                "role": role,
                "protocol_id": partition.protocol_id,
                "outer_fold": partition.outer_fold,
                "stop_fold": partition.expected_stop_fold,
            }
            if fold_values is not None:
                row["assigned_fold"] = fold_values[index]
            if group_values is not None:
                row["isolation_group"] = "" if _is_missing(group_values[index]) else group_values[index]
            rows.append(row)

    rows: list[dict[str, Any]] = []
    add_rows(fit_frame, partition.fit_ids, "outer_fit", rows)
    stop_id_set = set(partition.stop_ids)
    stop_frame_ids = tuple(record_id for record_id in partition.fit_ids if record_id in stop_id_set)
    inner_fit_ids = tuple(record_id for record_id in partition.fit_ids if record_id not in stop_id_set)
    add_rows(fit_frame, inner_fit_ids, "inner_fit", rows)
    add_rows(fit_frame, stop_frame_ids, "internal_stop", rows)
    add_rows(query_frame, partition.query_ids, "outer_query", rows)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _write_progress(progress_path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace only the dedicated live-progress JSON file."""

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = progress_path.with_name(progress_path.name + ".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary_path.open("w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, progress_path)


def make_progress_callback(
    tf: Any,
    progress_path: str | Path,
    *,
    protocol_id: str,
    outer_fold: int,
    stage: str,
    max_epochs: int,
    interval: int = 25,
) -> Any:
    """Create an observational callback; it reads logs and writes progress only."""

    if interval < 1:
        raise ValueError("Progress interval must be positive")
    destination = Path(progress_path).expanduser()

    class ProgressOnly(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = 0
            self.loss: float | None = None
            self.val_loss: float | None = None

        def _record(self, status: str) -> dict[str, Any]:
            return {
                "status": status,
                "protocol": protocol_id,
                "protocol_id": protocol_id,
                "fold": int(outer_fold),
                "stage": stage,
                "epoch": int(self.epoch),
                "max_epochs": int(max_epochs),
                "loss": self.loss,
                "val_loss": self.val_loss,
                "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

        def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
            _write_progress(destination, self._record("running"))

        def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
            logs = logs or {}
            self.epoch = int(epoch) + 1
            self.loss = _json_safe_float(logs.get("loss"))
            self.val_loss = _json_safe_float(logs.get("val_loss"))
            if self.epoch % interval != 0:
                return
            record = self._record("running")
            _write_progress(destination, record)
            print(
                "EASYDESIGN_PROGRESS "
                + json.dumps(record, sort_keys=True, allow_nan=False),
                flush=True,
            )

        def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
            if logs:
                self.loss = _json_safe_float(logs.get("loss", self.loss))
                self.val_loss = _json_safe_float(logs.get("val_loss", self.val_loss))
            _write_progress(destination, self._record("complete"))

    return ProgressOnly()


def _runtime_manifest(tf: Any, tf_keras: Any) -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": str(tf.__version__),
        "tf_keras": str(getattr(tf_keras, "__version__", "unknown")),
        "numpy": str(np.__version__),
        "legacy_keras_env": os.environ.get("TF_USE_LEGACY_KERAS", ""),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing EasyDesign metadata: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def fit_predict_fold(
    fit_frame: Any,
    stop_fold_ids: Iterable[Any],
    query_frame: Any,
    *,
    outer_fold: int,
    protocol_id: str,
    output_dir: str | Path,
    config: Mapping[str, Any] | str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select epoch in the protocol's next fold, reset, refit, and predict query.

    The internal stop IDs must be a subset of the outer-fit records.  Only the
    fit partition is read for labels; query labels are rejected if present.
    This function is invoked by the coordinator's one-item training queue and
    must not be called for a fold until that queue grants the fit.
    """

    partition = prepare_partition(
        fit_frame,
        stop_fold_ids,
        query_frame,
        outer_fold=outer_fold,
        protocol_id=protocol_id,
    )
    recipe = _resolve_recipe(config)
    labels, label_column = _get_labels(fit_frame, recipe)
    x_fit = encode_sequences(fit_frame)
    x_query = encode_sequences(query_frame)
    if len(x_fit) != len(partition.fit_ids) or len(x_query) != len(partition.query_ids):
        raise AssertionError("Encoded matrices no longer align with validated record IDs")
    if x_fit.shape[1:] != (45, 10) or x_query.shape[1:] != (45, 10):
        raise AssertionError("EasyDesign input tensor must have the frozen shape (45, 10)")

    tf, tf_keras = _load_tensorflow()
    seed = int(FROZEN_RECIPE["seed"]) + partition.outer_fold
    inner_x = x_fit[partition.inner_fit_mask]
    inner_y = labels[partition.inner_fit_mask]
    stop_x = x_fit[partition.stop_mask]
    stop_y = labels[partition.stop_mask]
    if not len(inner_y) or not len(stop_y):
        raise ValueError("Inner-fit and stopping-validation arrays must both be non-empty")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    id_manifest = _write_id_manifest(
        destination,
        fit_frame=fit_frame,
        query_frame=query_frame,
        partition=partition,
    )
    group_counts = {
        "outer_fit": _group_count(fit_frame, partition.group_column),
        "inner_fit": None,
        "internal_stop": None,
        "outer_query": _group_count(query_frame, partition.group_column),
    }
    if partition.group_column:
        fit_groups = _column_values(fit_frame, partition.group_column)
        stop_groups = [
            value
            for value, is_stop in zip(fit_groups, partition.stop_mask)
            if is_stop and not _is_missing(value)
        ]
        inner_groups = [
            value
            for value, is_stop in zip(fit_groups, partition.stop_mask)
            if not is_stop and not _is_missing(value)
        ]
        group_counts["inner_fit"] = len({str(value) for value in inner_groups})
        group_counts["internal_stop"] = len({str(value) for value in stop_groups})

    metadata: dict[str, Any] = {
        "status": "running",
        "protocol_id": partition.protocol_id,
        "outer_fold": partition.outer_fold,
        "expected_stop_fold": partition.expected_stop_fold,
        "assignment_check": partition.assignment_check,
        "seed": seed,
        "recipe": dict(FROZEN_RECIPE),
        "label_column_used": label_column,
        "query_label_columns_accessed": [],
        "n_outer_fit": len(partition.fit_ids),
        "n_inner_fit": int(np.sum(partition.inner_fit_mask)),
        "n_internal_stop": int(np.sum(partition.stop_mask)),
        "n_outer_query": len(partition.query_ids),
        "group_counts": group_counts,
        "fit_stop_query_id_manifest": str(id_manifest),
        "progress_path": str(destination / "progress.json"),
        "training_config_path": str(recipe.training_config_path) if recipe.training_config_path else None,
        "training_config_sha256": recipe.training_config_sha256,
        "context_diagnostics": {
            "outer_fit": _context_profile(fit_frame),
            "outer_query": _context_profile(query_frame),
        },
        "source_model_saved_model_pb_sha256": recipe.model_pb_sha256,
        "runtime": _runtime_manifest(tf, tf_keras),
        "implementation_notes": [
            "Released regressor graph/configurations and local-layer weights are preserved.",
            "Local connections use the already parity-audited vectorized equivalent.",
            "Gap is encoded in the fifth alphabet channel; labels are not used to build context.",
            "No author augmentation, classifier shift, clamp, or prediction gate is used.",
            "The internal stop fold is protocol-adapted; the epoch algorithm and hyperparameters are unchanged.",
        ],
    }

    try:
        inner_model = reset_model(recipe, tf, seed, vectorized=True, compile_for_training=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=FROZEN_RECIPE["early_stopping_patience"],
            min_delta=FROZEN_RECIPE["min_delta"],
            mode="min",
            restore_best_weights=False,
        )
        inner_progress = make_progress_callback(
            tf,
            destination / "progress.json",
            protocol_id=protocol_id,
            outer_fold=outer_fold,
            stage="inner_stopping",
            max_epochs=FROZEN_RECIPE["max_epochs"],
        )
        inner_history = inner_model.fit(
            inner_x,
            inner_y,
            validation_data=(stop_x, stop_y),
            batch_size=FROZEN_RECIPE["batch_size"],
            epochs=FROZEN_RECIPE["max_epochs"],
            callbacks=[early_stopping, inner_progress],
            verbose=0,
        )
        validation_losses = [float(value) for value in inner_history.history.get("val_loss", [])]
        if not validation_losses or not np.isfinite(validation_losses).all():
            raise RuntimeError("Internal stopping fit produced an empty or non-finite val_loss history")
        selected_epoch = int(np.argmin(validation_losses)) + 1

        # Reset to the same seed and initialize again; refit all outer-fit IDs
        # for the selected epoch count, including the stopping-validation IDs.
        final_model = reset_model(recipe, tf, seed, vectorized=True, compile_for_training=True)
        refit_progress = make_progress_callback(
            tf,
            destination / "progress.json",
            protocol_id=protocol_id,
            outer_fold=outer_fold,
            stage="reset_refit",
            max_epochs=selected_epoch,
        )
        final_model.fit(
            x_fit,
            labels,
            batch_size=FROZEN_RECIPE["batch_size"],
            epochs=selected_epoch,
            callbacks=[refit_progress],
            verbose=0,
        )
        predictions = np.asarray(
            final_model.predict(
                x_query,
                batch_size=FROZEN_RECIPE["batch_size"],
                verbose=0,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if predictions.shape != (len(partition.query_ids),) or not np.isfinite(predictions).all():
            raise RuntimeError("EasyDesign returned a non-finite or misaligned query prediction vector")

        metadata.update(
            {
                "status": "complete",
                "selected_epoch": selected_epoch,
                "inner_epochs_run": len(validation_losses),
                "inner_validation_mse_min": min(validation_losses),
                "inner_validation_mse_by_epoch": validation_losses,
                "reset_refit_epochs": selected_epoch,
                "n_predictions": len(predictions),
            }
        )
        _write_json(destination / "epoch_selection.json", metadata)
        return predictions, metadata
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        failure_path = destination / "epoch_selection.failed.json"
        if not failure_path.exists():
            _write_json(failure_path, metadata)
        raise
