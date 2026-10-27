"""Non-training numerical parity check for the vectorized EasyDesign graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import (
    EasyDesignRecipe,
    _load_tensorflow,
    _resolve_recipe,
    encode_sequences,
    reset_model,
    vectorize_model,
)


def _parity_report(source: Any, vectorized: Any, x: np.ndarray, y: np.ndarray, tf: Any) -> dict[str, Any]:
    def evaluate(model: Any) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
        with tf.GradientTape() as tape:
            prediction = model(x, training=False)
            target = tf.reshape(tf.convert_to_tensor(y, dtype=prediction.dtype), [-1, 1])
            mse = tf.reduce_mean(tf.square(prediction - target))
            regularization = tf.add_n(model.losses) if model.losses else tf.constant(0.0)
            objective = mse + regularization
        gradients = tape.gradient(objective, model.trainable_weights)
        if any(gradient is None for gradient in gradients):
            missing = [
                variable.name
                for variable, gradient in zip(model.trainable_weights, gradients)
                if gradient is None
            ]
            raise AssertionError(f"Trainable variables without parity gradients: {missing}")
        keys = [variable.name for variable in model.trainable_weights]
        if len(keys) != len(set(keys)):
            raise AssertionError("Trainable variable names are not unique in the restored graph")
        return (
            prediction.numpy(),
            float(objective.numpy()),
            {key: gradient.numpy() for key, gradient in zip(keys, gradients)},
        )

    pred_source, objective_source, grads_source = evaluate(source)
    pred_vectorized, objective_vectorized, grads_vectorized = evaluate(vectorized)
    if grads_source.keys() != grads_vectorized.keys():
        raise AssertionError("Released and vectorized graphs expose different trainable variables")
    gradient_max_abs = {
        key: float(np.max(np.abs(grads_source[key] - grads_vectorized[key])))
        for key in grads_source
    }
    gradient_relative_l2 = {
        key: float(
            np.linalg.norm(grads_source[key] - grads_vectorized[key])
            / (np.linalg.norm(grads_source[key]) + 1e-7)
        )
        for key in grads_source
    }
    prediction_error = float(np.max(np.abs(pred_source - pred_vectorized)))
    objective_error = abs(objective_source - objective_vectorized)
    if not np.allclose(pred_source, pred_vectorized, rtol=2e-5, atol=2e-5):
        raise AssertionError(f"Forward parity failed: max abs error {prediction_error}")
    if objective_error >= 2e-5:
        raise AssertionError(f"Objective parity failed: abs error {objective_error}")
    if not all(
        np.allclose(grads_source[key], grads_vectorized[key], rtol=2e-4, atol=2e-5)
        for key in grads_source
    ):
        raise AssertionError("At least one trainable-variable gradient failed parity")
    return {
        "status": "pass",
        "n": int(len(x)),
        "parameter_count_source": int(source.count_params()),
        "parameter_count_vectorized": int(vectorized.count_params()),
        "prediction_max_abs_error": prediction_error,
        "objective_abs_error": objective_error,
        "gradient_max_abs_errors": gradient_max_abs,
        "gradient_relative_l2_errors": gradient_relative_l2,
        "n_trainable_gradient_tensors_checked": len(grads_source),
        "all_trainable_gradients_checked": True,
    }


def check_vectorization_parity(
    config: Mapping[str, Any] | str | Path,
    sample_frame: Any,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare forward/objective/all gradients on the released and reset graphs.

    This check does not call ``fit``.  The sample frame must carry a finite
    processed-label column solely for the objective/gradient parity audit.
    """

    recipe: EasyDesignRecipe = _resolve_recipe(config)
    if "label_normalized" in sample_frame.columns:
        label_column = "label_normalized"
    elif "y_true" in sample_frame.columns:
        label_column = "y_true"
    else:
        raise ValueError("Parity sample must contain label_normalized or y_true")
    sample = sample_frame
    if "context_status" in sample.columns:
        sample = sample.loc[sample["context_status"].eq("ok")]
    if len(sample) < 2:
        raise ValueError("Parity sample needs at least two context_status=ok rows")
    sample = sample.iloc[:7].copy()
    x = encode_sequences(sample)
    y = np.asarray(sample[label_column], dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Parity sample labels must be finite")

    tf, tf_keras = _load_tensorflow()
    pretrained = tf.keras.models.load_model(str(recipe.model_dir), compile=False)
    pretrained_vectorized = vectorize_model(pretrained, tf)
    pretrained_report = _parity_report(pretrained, pretrained_vectorized, x, y, tf)
    pretrained_report["stage"] = "pretrained_released_weights"

    reset_source = reset_model(
        recipe,
        tf,
        seed=42,
        vectorized=False,
        compile_for_training=False,
    )
    reset_vectorized = vectorize_model(reset_source, tf)
    reset_report = _parity_report(reset_source, reset_vectorized, x, y, tf)
    reset_report["stage"] = "frozen_seed_reset_weights"

    report = {
        "status": "pass",
        "source_model_saved_model_pb_sha256": recipe.model_pb_sha256,
        "training_config_path": str(recipe.training_config_path) if recipe.training_config_path else None,
        "training_config_sha256": recipe.training_config_sha256,
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "tensorflow": tf.__version__,
            "tf_keras": getattr(tf_keras, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "sample_record_ids": [str(value) for value in sample["record_id"].tolist()],
        "label_column_for_objective_only": label_column,
        "training_was_run": False,
        "stages": [pretrained_report, reset_report],
    }
    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite parity report: {destination}")
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return report


def check_frozen_runtime(
    config: Mapping[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load/reset/compile once and verify the frozen optimizer without fitting."""

    recipe = _resolve_recipe(config)
    tf, tf_keras = _load_tensorflow()
    model = reset_model(
        recipe,
        tf,
        seed=42,
        vectorized=True,
        compile_for_training=True,
    )
    optimizer = model.optimizer
    learning_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
    if optimizer.__class__.__name__ != "Adam":
        raise AssertionError(f"Expected legacy Adam, observed {optimizer.__class__.__name__}")
    if not np.isclose(learning_rate, 0.003684121184051037, rtol=0, atol=1e-12):
        raise AssertionError(f"Frozen Adam learning rate changed: {learning_rate}")
    if model.count_params() != 214569 or len(model.trainable_weights) != 14:
        raise AssertionError(
            "Frozen released graph shape changed: "
            f"params={model.count_params()}, trainable={len(model.trainable_weights)}"
        )
    report = {
        "status": "pass",
        "source_model_saved_model_pb_sha256": recipe.model_pb_sha256,
        "training_config_sha256": recipe.training_config_sha256,
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "tensorflow": tf.__version__,
            "tf_keras": getattr(tf_keras, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "graph": {
            "input_shape": [45, 10],
            "output_shape": [int(value) if value is not None else None for value in model.output_shape],
            "parameter_count": int(model.count_params()),
            "trainable_weight_count": len(model.trainable_weights),
            "regularization_loss_count": len(model.losses),
        },
        "optimizer": {
            "class": optimizer.__class__.__name__,
            "module": optimizer.__class__.__module__,
            "learning_rate": learning_rate,
            "config": optimizer.get_config(),
        },
        "training_was_run": False,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite runtime preflight: {destination}")
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return report
