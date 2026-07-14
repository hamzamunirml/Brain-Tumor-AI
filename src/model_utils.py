"""
Model loading and prediction utilities.
"""

import os
from typing import Dict

import numpy as np
from tensorflow.keras.layers import InputLayer as _InputLayer  # type: ignore[reportMissingModuleSource]
from tensorflow.keras.models import load_model as keras_load_model  # type: ignore[reportMissingModuleSource]

from src.config import CLASS_NAMES, MODEL_CONFIGS, MODELS_DIR


class _PatchedInputLayer(_InputLayer):
    """
    Compatibility shim for models saved with Keras 3, whose InputLayer
    config uses the 'batch_shape' key. Older Keras (bundled with some
    TensorFlow versions) only understands 'batch_input_shape', so loading
    such a file directly could raise:
        TypeError: Unrecognized keyword arguments: ['batch_shape']
    This subclass rewrites the config on the fly before deserializing.
    Kept as a defensive measure even though all models are now saved in
    native `.keras` format.
    """

    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config:
            config = dict(config)
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)


def get_model_path(model_name: str) -> str:
    """
    Resolve the full path to a registered model's `.keras` file.

    Also checks an optional `fallback_filename` in MODEL_CONFIGS (e.g. a
    legacy `.h5` export), in case a model was ever re-saved in the older
    format. Raises FileNotFoundError only when NEITHER variant exists.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_name]
    primary_path = os.path.join(MODELS_DIR, cfg["filename"])
    if os.path.exists(primary_path):
        return primary_path

    fallback_name = cfg.get("fallback_filename")
    if fallback_name:
        fallback_path = os.path.join(MODELS_DIR, fallback_name)
        if os.path.exists(fallback_path):
            return fallback_path

    # Neither exists — return the primary path so the caller's
    # FileNotFoundError message points at the expected filename.
    return primary_path


def load_model(model_name: str):
    """
    Load a trained Keras model by its registry name (see MODEL_CONFIGS in
    src/config.py: "ResNet50", "MobileNetV2", "Custom CNN").
    Raises FileNotFoundError with a clear message if the `.keras` file is
    missing from the models/ folder (common after a fresh clone, before
    copying the trained weights from brain_tumor_deep_learning.ipynb's
    output).
    """
    path = get_model_path(model_name)
    if not os.path.exists(path):
        cfg = MODEL_CONFIGS[model_name]
        expected = cfg["filename"]
        if cfg.get("fallback_filename"):
            expected += f" (or {cfg['fallback_filename']})"
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Copy '{expected}' from brain_tumor_deep_learning.ipynb's "
            f"models/ output into this project's models/ folder."
        )
    return keras_load_model(
        path,
        custom_objects={"InputLayer": _PatchedInputLayer},
        compile=False,
    )


def predict_image(model, preprocessed_array: np.ndarray) -> Dict:
    """
    Run inference on a single preprocessed image batch.

    `preprocessed_array` must be RAW pixel values in [0, 255] — every model
    in this project normalizes internally (see src/preprocessing.py).

    Returns
    -------
    dict with keys:
        - label: str ("Healthy" or "Tumor")
        - confidence: float (0-1, confidence in the predicted label)
        - probabilities: dict mapping each class name to its probability
    """
    raw_pred = model.predict(preprocessed_array, verbose=0)
    tumor_prob = float(raw_pred.flatten()[0])  # sigmoid output: P(class == "Tumor")
    healthy_prob = 1.0 - tumor_prob

    label_idx = 1 if tumor_prob >= 0.5 else 0
    label = CLASS_NAMES[label_idx]
    confidence = tumor_prob if label_idx == 1 else healthy_prob

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "Healthy": healthy_prob,
            "Tumor": tumor_prob,
        },
    }
