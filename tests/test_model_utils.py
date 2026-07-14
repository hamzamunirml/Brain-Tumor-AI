"""
Unit tests for src/model_utils.py
Run with: pytest tests/test_model_utils.py -v

These tests use a lightweight stub model instead of a real trained .keras
file so they run fast and don't require the (large) downloaded weights to
be present in CI or on a fresh clone. get_model_path()/load_model()
file-not-found behavior is tested separately against the real models/ folder.
"""

import numpy as np
import pytest

from src import model_utils
from src.config import MODEL_CONFIGS


class StubModel:
    """A fake Keras model that returns a fixed sigmoid-style prediction."""

    def __init__(self, fixed_output: float):
        self.fixed_output = fixed_output

    def predict(self, x, verbose=0):
        return np.array([[self.fixed_output]], dtype="float32")


class TestGetModelPath:
    def test_known_model_returns_path_ending_in_its_filename(self):
        path = model_utils.get_model_path("ResNet50")
        assert path.endswith(MODEL_CONFIGS["ResNet50"]["filename"])

    def test_all_registered_models_resolve_a_path(self):
        for name in MODEL_CONFIGS:
            path = model_utils.get_model_path(name)
            assert path.endswith(MODEL_CONFIGS[name]["filename"])

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError):
            model_utils.get_model_path("NotARealModel")


class TestLoadModel:
    def test_missing_file_raises_file_not_found_with_helpful_message(
        self, tmp_path, monkeypatch
    ):
        # Point MODELS_DIR at an empty temp directory so this test is
        # correct regardless of whether the developer's real models/
        # folder already has the trained .keras files in it (e.g. on a
        # dev machine, as opposed to a fresh clone/CI runner).
        monkeypatch.setattr(model_utils, "MODELS_DIR", str(tmp_path))

        with pytest.raises(FileNotFoundError, match="Copy"):
            model_utils.load_model("ResNet50")


class TestPredictImage:
    def test_high_output_predicts_tumor(self):
        model = StubModel(fixed_output=0.92)
        dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")

        result = model_utils.predict_image(model, dummy_input)

        assert result["label"] == "Tumor"
        assert result["confidence"] == pytest.approx(0.92)

    def test_low_output_predicts_healthy(self):
        model = StubModel(fixed_output=0.08)
        dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")

        result = model_utils.predict_image(model, dummy_input)

        assert result["label"] == "Healthy"
        assert result["confidence"] == pytest.approx(0.92)  # 1 - 0.08

    def test_probabilities_sum_to_one(self):
        model = StubModel(fixed_output=0.37)
        dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")

        result = model_utils.predict_image(model, dummy_input)

        total = result["probabilities"]["Healthy"] + result["probabilities"]["Tumor"]
        assert total == pytest.approx(1.0)

    def test_boundary_case_exactly_half_predicts_tumor(self):
        # >= 0.5 threshold should classify as Tumor per model_utils logic
        model = StubModel(fixed_output=0.5)
        dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")

        result = model_utils.predict_image(model, dummy_input)

        assert result["label"] == "Tumor"
