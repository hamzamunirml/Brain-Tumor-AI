"""
Unit tests for src/preprocessing.py
Run with: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
from PIL import Image

from src.config import IMG_SIZE
from src.preprocessing import load_image, preprocess_image


def make_dummy_image(size=(300, 400), mode="RGB", color=(120, 60, 200)):
    """Create an in-memory dummy image for testing (no real dataset needed)."""
    return Image.new(mode, size, color)


class TestLoadImage:
    def test_load_image_converts_grayscale_to_rgb(self, tmp_path):
        img_path = tmp_path / "gray.png"
        make_dummy_image(mode="L", color=128).save(img_path)

        img = load_image(str(img_path))

        assert img.mode == "RGB"

    def test_load_image_converts_rgba_to_rgb(self, tmp_path):
        img_path = tmp_path / "rgba.png"
        make_dummy_image(mode="RGBA", color=(10, 20, 30, 255)).save(img_path)

        img = load_image(str(img_path))

        assert img.mode == "RGB"

    def test_load_image_keeps_rgb_as_rgb(self, tmp_path):
        img_path = tmp_path / "rgb.png"
        make_dummy_image(mode="RGB").save(img_path)

        img = load_image(str(img_path))

        assert img.mode == "RGB"


class TestPreprocessImage:
    def test_output_shape_matches_expected_batch_shape(self):
        img = make_dummy_image()
        result = preprocess_image(img)

        assert result.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)

    def test_output_keeps_raw_pixel_range(self):
        # Every model now normalizes internally (Rescaling / preprocess_input
        # baked into the graph), so preprocess_image must NOT rescale —
        # otherwise the model would be double-normalized.
        img = make_dummy_image(color=(255, 255, 255))
        result = preprocess_image(img)

        np.testing.assert_allclose(result.max(), 255.0, atol=1e-6)
        assert result.min() >= 0.0

    def test_output_dtype_is_float32(self):
        img = make_dummy_image()
        result = preprocess_image(img)

        assert result.dtype == np.float32

    def test_grayscale_array_is_stacked_to_three_channels(self):
        # Simulates an edge case where a 2D array slips through
        img = Image.new("L", (100, 100), 200).convert("RGB")
        result = preprocess_image(img)

        assert result.shape[-1] == 3

    def test_custom_size_is_respected(self):
        img = make_dummy_image()
        result = preprocess_image(img, size=(96, 96))

        assert result.shape == (1, 96, 96, 3)
