"""
Image preprocessing utilities.

Keeping this in its own module (instead of inline in app.py) means the
exact same preprocessing logic used at inference time can be unit-tested
independently of Streamlit and of any trained model file.

NOTE: as of brain_tumor_deep_learning.ipynb, every saved model (Custom CNN,
MobileNetV2, ResNet50) bakes its own normalization directly into the model
graph (a Rescaling layer or that architecture's `preprocess_input`, applied
as the very first layer). So preprocess_image() here only resizes and
returns RAW 0-255 pixel values — do NOT divide by 255 in the app, or the
model would effectively be double-normalized and give wrong predictions.
"""

from typing import Dict

import numpy as np
from PIL import Image

from src.config import IMG_SIZE


def load_image(file) -> Image.Image:
    """
    Load an image from a file path or a file-like object (e.g. Streamlit's
    UploadedFile) and convert it to RGB (drops alpha channel / grayscale
    quirks that some CT/MRI exports have).
    """
    img = Image.open(file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def preprocess_image(img: Image.Image, size=IMG_SIZE) -> np.ndarray:
    """
    Resize a PIL image into a model-ready batch of shape (1, H, W, 3).

    Returns RAW pixel values in [0, 255] (float32) — every model in this
    project normalizes internally, so no rescale/mean-subtraction happens
    here. Passing an already-rescaled array to these models would silently
    produce wrong predictions.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (any size, any mode - call load_image() first to
        guarantee RGB).
    size : tuple
        Target (width, height) to resize to. Defaults to the project's
        IMG_SIZE (224, 224), matching every model's expected input shape.

    Returns
    -------
    np.ndarray of shape (1, size[0], size[1], 3), dtype float32, range [0, 255]
    """
    img = img.resize(size)
    arr = np.array(img).astype("float32")

    if arr.ndim == 2:  # safety net for any remaining grayscale images
        arr = np.stack([arr] * 3, axis=-1)

    return np.expand_dims(arr, axis=0)


def detect_scan_type(img: Image.Image) -> Dict:
    """
    Best-effort HEURISTIC guess of whether an uploaded brain scan is a
    CT or an MRI image.

    IMPORTANT: This is NOT a trained classifier — there is no labeled
    CT-vs-MRI dataset/model in this project. It's a rule-of-thumb based
    on well-known visual differences between the two modalities:

      - CT images: bone (skull) is very bright/high-intensity (high HU),
        producing a distinct near-white ring around the brain, and the
        background outside the skull is typically pure black (air).
      - MRI images: bone is usually dark/low-signal on most sequences
        (cortical bone has very little proton density), so there is
        rarely a bright bone ring; soft tissue instead shows a wider
        range of mid-gray contrast.

    NOTE: an earlier version of this heuristic also scored "colorfulness"
    (how far the R/G/B channels diverge per pixel). That signal turned out
    to be close to 0 for virtually every medical scan — both CT and MRI
    are almost always grayscale-derived when exported as JPG/PNG — so it
    was silently biasing every image toward "CT" regardless of content.
    It has been removed in favor of `bright_ratio`, which empirically
    separates the two modalities much better.

    Because real-world scans vary a lot (windowing, sequence type,
    contrast agents, etc.), this heuristic is still approximate. Treat
    the output as a hint, not a diagnosis, and prefer a properly trained
    classifier for production use.

    Returns
    -------
    dict with keys:
        - scan_type: "CT" or "MRI"
        - confidence: float (0-1), heuristic confidence score
        - is_heuristic: True (always, flags this isn't a trained model)
    """
    arr = np.array(img.convert("RGB")).astype("float32")
    gray = arr.mean(axis=-1)

    # 1. Bright-pixel ratio: fraction of near-white pixels (>200/255).
    #    CT's bright bone ring pushes this noticeably higher than MRI,
    #    where bone is typically dark. This is the primary signal.
    bright_ratio = float(np.mean(gray > 200))

    # 2. Background darkness ratio: fraction of near-black pixels.
    #    A weak secondary signal — CT backgrounds tend to be a very
    #    uniform pure black (air), slightly more often than MRI.
    dark_ratio = float(np.mean(gray < 15))

    # --- weighted scoring (heuristic, not learned) ---
    ct_score = 0.0
    mri_score = 0.0

    if bright_ratio > 0.02:
        ct_score += 2.0
    else:
        mri_score += 1.5

    if dark_ratio > 0.40:
        ct_score += 0.5
    else:
        mri_score += 0.5

    total = ct_score + mri_score
    if total == 0:
        scan_type, confidence = "MRI", 0.5
    elif ct_score >= mri_score:
        scan_type, confidence = "CT", ct_score / total
    else:
        scan_type, confidence = "MRI", mri_score / total

    return {
        "scan_type": scan_type,
        "confidence": round(confidence, 2),
        "is_heuristic": True,
    }
