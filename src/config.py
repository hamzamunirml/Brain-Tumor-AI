"""
Central configuration for the Brain Tumor Classifier project.
Keep all paths, constants, and per-model settings here so nothing
is hardcoded/duplicated across app.py, src modules, or tests.

NOTE: this matches the models actually produced by
brain_tumor_deep_learning.ipynb — 3 deep learning models (Custom CNN,
MobileNetV2, ResNet50), each saved in native Keras 3 `.keras` format.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Image / class settings
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Tumor"]  # index 0 -> Healthy, index 1 -> Tumor
# (Matches the notebook's LabelEncoder: alphabetical -> Healthy=0, Tumor=1)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# IMPORTANT: unlike the older classical-ML setup, these deep learning models
# do NOT need an app-side "rescale" flag. In brain_tumor_deep_learning.ipynb
# every model bakes its own normalization directly into the graph:
#   - Custom CNN   -> `layers.Rescaling(1/255)` as the first layer
#   - MobileNetV2  -> `mobilenet_v2.preprocess_input` as the first layer
#   - ResNet50     -> `resnet50.preprocess_input` as the first layer
# So the app always feeds RAW 0-255 float32 pixels to every model — see
# src/preprocessing.py::preprocess_image().
MODEL_CONFIGS = {
    "ResNet50": {
        "filename": "resnet50_best.keras",
        "description": "Transfer learning (ImageNet) — best overall: 96.6% accuracy, 0.993 AUC",
    },
    "MobileNetV2": {
        "filename": "mobilenetv2_best.keras",
        "description": "Transfer learning (ImageNet), lightweight and fast",
    },
    "Custom CNN": {
        "filename": "custom_cnn_best.keras",
        "description": "Custom CNN trained from scratch",
    },
}

# ResNet50 was selected as the best model in the notebook (highest F1/AUC
# on the held-out test set — see models/best_model_metadata.json).
DEFAULT_MODEL = "ResNet50"
