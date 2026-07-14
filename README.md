# Brain Tumor Detector (CT & MRI)

<img width="1600" height="500" alt="image" src="https://github.com/user-attachments/assets/8561765b-5f97-499c-bc5d-54406caeb148" />


Streamlit web app that classifies brain CT/MRI scan images as **Healthy** or
**Tumor**, using models trained on the
[Brain Tumor Multimodal Image (CT & MRI)](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)
Kaggle dataset.

## Project structure

```
brain_tumor_classifier/
├── app.py                     # Streamlit app entry point
├── requirements.txt
├── src/
│   ├── config.py               # paths, class names, per-model settings
│   ├── preprocessing.py        # image loading + resize/normalize
│   └── model_utils.py          # model loading + prediction logic
├── tests/
│   ├── test_preprocessing.py
│   └── test_model_utils.py
└── models/                     # put downloaded .h5 files here (not included)
```

## Setup

Requires **Python 3.10.11**.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Add trained models

Download the `.h5` files from your Kaggle notebook's Output panel and place
them inside `models/`:

- `cnn_best.h5`
- `mobilenetv2_best.h5`
- `densenet121_best.h5`
- `efficientnetb0_best.h5` (optional)

## Run the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest tests/ -v
```

## Model performance (test set)

| Model          | Accuracy | AUC   |
|----------------|----------|-------|
| CNN            | 97.0%    | 0.995 |
| MobileNetV2    | 95.3%    | 0.990 |
| DenseNet121    | 94.5%    | 0.987 |
| EfficientNetB0 | 55.3%*   | 0.716*|

\* EfficientNetB0's low score was caused by double image normalization during
training and is expected to improve significantly (~90%+) once retrained
with the corrected preprocessing (see training notebook, Section 6 note).

## Disclaimer

Educational / portfolio project. Not intended for real medical diagnosis.
