# Tomato Leaf Disease Screening (AI prototype)

An early-screening aid for a farmer cooperative in Telangana. Field officers
upload phone photos of **tomato** leaves and the app says which leaves look
diseased, how confident it is, and which cases to inspect first.

> **This tool supports, and does not replace, advice from agricultural experts.**
> Every result must be confirmed by a field officer or plant-pathology expert
> before any treatment decision.

| | |
|---|---|
| **Crop** | Tomato |
| **Classes** | Healthy, Early Blight, Late Blight, Leaf Mold |
| **Model** | MobileNetV2 (ImageNet pre-trained) + new 4-class layer (transfer learning) |
| **Training data** | PlantVillage (public, lab photos), 665 images per class |
| **Unseen test data** | 572 held-out PlantVillage photos + 353 real field photos (PlantDoc) + 389 photos of diseases it was never trained on |
| **Uncertain rule** | top confidence < 70% -> "Uncertain - expert inspection required" |
| **Interface** | Gradio web page (single photo + batch triage) |

## Quick start (Windows)

You need **Python 3.10 - 3.13** (python.org, tick "Add Python to PATH").

```bat
cd %USERPROFILE%\Documents\crop-leaf-screening
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860** in your browser. The trained model is already in
`models/`, so the app works without retraining.

### Example

1. Click the sample `lab_early_blight_1.jpg`
   -> **Early Blight**, confidence 99.7%, priority *2 - Inspect soon*.
2. Click `field_healthy_1.jpg` (a real field photo)
   -> best guess Early Blight at only 60% -> **Uncertain - expert inspection required**.
3. Click `field_not_trained_mosaic_virus.jpg` (a disease the model never learned)
   -> wrongly says Early Blight at 72.8%. This shows a real limitation (see report).
4. Batch triage tab: drop all the samples -> table sorted with the most urgent on top.

## Re-create everything from scratch

```bat
python run_all.py
```

That runs these steps (each can also be run on its own):

| Step | Command | What it does |
|---|---|---|
| 1 | `python src/prepare_data.py` | Downloads only the tomato folders (needs `git`), splits 70/15/15 |
| 2 | `python src/train.py` | Trains the model (about 15-25 min on a laptop CPU) |
| 3 | `python src/evaluate.py` | Confusion matrices, per-class precision/recall/F1, field test |
| 4 | `python src/stress_test.py` | Lighting and background robustness test |
| 5 | `python src/make_samples.py` | Puts example images in `samples/` |

Predict from the command line:

```bat
python src/predictor.py samples\field_early_blight_1.jpg
```

## Project layout

```
crop-leaf-screening/
|-- app.py                 Gradio upload-and-predict interface
|-- run_all.py             runs the full pipeline
|-- requirements.txt
|-- src/
|   |-- config.py          classes, threshold, paths, disclaimer (edit here)
|   |-- prepare_data.py    download + split datasets
|   |-- train.py           transfer learning with MobileNetV2
|   |-- predictor.py       shared prediction + "uncertain" rule (+ CLI)
|   |-- evaluate.py        confusion matrices, per-class scores, threshold analysis
|   |-- stress_test.py     lighting / background / blur tests
|   |-- make_samples.py    example images for the app
|   `-- plots.py           chart style
|-- models/                trained model + class names
|-- results/               all test outputs (PNG, TXT, CSV, JSON)
|-- samples/               unseen example images
|-- docs/REPORT.md         results, limitations, responsible-use statement
`-- data/                  downloaded images (git-ignored)
```

## Results (unseen images only)

| Test set | Images | Accuracy | Sent to "Uncertain" | Accuracy when confident |
|---|---|---|---|---|
| Lab photos (PlantVillage held-out) | 572 | **92.3%** | 8.7% | 94.4% |
| Real field photos (PlantDoc) | 353 | **44.5%** | 20.7% | 48.6% |
| Diseases never trained on | 389 | - | 23.7% | - |

The big drop on field photos comes from lighting, background and dataset
differences. See **[docs/REPORT.md](docs/REPORT.md)** for confusion matrices,
per-class scores, the lighting/background stress test and the limitations.

## Datasets and credits

- PlantVillage: Hughes & Salathe (2015), github.com/spMohanty/PlantVillage-Dataset
- PlantDoc: Singh et al. (2020), github.com/pratikkayal/PlantDoc-Dataset
- MobileNetV2: Sandler et al. (2018), via Keras Applications
