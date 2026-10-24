"""
Central settings for the Tomato Leaf Screening prototype.
Change values here instead of editing every script.
"""
import os
from pathlib import Path

# ---- Folders --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
# prepared train/val/test images (not in git); set LEAF_DATA_DIR to keep them elsewhere
DATA_DIR = Path(os.environ.get("LEAF_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # downloaded datasets (not in git)
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "samples"

MODEL_PATH = MODEL_DIR / "tomato_leaf_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# ---- Crop and classes -----------------------------------------------------
CROP = "Tomato"

# Folder name in PlantVillage  ->  friendly label shown in the app
PLANTVILLAGE_CLASSES = {
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___healthy": "Healthy",
}

# Folder name in PlantDoc (real field photos, used ONLY for testing)
PLANTDOC_CLASSES = {
    "Tomato Early blight leaf": "Early Blight",
    "Tomato leaf late blight": "Late Blight",
    "Tomato mold leaf": "Leaf Mold",
    "Tomato leaf": "Healthy",
}
# Diseases the model was NEVER trained on (used to test the "uncertain" rule)
PLANTDOC_OUT_OF_SCOPE = [
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
]

# ---- Data split -----------------------------------------------------------
MAX_IMAGES_PER_CLASS = 950          # keeps the classes balanced
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

# ---- Model ----------------------------------------------------------------
IMG_SIZE = 224                      # MobileNetV2's native input size
BATCH_SIZE = 32
EPOCHS = 8                          # phase 1: train only the new final layer
LEARNING_RATE = 1e-3
FINE_TUNE_EPOCHS = 6                # phase 2: also adjust the top of MobileNetV2
FINE_TUNE_LAYERS = 40               # how many of MobileNetV2's last layers to unfreeze
FINE_TUNE_LEARNING_RATE = 1e-5      # small, so we don't erase what it already knows

# ---- Screening rule -------------------------------------------------------
# If the top probability is below this, the app says
# "Uncertain - expert inspection required".
CONFIDENCE_THRESHOLD = 0.70

# Simple triage order for the batch view (1 = inspect first).
# Late blight can destroy a field within days, and uncertain cases need a
# human to look, so both go to the top. Adjust with your field officers.
PRIORITY = {
    "Uncertain": 1,
    "Late Blight": 1,
    "Early Blight": 2,
    "Leaf Mold": 2,
    "Healthy": 3,
}
PRIORITY_TEXT = {1: "1 - Inspect first", 2: "2 - Inspect soon", 3: "3 - Routine"}

DISCLAIMER = (
    "This tool is a screening aid that supports, and does not replace, "
    "advice from agricultural experts. Always confirm with a field officer "
    "or plant-pathology expert before treating a crop."
)
