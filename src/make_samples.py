"""
Copy a few held-out test images into samples/ so the app has clickable examples.

    python src/make_samples.py

None of these images were used for training.
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def save(src, name):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.thumbnail((512, 512))
    im.save(C.SAMPLES_DIR / name, quality=90)


def main():
    random.seed(7)
    C.SAMPLES_DIR.mkdir(exist_ok=True)
    for label in C.PLANTVILLAGE_CLASSES.values():
        slug = label.lower().replace(" ", "_")
        lab = sorted((C.DATA_DIR / "test" / label).iterdir())
        for i, f in enumerate(random.sample(lab, 2)):
            save(f, f"lab_{slug}_{i + 1}.jpg")
        field = sorted((C.DATA_DIR / "field_test" / label).iterdir())
        for i, f in enumerate(random.sample(field, 2)):
            save(f, f"field_{slug}_{i + 1}.jpg")
    oos = sorted((C.DATA_DIR / "field_out_of_scope" / "Tomato leaf mosaic virus").iterdir())
    save(random.choice(oos), "field_not_trained_mosaic_virus.jpg")
    print("Samples written to", C.SAMPLES_DIR)


if __name__ == "__main__":
    main()
