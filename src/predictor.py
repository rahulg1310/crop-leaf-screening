"""
Shared prediction logic used by the app, the evaluation and the CLI.

    python src/predictor.py path/to/leaf.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

UNCERTAIN_LABEL = "Uncertain - expert inspection required"


def load_image(source, size=C.IMG_SIZE):
    """Open a file path or PIL image, fix phone rotation, make it RGB 224x224."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = ImageOps.exif_transpose(img)          # phone photos are often rotated
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)    # values 0-255; model rescales itself


class LeafScreener:
    def __init__(self, model_path=C.MODEL_PATH, threshold=C.CONFIDENCE_THRESHOLD):
        import tensorflow as tf  # imported here so `--help` etc. stay fast
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = json.loads(Path(C.CLASS_NAMES_PATH).read_text())
        self.threshold = threshold

    def probabilities(self, images):
        """images: array (N, 224, 224, 3) -> array (N, n_classes)."""
        return self.model.predict(np.asarray(images), verbose=0)

    def decide(self, probs):
        """Turn one probability vector into a screening outcome."""
        top = int(np.argmax(probs))
        confidence = float(probs[top])
        predicted = self.class_names[top]
        uncertain = confidence < self.threshold
        return {
            "predicted_class": predicted,
            "confidence": confidence,
            "uncertain": uncertain,
            "outcome": UNCERTAIN_LABEL if uncertain else predicted,
            "all_scores": {n: float(p) for n, p in zip(self.class_names, probs)},
        }

    def predict(self, source):
        probs = self.probabilities(load_image(source)[None])[0]
        return self.decide(probs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predictor.py <image> [<image> ...]")
        sys.exit(1)
    screener = LeafScreener()
    for path in sys.argv[1:]:
        r = screener.predict(path)
        print(f"{path}\n  Outcome   : {r['outcome']}\n"
              f"  Top class : {r['predicted_class']} ({r['confidence']:.1%})")
        for name, p in sorted(r["all_scores"].items(), key=lambda x: -x[1]):
            print(f"    {name:13s} {p:6.1%}")
    print("\n" + C.DISCLAIMER)
