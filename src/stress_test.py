"""
Step 4 - Lighting and background stress test.

    python src/stress_test.py

Takes the held-out lab test images (never used for training) and changes them
the way real phone photos differ: dim light, harsh sun, evening colour cast,
shadows, blur, a cluttered soil/grass background, and heavy compression.
Then measures how accuracy and the "uncertain" rate change.
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import plots as P
from predictor import LeafScreener

S = C.IMG_SIZE
EXTS = {".jpg", ".jpeg", ".png"}


# ---- The image changes ----------------------------------------------------
def dim_light(img):
    return ImageEnhance.Brightness(img).enhance(0.45)


def harsh_sun(img):
    img = ImageEnhance.Brightness(img).enhance(1.7)
    return ImageEnhance.Contrast(img).enhance(0.8)


def evening_cast(img):
    a = np.asarray(img).astype(np.float32)
    a[..., 0] *= 1.18; a[..., 1] *= 1.0; a[..., 2] *= 0.7
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def leaf_shadow(img, rng):
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    x = rng.randint(S // 4, 3 * S // 4)
    d.polygon([(x, 0), (S, 0), (S, S), (x - S // 3, S)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(8))
    dark = ImageEnhance.Brightness(img).enhance(0.35)
    return Image.composite(dark, img, mask)


def out_of_focus(img):
    return img.filter(ImageFilter.GaussianBlur(3))


def field_background(img, rng):
    """Shrink the leaf photo and place it on a busy soil-and-grass scene."""
    nr = np.random.default_rng(rng.randint(0, 10**6))
    base = np.zeros((S, S, 3), np.float32)
    base[...] = (110, 85, 60)                                    # soil brown
    base += nr.normal(0, 22, (S, S, 1))                          # grain
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(bg)
    for _ in range(45):                                          # grass / other leaves
        x, y = nr.integers(0, S, 2)
        w, h = nr.integers(6, 30, 2)
        g = int(nr.integers(90, 170))
        d.ellipse([x, y, x + w, y + h], fill=(int(g * 0.45), g, int(g * 0.3)))
    bg = bg.filter(ImageFilter.GaussianBlur(1.2))
    k = int(S * 0.6)
    small = img.resize((k, k))
    ox, oy = rng.randint(0, S - k), rng.randint(0, S - k)
    bg.paste(small, (ox, oy))
    return bg


def low_quality(img):
    small = img.resize((80, 80))
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=15)
    return Image.open(buf).convert("RGB").resize((S, S))


CONDITIONS = {
    "Original": lambda im, r: im,
    "Dim light": lambda im, r: dim_light(im),
    "Harsh sunlight": lambda im, r: harsh_sun(im),
    "Evening colour cast": lambda im, r: evening_cast(im),
    "Partial shadow": leaf_shadow,
    "Out of focus": lambda im, r: out_of_focus(im),
    "Cluttered background": field_background,
    "Low-quality photo": lambda im, r: low_quality(im),
}


def main(only=None):
    """only: optional list of condition names (run a few at a time on slow machines;
    results are merged into results/stress_test.json)."""
    screener = LeafScreener()
    names = screener.class_names
    items = []
    for k, n in enumerate(names):
        for f in sorted((C.DATA_DIR / "test" / n).iterdir()):
            if f.suffix.lower() in EXTS:
                items.append((Image.open(f).convert("RGB").resize((S, S), Image.BILINEAR), k))
    y = np.array([k for _, k in items])
    todo = [c for c in CONDITIONS if not only or c in only]
    print(f"Stress-testing {len(items)} held-out images under {len(todo)} conditions")

    out_json = C.RESULTS_DIR / "stress_test.json"
    saved = json.loads(out_json.read_text()) if (only and out_json.exists()) else {}
    for cname in todo:
        rng = random.Random(C.SEED)
        imgs = [CONDITIONS[cname](im, rng) for im, _ in items]
        probs = screener.probabilities(np.stack([np.asarray(i, np.float32) for i in imgs]))
        conf = probs.max(1); correct = probs.argmax(1) == y
        confident = conf >= screener.threshold
        saved[cname] = {
            "accuracy_top1": float(correct.mean()),
            "flagged_uncertain": float((~confident).mean()),
            "accuracy_on_confident": float(correct[confident].mean()) if confident.any() else None,
            "wrong_but_confident": float((~correct & confident).mean()),
        }
        print(f"   {cname:22s} acc={correct.mean():.1%}  uncertain={(~confident).mean():.1%}"
              f"  wrong-but-confident={(~correct & confident).mean():.1%}")

    results = {c: saved[c] for c in CONDITIONS if c in saved}   # keep a fixed order
    out_json.write_text(json.dumps(results, indent=2))
    groups = list(results)
    P.grouped_bar_plot(
        groups,
        {"Top-1 accuracy": [results[g]["accuracy_top1"] for g in groups],
         "Flagged uncertain": [results[g]["flagged_uncertain"] for g in groups],
         "Wrong but confident": [results[g]["wrong_but_confident"] for g in groups]},
        "Lighting and background stress test (held-out lab images)", "Share of images",
        C.RESULTS_DIR / "stress_test.png")

    example = items[5][0]
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(2.1 * len(CONDITIONS), 2.6))
    for ax, (cname, fn) in zip(axes, CONDITIONS.items()):
        ax.imshow(fn(example, random.Random(C.SEED)))
        ax.set_title(cname, fontsize=8, fontweight="normal"); ax.axis("off")
    fig.suptitle("The same unseen test leaf under each condition", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "stress_examples.png", dpi=120)
    print("Saved results/stress_test.png and results/stress_examples.png")


if __name__ == "__main__":
    # e.g.  python src/stress_test.py                      (all conditions)
    #       python src/stress_test.py "Dim light" "Out of focus"
    main(sys.argv[1:] or None)
