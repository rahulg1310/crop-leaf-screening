"""
Step 1 - Download the datasets and split them into train / val / test.

    python src/prepare_data.py

What it does
1. Downloads ONLY the tomato folders of two public datasets (needs git):
   - PlantVillage (lab photos, plain background)  -> used for training
   - PlantDoc     (real field photos)             -> used ONLY for testing
2. Takes up to 950 PlantVillage images per class and splits them 70/15/15.
3. Copies the PlantDoc tomato photos into data/field_test (never trained on).

Result:
    data/train/<class>/...      data/val/<class>/...      data/test/<class>/...
    data/field_test/<class>/... data/field_out_of_scope/<disease>/...
"""
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

PLANTVILLAGE_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def run(cmd, cwd=None):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sparse_clone(url, dest: Path, patterns):
    """Clone only the folders we need (saves gigabytes)."""
    if dest.exists():
        print(f"  {dest.name} already downloaded - skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)])
    run(["git", "sparse-checkout", "set", "--no-cone", *patterns], cwd=dest)


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def copy_all(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out_dir / f.name)


def main():
    print("1) Downloading datasets ...")
    pv = C.RAW_DIR / "PlantVillage-Dataset"
    pd = C.RAW_DIR / "PlantDoc-Dataset"
    sparse_clone(PLANTVILLAGE_REPO, pv,
                 [f"/raw/color/{name}/" for name in C.PLANTVILLAGE_CLASSES])
    sparse_clone(PLANTDOC_REPO, pd, ["/test/Tomato*", "/train/Tomato*"])

    print("2) Splitting PlantVillage into train / val / test ...")
    rng = random.Random(C.SEED)
    for split in ("train", "val", "test"):
        shutil.rmtree(C.DATA_DIR / split, ignore_errors=True)
    for folder, label in C.PLANTVILLAGE_CLASSES.items():
        files = list_images(pv / "raw" / "color" / folder)
        rng.shuffle(files)
        files = files[: C.MAX_IMAGES_PER_CLASS]
        n_train = int(len(files) * C.SPLIT["train"])
        n_val = int(len(files) * C.SPLIT["val"])
        parts = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, chunk in parts.items():
            copy_all(chunk, C.DATA_DIR / split / label)
        print(f"   {label:13s} train={len(parts['train'])} "
              f"val={len(parts['val'])} test={len(parts['test'])}")

    print("3) Collecting PlantDoc field photos (test only) ...")
    for out in ("field_test", "field_out_of_scope"):
        shutil.rmtree(C.DATA_DIR / out, ignore_errors=True)
    for folder, label in C.PLANTDOC_CLASSES.items():
        files = []
        for split in ("train", "test"):  # both PlantDoc splits are unseen by our model
            d = pd / split / folder
            if d.exists():
                files += list_images(d)
        copy_all(files, C.DATA_DIR / "field_test" / label)
        print(f"   {label:13s} field photos={len(files)}")
    for folder in C.PLANTDOC_OUT_OF_SCOPE:
        files = []
        for split in ("train", "test"):
            d = pd / split / folder
            if d.exists():
                files += list_images(d)
        copy_all(files, C.DATA_DIR / "field_out_of_scope" / folder)
        print(f"   (not trained) {folder}: {len(files)}")

    print("Done. Data is in", C.DATA_DIR)


if __name__ == "__main__":
    main()
