"""
Run the whole pipeline in order (about 15-30 minutes on a laptop CPU):

    python run_all.py

1. prepare_data  - download tomato images and split train / val / test
2. train         - transfer learning with MobileNetV2
3. evaluate      - confusion matrices, per-class scores, field test
4. stress_test   - lighting and background robustness
5. make_samples  - example images for the app
Then start the app with:  python app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["prepare_data", "train", "evaluate", "stress_test", "make_samples"]

for step in STEPS:
    print(f"\n========== {step} ==========")
    subprocess.run([sys.executable, str(ROOT / "src" / f"{step}.py")], check=True)

print("\nAll steps finished. Start the app with:  python app.py")
