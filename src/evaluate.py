"""
Step 3 - Test the trained model on images it has NEVER seen.

    python src/evaluate.py

Three test sets:
  A. Lab test      - held-out PlantVillage photos (same style as training)
  B. Field test    - PlantDoc photos taken in real fields (different style)
  C. Out-of-scope  - field photos of tomato diseases we did NOT train on
                     (the honest answer for these is "Uncertain")

Outputs go to results/: confusion matrices, per-class precision / recall / F1,
a threshold analysis and a CSV of every prediction.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import plots as P
from predictor import LeafScreener, load_image

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_folder(root: Path, class_names=None):
    """Return images, integer labels (or folder names) and file paths."""
    images, labels, paths = [], [], []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                images.append(load_image(f))
            except Exception as e:  # a few web images are broken
                print("   skipped unreadable", f.name, e)
                continue
            labels.append(class_names.index(sub.name) if class_names else sub.name)
            paths.append(f)
    return np.stack(images), labels, paths


def screening_summary(y_true, probs, threshold, n_classes):
    top = probs.argmax(1); conf = probs.max(1)
    confident = conf >= threshold
    correct = top == np.asarray(y_true)
    return {
        "images": int(len(y_true)),
        "accuracy_top1": float(correct.mean()),
        "flagged_uncertain": float((~confident).mean()),
        "accuracy_on_confident": float(correct[confident].mean()) if confident.any() else None,
        "wrong_but_confident": int((~correct & confident).sum()),
        "wrong_and_flagged": int((~correct & ~confident).sum()),
    }


def evaluate_set(screener, name, title, folder, names):
    print(f"\n== {title} ==")
    X, y, paths = load_folder(folder, names)
    probs = screener.probabilities(X)
    pred = probs.argmax(1); conf = probs.max(1)
    n = len(names)
    th = screener.threshold

    # 1) Standard confusion matrix (top-1 class)
    cm = confusion_matrix(y, pred, labels=range(n))
    P.confusion_matrix_plot(cm, names, names, f"{title}: confusion matrix (top-1)",
                            C.RESULTS_DIR / f"confusion_matrix_{name}.png")

    # 2) Screening confusion matrix (adds an 'Uncertain' column)
    outcome = np.where(conf >= th, pred, n)
    cm_s = confusion_matrix(y, outcome, labels=range(n + 1))[:n]
    P.confusion_matrix_plot(cm_s, names, names + ["Uncertain"],
                            f"{title}: screening outcome (threshold {th:.0%})",
                            C.RESULTS_DIR / f"screening_matrix_{name}.png",
                            note="'Uncertain' = confidence below threshold -> sent for expert inspection")

    report = classification_report(y, pred, labels=range(n), target_names=names,
                                   digits=3, zero_division=0)
    report_dict = classification_report(y, pred, labels=range(n), target_names=names,
                                        output_dict=True, zero_division=0)
    summary = screening_summary(y, probs, th, n)
    print(report)
    print(json.dumps(summary, indent=2))
    (C.RESULTS_DIR / f"classification_report_{name}.txt").write_text(
        f"{title}\n{'=' * len(title)}\n\n{report}\n"
        f"Screening at threshold {th:.0%}:\n{json.dumps(summary, indent=2)}\n")

    with open(C.RESULTS_DIR / f"predictions_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true_class", "predicted_class", "confidence", "outcome"] + names)
        for p_, yt, pr, cf, pv in zip(paths, y, pred, conf, probs):
            w.writerow([p_.name, names[yt], names[pr], f"{cf:.4f}",
                        names[pr] if cf >= th else "Uncertain"] + [f"{v:.4f}" for v in pv])
    return {"summary": summary, "per_class": report_dict, "confusion_matrix": cm.tolist(),
            "screening_matrix": cm_s.tolist()}, y, probs


def evaluate_out_of_scope(screener, names):
    folder = C.DATA_DIR / "field_out_of_scope"
    if not folder.exists():
        return None
    print("\n== Out-of-scope diseases (never trained) ==")
    X, diseases, _ = load_folder(folder)
    probs = screener.probabilities(X)
    conf = probs.max(1); pred = probs.argmax(1)
    result = {}
    for d in sorted(set(diseases)):
        idx = [i for i, v in enumerate(diseases) if v == d]
        flagged = float((conf[idx] < screener.threshold).mean())
        counts = {names[k]: int((pred[idx] == k).sum()) for k in range(len(names))}
        result[d] = {"images": len(idx), "flagged_uncertain": flagged, "top1_guesses": counts}
        print(f"   {d:28s} n={len(idx):3d} flagged uncertain={flagged:.0%}  guesses={counts}")
    result["_overall_flagged_uncertain"] = float((conf < screener.threshold).mean())
    return result


def threshold_sweep(curves):
    """How the threshold trades 'cases sent to experts' against accuracy."""
    ths = [round(t, 2) for t in np.arange(0.40, 0.96, 0.05)]
    rows = {}
    acc_series, cov_series = {}, {}
    for label, (y, probs) in curves.items():
        y = np.asarray(y); conf = probs.max(1); correct = probs.argmax(1) == y
        accs, covs = [], []
        for t in ths:
            keep = conf >= t
            accs.append(float(correct[keep].mean()) if keep.any() else 1.0)
            covs.append(float(keep.mean()))
        acc_series[f"{label}: accuracy when confident"] = accs
        cov_series[f"{label}: share auto-screened"] = covs
        rows[label] = [{"threshold": t, "accuracy_on_confident": a, "share_confident": c}
                       for t, a, c in zip(ths, accs, covs)]
    P.line_plot(ths, {**acc_series, **cov_series},
                "Choosing the uncertainty threshold", "Confidence threshold", "Share of images",
                C.RESULTS_DIR / "threshold_analysis.png", vline=C.CONFIDENCE_THRESHOLD,
                vlabel=f"chosen = {C.CONFIDENCE_THRESHOLD:.2f}")
    return rows


def main():
    C.RESULTS_DIR.mkdir(exist_ok=True)
    screener = LeafScreener()
    names = screener.class_names
    out = {"classes": names, "threshold": screener.threshold}

    out["lab_test"], y_lab, p_lab = evaluate_set(
        screener, "lab_test", "Lab test (PlantVillage held-out)", C.DATA_DIR / "test", names)
    curves = {"Lab": (y_lab, p_lab)}
    if (C.DATA_DIR / "field_test").exists():
        out["field_test"], y_f, p_f = evaluate_set(
            screener, "field_test", "Field test (PlantDoc real photos)",
            C.DATA_DIR / "field_test", names)
        curves["Field"] = (y_f, p_f)
    out["out_of_scope"] = evaluate_out_of_scope(screener, names)
    out["threshold_sweep"] = threshold_sweep(curves)

    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\nAll results written to", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
