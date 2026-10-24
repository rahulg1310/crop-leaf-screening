"""
Tomato Leaf Screening - upload-and-predict web app.

    python app.py            -> open http://127.0.0.1:7860 in your browser

Tab 1: upload one leaf photo -> predicted class, confidence and all scores.
Tab 2: upload many photos   -> a table sorted by inspection priority.
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
from predictor import LeafScreener, UNCERTAIN_LABEL

screener = LeafScreener()

CSS = """
.result-card {border-radius: 12px; padding: 18px 20px; border: 1px solid #d9d8d3;}
.result-card h2 {margin: 4px 0 6px 0; font-size: 1.6rem;}
.result-card p {margin: 2px 0; color: #52514e;}
.ok      {background: #eef6ee; border-color: #0ca30c;}
.disease {background: #fdf1ea; border-color: #ec835a;}
.unsure  {background: #fff6df; border-color: #fab219;}
.disclaimer {font-size: 0.9rem; color: #52514e; border-left: 4px solid #fab219;
             padding: 8px 12px; background: #fffaf0;}
"""


def result_card(r):
    conf = f"{r['confidence']:.1%}"
    if r["uncertain"]:
        css, icon, head = "unsure", "&#9888;", UNCERTAIN_LABEL
        detail = (f"Best guess was <b>{r['predicted_class']}</b> at only {conf} "
                  f"(below the {C.CONFIDENCE_THRESHOLD:.0%} threshold). "
                  "Please send this case to a field officer.")
    elif r["predicted_class"] == "Healthy":
        css, icon, head = "ok", "&#10004;", "Healthy"
        detail = f"Confidence {conf}. No disease pattern detected."
    else:
        css, icon, head = "disease", "&#9679;", r["predicted_class"]
        detail = f"Confidence {conf}. Possible disease - arrange inspection."
    priority = C.PRIORITY_TEXT[C.PRIORITY["Uncertain" if r["uncertain"] else r["predicted_class"]]]
    return (f"<div class='result-card {css}'><p>Screening outcome</p>"
            f"<h2>{icon} {head}</h2><p>{detail}</p>"
            f"<p><b>Inspection priority:</b> {priority}</p></div>")


def predict_one(image):
    if image is None:
        return "<p>Upload a tomato leaf photo to start.</p>", None
    r = screener.predict(image)
    return result_card(r), r["all_scores"]


def predict_many(files):
    if not files:
        return []
    rows = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        r = screener.predict(path)
        key = "Uncertain" if r["uncertain"] else r["predicted_class"]
        rows.append([C.PRIORITY_TEXT[C.PRIORITY[key]], Path(path).name, r["outcome"],
                     r["predicted_class"], round(r["confidence"] * 100, 1)])
    rows.sort(key=lambda row: (row[0], -row[4] if row[2] != UNCERTAIN_LABEL else row[4]))
    return rows


examples = sorted(str(p) for p in C.SAMPLES_DIR.glob("*.jpg"))

# Gradio 6 takes CSS in launch(); older versions take it in Blocks().
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

with gr.Blocks(title="Tomato Leaf Screening", **({} if GRADIO_6 else {"css": CSS})) as demo:
    gr.Markdown(
        f"# Tomato Leaf Disease Screening\n"
        f"Early screening aid for field officers. Crop: **{C.CROP}**. "
        f"Classes: **{', '.join(screener.class_names)}**. "
        f"Predictions below **{C.CONFIDENCE_THRESHOLD:.0%}** confidence are marked "
        f"*Uncertain - expert inspection required*.")
    gr.HTML(f"<div class='disclaimer'><b>Important:</b> {C.DISCLAIMER}</div>")

    with gr.Tab("Single photo"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Leaf photo (one leaf, close-up)",
                               sources=["upload", "webcam", "clipboard"], height=340)
                btn = gr.Button("Screen this leaf", variant="primary")
            with gr.Column():
                card = gr.HTML("<p>Upload a tomato leaf photo to start.</p>")
                scores = gr.Label(label="Confidence for each class", num_top_classes=4)
        btn.click(predict_one, img, [card, scores])
        img.change(predict_one, img, [card, scores])
        if examples:
            gr.Examples(examples, inputs=img, label="Sample test images (not used in training)")

    with gr.Tab("Batch triage"):
        gr.Markdown("Upload several photos received from farmers. The table is sorted so "
                    "the cases to inspect first are at the top.")
        files = gr.File(file_count="multiple", file_types=["image"], label="Leaf photos")
        run = gr.Button("Screen all", variant="primary")
        table = gr.Dataframe(headers=["Priority", "File", "Outcome", "Top class", "Confidence %"],
                             interactive=False, wrap=True)
        run.click(predict_many, files, table)

    gr.Markdown("Model: MobileNetV2 (transfer learning) trained on PlantVillage tomato "
                "leaves. See `results/` and `docs/REPORT.md` for test results and limitations.")

if __name__ == "__main__":
    demo.launch(**({"css": CSS} if GRADIO_6 else {}))
