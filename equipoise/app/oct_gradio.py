"""Gradio demo for M10 OCT screening. Run: `python app/oct_gradio.py` (http://localhost:7861).

Upload a retinal OCT B-scan and see the ResNet18 and YOLO class probabilities with a
Grad-CAM heatmap. Uses the public Kermany dataset: not linked to Protocol T patients and
not part of the treatment recommendation. Research prototype, not for clinical use.
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from equipoise.vision.infer import NOT_PROTOCOL_T, OctScreener, example_images

SCREENER = OctScreener.load()


def screen(image: np.ndarray):
    if image is None:
        return None, None, None, None
    out = SCREENER.predict(image)
    return out["preprocessed"], out["gradcam"], out["resnet"], out.get("yolo")


demo = gr.Interface(
    fn=screen,
    inputs=gr.Image(type="numpy", label="OCT B-scan"),
    outputs=[gr.Image(label="Preprocessed (median + CLAHE, 224 px)"),
             gr.Image(label="Grad-CAM (ResNet18)"),
             gr.Label(label="ResNet18 probabilities"),
             gr.Label(label="YOLO-cls probabilities")],
    examples=[[str(p)] for p in example_images()],
    title="Equipoise M10: OCT screening demo",
    description=NOT_PROTOCOL_T,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_port=7861)
