"""Single-image OCT screening for the Gradio demo and the Streamlit tab.

Loads the trained ResNet18 (and the YOLO model, when present) from `models/oct/`, applies
the same OpenCV preprocessing used in training, and returns class probabilities plus a
Grad-CAM overlay. The images come from a separate public dataset: this is a screening
front-end demo, not linked to Protocol T and never an input to the treatment model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from equipoise.config import Config, load_config
from equipoise.vision.preprocess import preprocess

NOT_PROTOCOL_T = ("OCT screening demo on the public Kermany dataset. These images are not "
                  "linked to Protocol T patients and are not used by the treatment model. "
                  "Research prototype, not for clinical use.")


def available(cfg: Config | None = None) -> bool:
    cfg = cfg or load_config()
    d = cfg.paths.models_dir / "oct"
    return (d / "meta.json").is_file() and (d / "resnet18.pt").is_file()


@dataclass
class OctScreener:
    resnet: Any
    yolo: Any
    classes: list[str]
    p: dict[str, Any]

    @classmethod
    def load(cls, cfg: Config | None = None) -> OctScreener:
        import torch

        from equipoise.vision.resnet import build_model

        cfg = cfg or load_config()
        d = cfg.paths.models_dir / "oct"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        ckpt = torch.load(d / meta["resnet"], map_location="cpu", weights_only=True)
        model = build_model(len(meta["classes"]), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        yolo_model = None
        if (d / meta["yolo"]).is_file():
            from equipoise.vision.yolo import _quiet_settings

            _quiet_settings()
            from ultralytics import YOLO

            yolo_model = YOLO(str(d / meta["yolo"]))
        return cls(model.eval(), yolo_model, list(meta["classes"]), dict(meta["preprocess"]))

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Probabilities per class for each model, plus the preprocessed image and Grad-CAM."""
        import torch
        from PIL import Image

        from equipoise.vision.resnet import gradcam_overlay, transforms

        img = preprocess(image, self.p)
        with torch.no_grad():
            x = transforms(False)(Image.fromarray(img)).unsqueeze(0)
            prob_r = torch.softmax(self.resnet(x), 1)[0].numpy()
        out: dict[str, Any] = {"preprocessed": img,
                               "resnet": dict(zip(self.classes, map(float, prob_r),
                                                  strict=True))}
        if self.yolo is not None:
            r = self.yolo.predict(np.repeat(img[..., None], 3, axis=2),
                                  imgsz=int(self.p["image_size"]), device="cpu",
                                  verbose=False)[0]
            names = self.yolo.names
            out["yolo"] = {names[i]: float(v) for i, v in enumerate(r.probs.data.numpy())}
        _, out["gradcam"] = gradcam_overlay(self.resnet, img, int(prob_r.argmax()))
        out["note"] = NOT_PROTOCOL_T
        return out


def example_images(cfg: Config | None = None, per_class: int = 1) -> list[Path]:
    """A few held-out test images (local raw subset), for demos."""
    cfg = cfg or load_config()
    root = cfg.paths.root / cfg.raw["vision"]["raw_dir"] / "test"
    return [p for c in sorted(root.glob("*")) for p in sorted(c.glob("*"))[:per_class]]
