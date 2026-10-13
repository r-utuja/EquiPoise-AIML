"""M10 stage: OCT screening on the Kermany dataset (optional, separate from Protocol T).

Prepares a balanced, patient-grouped subset with OpenCV preprocessing, fine-tunes a YOLO
classification model and a ResNet18 baseline with the same budget, reports accuracy and
per-class precision / recall / F1 with confusion matrices on the official test split, and
draws Grad-CAM heatmaps for the ResNet. These images are NOT linked to Protocol T
patients; the module is a screening front-end demo, never an input to the treatment model.
"""

from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from equipoise import tracking
from equipoise.config import Config
from equipoise.plotting import SEQUENTIAL, SERIES
from equipoise.vision import resnet, yolo
from equipoise.vision.data import (
    assign_splits,
    load_manifest,
    patient_overlap,
    prepare,
    prepared_dir,
    raw_dir,
    summary,
)
from equipoise.vision.preprocess import before_after_figure, read_gray

STAGE = "vision"
NOTE = "Kermany OCT dataset: not linked to Protocol T patients"


def models_dir(cfg: Config):
    return cfg.paths.models_dir / "oct"


def metrics_rows(name: str, y: np.ndarray, prob: np.ndarray, classes: list[str]
                 ) -> tuple[dict, pd.DataFrame]:
    pred = prob.argmax(1)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(len(classes)),
                                                 zero_division=0)
    per = pd.DataFrame({"model": name, "class": classes, "precision": p, "recall": r,
                        "f1": f, "support": n})
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return ({"Model": name, "Accuracy": accuracy_score(y, pred), "Macro precision": mp,
             "Macro recall": mr, "Macro F1": mf, "n_test": len(y)}, per)


def _confusion_plot(results: dict[str, tuple[np.ndarray, np.ndarray]], classes: list[str]
                    ) -> plt.Figure:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6))
    for ax, (name, (y, prob)) in zip(np.atleast_1d(axes), results.items(), strict=True):
        cm = confusion_matrix(y, prob.argmax(1), labels=range(len(classes)))
        ax.imshow(cm, cmap=SEQUENTIAL)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(name)
        ax.grid(False)
    fig.suptitle(f"Confusion matrices on the official test split ({NOTE})", fontsize=10)
    fig.tight_layout()
    return fig


def _curves_plot(history: list[dict]) -> plt.Figure:
    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(h.epoch, h.train_loss, "o-", color=SERIES[0], label="train")
    a1.plot(h.epoch, h.val_loss, "o-", color=SERIES[1], label="validation")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Cross-entropy loss")
    a1.legend()
    a2.plot(h.epoch, h.train_acc, "o-", color=SERIES[0], label="train")
    a2.plot(h.epoch, h.val_acc, "o-", color=SERIES[1], label="validation")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Accuracy")
    a2.legend()
    fig.suptitle("ResNet18 fine-tuning (epoch 1: backbone frozen, head only)")
    fig.tight_layout()
    return fig


def _gradcam_plot(model, items: list[tuple[str, str, int, int]], classes: list[str]
                  ) -> plt.Figure:
    fig, axes = plt.subplots(2, len(items), figsize=(2.6 * len(items), 5.6))
    for c, (path, true_cls, true_k, pred_k) in enumerate(items):
        img = read_gray(path)
        _, overlay = resnet.gradcam_overlay(model, img, pred_k)
        for r, view in enumerate([img, overlay]):
            ax = axes[r, c]
            ax.imshow(view, cmap="gray" if r == 0 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        axes[0, c].set_title(f"true {true_cls}\npred {classes[pred_k]}", fontsize=9,
                             color="black" if true_k == pred_k else SERIES[7])
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.suptitle(f"Where the ResNet looks (Grad-CAM, last residual block); {NOTE}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    if synthetic:
        raise ValueError("M10 uses the Kermany OCT images; there is no synthetic version")
    v = cfg.raw["vision"]
    p = v["preprocess"]
    size = int(p["image_size"])
    torch.manual_seed(cfg.seed)
    df = assign_splits(load_manifest(cfg), float(v["val_fraction"]), cfg.seed)
    data = prepare(df, cfg, prepared_dir(cfg))
    out = models_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    manifest = raw_dir(cfg) / "manifest.csv"
    with tracking.start_run(cfg, STAGE, manifest, synthetic, tags={"dataset": NOTE}):
        tracking.log_params({"vision": v})
        tracking.log_table(cfg, STAGE, summary(df), "dataset_summary")
        tracking.log_table(cfg, STAGE, patient_overlap(df), "patient_overlap")
        ex = df[df.model_split == "train"].groupby("cls").head(1)
        tracking.log_figure(cfg, STAGE, before_after_figure(list(ex.abs_path), list(ex.cls), p),
                            "preprocessing")

        # ResNet18 ---------------------------------------------------------------------
        rp = v["resnet"]
        res = resnet.train(data, rp, cfg.seed)
        classes = res.classes
        tracking.log_table(cfg, STAGE, pd.DataFrame(res.history), "resnet_history")
        tracking.log_figure(cfg, STAGE, _curves_plot(res.history), "resnet_training")
        y_r, prob_r, paths = resnet.predict_folder(res.model, data / "test",
                                                   int(rp["batch_size"]))
        torch.save({"state_dict": res.model.state_dict(), "classes": classes,
                    "image_size": size}, out / "resnet18.pt")

        # YOLO -------------------------------------------------------------------------
        yp = v["yolo"]
        ymodel, best = yolo.train(data, yp, size, cfg.seed, out)
        tracking.log_table(cfg, STAGE, yolo.history(out), "yolo_history")
        y_y, prob_y = yolo.predict_folder(ymodel, data / "test", classes, size)
        shutil.copy2(best, out / "yolo_cls_best.pt")

        # comparison -------------------------------------------------------------------
        results = {"ResNet18 (transfer learning)": (y_r, prob_r),
                   f"YOLO classification ({yp['checkpoint']})": (y_y, prob_y)}
        rows, per = [], []
        for name, (y, prob) in results.items():
            row, per_cls = metrics_rows(name, y, prob, classes)
            rows.append(row)
            per.append(per_cls)
        comp = pd.DataFrame(rows)
        tracking.log_table(cfg, STAGE, comp, "comparison")
        tracking.log_table(cfg, STAGE, pd.concat(per, ignore_index=True), "per_class")
        tracking.log_figure(cfg, STAGE, _confusion_plot(results, classes), "confusion")

        # Grad-CAM: correct and wrong examples per class ---------------------------------
        pred = prob_r.argmax(1)
        items = []
        k_ex = int(v["gradcam_examples"])
        for k, cls in enumerate(classes):
            idx = np.where(y_r == k)[0]
            right = [i for i in idx if pred[i] == k][:k_ex]
            wrong = [i for i in idx if pred[i] != k][:1]
            items += [(paths[i], cls, k, int(pred[i])) for i in right + wrong]
        tracking.log_figure(cfg, STAGE, _gradcam_plot(res.model, items, classes), "gradcam")

        (out / "meta.json").write_text(json.dumps({
            "classes": classes, "image_size": size, "preprocess": p, "note": NOTE,
            "resnet": "resnet18.pt", "yolo": "yolo_cls_best.pt"}, indent=2), encoding="utf-8")
        for r in rows:
            key = "resnet" if r["Model"].startswith("ResNet") else "yolo"
            metrics[f"{key}_test_accuracy"] = float(r["Accuracy"])
            metrics[f"{key}_test_macro_f1"] = float(r["Macro F1"])
        metrics["resnet_best_epoch"] = float(res.best_epoch)
        metrics["n_train_images"] = float((df.model_split == "train").sum())
        metrics["n_test_images"] = float((df.model_split == "test").sum())
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
