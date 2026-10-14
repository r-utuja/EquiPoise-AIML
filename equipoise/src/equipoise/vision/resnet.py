"""ResNet18 transfer learning for OCT classification (the CNN baseline for M10).

A ResNet18 pretrained on ImageNet already knows generic edges and textures. We replace its
last layer with a 4-class head (CNV, DME, DRUSEN, NORMAL), train only that head for the
first epoch (the backbone is frozen), then fine-tune the whole network with AdamW. The
grayscale OCT scan is repeated over three channels and normalized with ImageNet
statistics, because the pretrained filters expect that input. The best epoch by validation
accuracy is kept. Seeds fix initialization and batch order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def transforms(train: bool):
    from torchvision import transforms as T

    steps = [T.Grayscale(num_output_channels=3)]
    if train:
        steps.append(T.RandomHorizontalFlip())
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def build_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def _loader(folder: Path, train: bool, batch: int, seed: int) -> DataLoader:
    from torchvision.datasets import ImageFolder

    ds = ImageFolder(str(folder), transform=transforms(train))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=train, generator=g, num_workers=0)


@dataclass
class TrainResult:
    model: nn.Module
    classes: list[str]
    history: list[dict[str, float]]
    best_epoch: int


def _run_epoch(model, loader, loss_fn, opt=None) -> tuple[float, float]:
    model.train(opt is not None)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((out.argmax(1) == y).sum())
            total += len(y)
    return loss_sum / total, correct / total


def train(data_dir: Path, p: dict[str, Any], seed: int, pretrained: bool = True) -> TrainResult:
    """Fine-tune on data_dir/train, select on data_dir/val."""
    torch.manual_seed(seed)
    tr = _loader(data_dir / "train", True, int(p["batch_size"]), seed)
    va = _loader(data_dir / "val", False, int(p["batch_size"]), seed)
    classes = list(tr.dataset.classes)
    model = build_model(len(classes), pretrained)
    loss_fn = nn.CrossEntropyLoss()
    freeze = int(p["freeze_backbone_epochs"])
    history, best, best_state = [], -1.0, None
    for epoch in range(1, int(p["epochs"]) + 1):
        frozen = epoch <= freeze
        for name, prm in model.named_parameters():
            prm.requires_grad = (not frozen) or name.startswith("fc.")
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        tl, ta = _run_epoch(model, tr, loss_fn, opt)
        vl, vacc = _run_epoch(model, va, loss_fn)
        history.append({"epoch": epoch, "backbone_frozen": frozen, "train_loss": tl,
                        "train_acc": ta, "val_loss": vl, "val_acc": vacc})
        if vacc > best:
            best, best_state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_epoch = int(max(history, key=lambda h: h["val_acc"])["epoch"])
    return TrainResult(model.eval(), classes, history, best_epoch)


@torch.no_grad()
def predict_folder(model: nn.Module, folder: Path, batch: int) -> tuple[np.ndarray, np.ndarray,
                                                                        list[str]]:
    """(true labels, class probabilities, file paths) for an ImageFolder directory."""
    loader = _loader(folder, False, batch, 0)
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x), 1).numpy())
        ys.append(y.numpy())
    paths = [s[0] for s in loader.dataset.samples]
    return np.concatenate(ys), np.concatenate(ps), paths


def gradcam_overlay(model: nn.Module, img224: np.ndarray, target: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM heatmap on the last residual block; returns (heatmap 0-1, RGB overlay)."""
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    x = transforms(False)(Image.fromarray(img224)).unsqueeze(0)
    targets = None if target is None else [ClassifierOutputTarget(int(target))]
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        heat = cam(input_tensor=x, targets=targets)[0]
    rgb = np.repeat(img224[..., None], 3, axis=2).astype(np.float32) / 255.0
    return heat, show_cam_on_image(rgb, heat, use_rgb=True)
