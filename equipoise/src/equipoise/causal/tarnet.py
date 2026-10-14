"""TARNet: a neural network with a shared representation and one outcome head per drug.

The shared layers (2 x 64) learn a representation of the patient that is useful for all
arms; each head (1 x 32) predicts the outcome under its own drug. During training a patient
only contributes to the head of the arm they actually received (the factual outcome), because
the other outcomes are never observed. The individual effect is head(drug) - head(bevacizumab).
We train it with SGD (with momentum), Adam and RMSprop from identical initial weights and
batches, and stop early when the validation loss stops improving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from equipoise.causal.learners import CATEModel
from equipoise.config import Config

OPTIMIZERS = ("SGD", "Adam", "RMSprop")


class TARNet(nn.Module):
    def __init__(self, n_in: int, n_arms: int, shared: list[int], head: list[int],
                 dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in shared:
            layers += [nn.Linear(width, h), nn.ELU(), nn.Dropout(dropout)]
            width = h
        self.shared = nn.Sequential(*layers)
        heads = []
        for _ in range(n_arms):
            hl: list[nn.Module] = []
            w = width
            for h in head:
                hl += [nn.Linear(w, h), nn.ELU()]
                w = h
            hl.append(nn.Linear(w, 1))
            heads.append(nn.Sequential(*hl))
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.shared(x)
        return torch.cat([h(z) for h in self.heads], dim=1)   # (n, n_arms) potential outcomes


@dataclass
class TrainResult:
    optimizer: str
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = np.inf


def _make_optimizer(name: str, params, cfg: Config) -> torch.optim.Optimizer:
    t = cfg.causal["tarnet"]
    lr, wd = float(t["learning_rates"][name]), float(t["weight_decay"])
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=float(t["momentum"]), weight_decay=wd)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)


def train_tarnet(X: np.ndarray, T: np.ndarray, Y: np.ndarray, cfg: Config, optimizer: str,
                 n_arms: int) -> tuple[TARNet, TrainResult, tuple[float, float]]:
    """Train on factual outcomes with early stopping on a stratified validation split.

    Returns the model (best epoch weights), the loss curves and the (mean, sd) used to
    standardize Y. Same seed -> same initial weights, split and batch order for every optimizer.
    """
    t = cfg.causal["tarnet"]
    idx_fit, idx_val = train_test_split(np.arange(len(Y)), test_size=float(
        t["validation_fraction"]), random_state=cfg.seed, stratify=T)
    mu, sd = float(Y[idx_fit].mean()), float(Y[idx_fit].std())
    Xt = torch.tensor(X, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.long)
    Yt = torch.tensor((Y - mu) / sd, dtype=torch.float32)

    torch.manual_seed(cfg.seed)
    model = TARNet(X.shape[1], n_arms, list(t["shared"]), list(t["head"]), float(t["dropout"]))
    opt = _make_optimizer(optimizer, model.parameters(), cfg)
    rng = np.random.default_rng(cfg.seed)
    res = TrainResult(optimizer)
    best_state, stale = None, 0
    bs = int(t["batch_size"])

    def factual_loss(idx: np.ndarray) -> torch.Tensor:
        out = model(Xt[idx])
        return torch.mean((out[torch.arange(len(idx)), Tt[idx]] - Yt[idx]) ** 2)

    for epoch in range(int(t["max_epochs"])):
        model.train()
        order = rng.permutation(idx_fit)
        total = 0.0
        for start in range(0, len(order), bs):
            batch = order[start:start + bs]
            opt.zero_grad()
            loss = factual_loss(batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val = float(factual_loss(idx_val))
        res.train_loss.append(total / len(idx_fit))
        res.val_loss.append(val)
        if val < res.best_val_loss - 1e-6:
            res.best_val_loss, res.best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(t["patience"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, res, (mu, sd)


class TARNetLearner(CATEModel):
    name = "TARNet (Adam)"

    def __init__(self, cfg: Config, optimizer: str = "Adam") -> None:
        super().__init__(cfg)
        self.optimizer = optimizer
        self.name = f"TARNet ({optimizer})"

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> TARNetLearner:
        self.model_, self.result_, (self.mu_, self.sd_) = train_tarnet(
            X, T, Y, self.cfg, self.optimizer, self.n_arms)
        return self

    def potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model_(torch.tensor(X, dtype=torch.float32)).numpy()
        return out * self.sd_ + self.mu_

    def effect(self, X: np.ndarray) -> np.ndarray:
        po = self.potential_outcomes(X)
        return po[:, 1:] - po[:, [0]]
