"""Seed every random number generator the project uses.

Call `seed_everything(cfg.seed)` at the start of each stage. It seeds Python's `random`,
NumPy's legacy global generator and, if installed, PyTorch (CPU and CUDA, deterministic
algorithms). scikit-learn and Optuna are seeded explicitly through `random_state` / sampler
`seed` arguments, which this module provides via `rng()`.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed random, numpy and torch so results reproduce run-to-run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed for libraries that use it
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng(seed: int) -> np.random.Generator:
    """A fresh NumPy Generator; prefer this over the global state in new code."""
    return np.random.default_rng(seed)
