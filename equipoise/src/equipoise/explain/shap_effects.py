"""SHAP explanations of the predicted treatment effect (aflibercept - bevacizumab).

SHAP splits one prediction into additive contributions, one per feature: starting from the
average predicted effect over background patients, each feature pushes this patient's effect
up or down, and the pushes add up exactly to the patient's prediction. We use the model-
agnostic permutation explainer on the effect function itself (so it works for whichever
learner M6 selected) with training patients as the background. Contributions of one-hot
columns are summed back to their original feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from equipoise.config import Config
from equipoise.preprocess.columns import feature_groups
from equipoise.search.objective import column_owner


def explain_effect(effect_model, background: np.ndarray, X: np.ndarray, names: list[str],
                   cfg: Config, comparison: int = 0) -> shap.Explanation:
    """Permutation SHAP of `effect_model.effect(X)[:, comparison]`."""
    def f(z: np.ndarray) -> np.ndarray:
        return effect_model.effect(z)[:, comparison]

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(f, masker, algorithm="permutation", feature_names=names,
                               seed=cfg.seed)
    return explainer(X, max_evals=2 * X.shape[1] + 1, silent=True)


def raw_feature_shap(values: np.ndarray, names: list[str], cfg: Config) -> pd.DataFrame:
    """(n_patients x raw features) SHAP values with one-hot columns summed per feature."""
    owners = column_owner([f"x__{n}" for n in names], list(feature_groups(cfg).all))
    return pd.DataFrame(values, columns=owners).T.groupby(level=0).sum().T


def global_importance(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per raw feature (letters), largest first."""
    imp = raw.abs().mean().sort_values(ascending=False)
    out = imp.rename("mean_abs_shap_letters").rename_axis("feature").reset_index()
    out["rank"] = np.arange(1, len(out) + 1)
    return out
