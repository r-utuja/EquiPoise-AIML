"""`build_preprocessor`: the one sklearn ColumnTransformer every later model uses.

Numeric features are prepared (log for skewed labs, CST harmonized across OCT machines,
DR severity ordinal-encoded), standardized, then imputed jointly (median, kNN or iterative
MICE-style) so an imputer can borrow strength across features. Nominal features are
mode-imputed and one-hot encoded; 0/1 flags are mode-imputed. Because it is a fitted sklearn
object placed inside a Pipeline, every statistic (medians, means, SDs, machine CST means,
neighbours) is learned from the training fold only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from equipoise.config import Config
from equipoise.features import TREATMENT_COL, assert_no_leakage
from equipoise.preprocess.columns import CST, OCT_MACHINE, ORDINAL, feature_groups
from equipoise.preprocess.harmonize import harmonizer_from_config

IMPUTERS = ("median", "knn", "iterative")


def make_imputer(kind: str, cfg: Config) -> Any:
    """Numeric imputer by name: median | knn | iterative (single, deterministic)."""
    p = cfg.preprocess
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "knn":
        return KNNImputer(n_neighbors=int(p["knn_neighbors"]))
    if kind == "iterative":
        return IterativeImputer(max_iter=int(p["mice_max_iter"]), random_state=cfg.seed)
    raise ValueError(f"unknown imputer {kind!r}; choose from {IMPUTERS}")


def build_preprocessor(cfg: Config, imputer: str | None = None, cst_method: str | None = None,
                       include_treatment: bool = False, full_rank: bool = False,
                       features: list[str] | None = None) -> ColumnTransformer:
    """ColumnTransformer over the X_ columns (plus one-hot treatment if requested).

    `treatment` is encoded as dummies for each non-reference arm (reference = bevacizumab).
    Columns not listed (Y_, M_, ids) are dropped, so outcomes cannot leak in.
    `full_rank=True` pools rare categories (`supervised.one_hot_min_frequency`) and drops one
    dummy per nominal feature, so unpenalized models with an intercept (OLS via the normal
    equations, IRLS logistic regression) have an invertible design matrix.
    `features` restricts the raw X_ columns used (e.g. a subset chosen by M4's search).
    """
    g = feature_groups(cfg)
    use_cst = True
    if features is not None:
        keep = set(features)
        g = replace(g, **{k: tuple(c for c in getattr(g, k) if c in keep)
                          for k in ("nominal", "ordinal", "binary", "log", "continuous")})
        use_cst = CST in keep
    cst_cols = [CST, OCT_MACHINE] if use_cst else []
    prepare = ColumnTransformer(
        [
            ("log", FunctionTransformer(np.log, feature_names_out="one-to-one"), list(g.log)),
            ("cst", harmonizer_from_config(cfg.preprocess["cst"], cst_method), cst_cols),
            ("ordinal", OrdinalEncoder(categories=[list(v) for v in ORDINAL.values()],
                                       handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       encoded_missing_value=np.nan), list(g.ordinal)),
            ("continuous", "passthrough", list(g.continuous)),
        ],
        verbose_feature_names_out=False,
    )
    numeric = Pipeline([
        ("prepare", prepare),
        ("scale", StandardScaler()),  # before kNN so distances weigh features equally
        ("impute", make_imputer(imputer or cfg.preprocess["imputer"], cfg)),
    ])
    onehot = (OneHotEncoder(drop="first", min_frequency=int(
        cfg.supervised.get("one_hot_min_frequency", 1)), handle_unknown="infrequent_if_exist",
        sparse_output=False) if full_rank
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot),
    ])
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", numeric, [*g.log, *cst_cols, *g.ordinal, *g.continuous]),
        ("nominal", nominal, list(g.nominal)),
        ("binary", SimpleImputer(strategy="most_frequent"), list(g.binary)),
    ]
    if include_treatment:
        arms = list(cfg.arms)
        transformers.append((
            "treatment",
            OneHotEncoder(categories=[arms], drop=[cfg.reference_arm], sparse_output=False),
            [TREATMENT_COL],
        ))
    inputs = {c for _, _, cols in transformers for c in cols}
    assert_no_leakage(inputs, allow_treatment=include_treatment)
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def one_hot_sparse(df: pd.DataFrame, cfg: Config) -> tuple[sp.csr_matrix, list[str]]:
    """Nominal features as a SciPy CSR matrix (the sparse one-hot block, for inspection)."""
    g = feature_groups(cfg)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X = enc.fit_transform(df[list(g.nominal)].astype(object).fillna("missing"))
    return sp.csr_matrix(X), list(enc.get_feature_names_out())
