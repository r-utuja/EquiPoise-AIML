"""Principal component analysis computed directly from the singular value decomposition.

After centring the standardized feature matrix X, the SVD X = U S V^T gives everything:
the rows of V^T are the principal directions, U*S are each patient's coordinates (scores),
and S^2 / (n - 1) are the variances along each direction. We implement this with
`numpy.linalg.svd` and check it against `sklearn.decomposition.PCA` (identical up to the
arbitrary sign of each component).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class SVDPCA:
    """Fitted PCA: mean, components (rows), per-component variance and its share."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray

    def transform(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        """Project onto the first k components (all if k is None)."""
        comps = self.components if k is None else self.components[:k]
        return (np.asarray(X, float) - self.mean) @ comps.T


def svd_pca(X: np.ndarray) -> SVDPCA:
    """PCA via thin SVD of the centred data; signs fixed so each component's largest loading
    is positive (makes results reproducible and comparable with sklearn)."""
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    signs = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    signs[signs == 0] = 1
    Vt = Vt * signs[:, None]
    var = S**2 / (len(X) - 1)
    return SVDPCA(mean, Vt, var, var / var.sum(), S)


def compare_with_sklearn(X: np.ndarray, ours: SVDPCA) -> dict[str, float]:
    """Max absolute differences between our SVD-PCA and sklearn's PCA (after sign alignment)."""
    sk = PCA(svd_solver="full").fit(X)
    k = min(len(sk.components_), len(ours.components))
    signs = np.sign(np.sum(sk.components_[:k] * ours.components[:k], axis=1))
    return {
        "max_abs_diff_components": float(np.abs(sk.components_[:k] * signs[:, None]
                                                - ours.components[:k]).max()),
        "max_abs_diff_explained_variance": float(np.abs(sk.explained_variance_[:k]
                                                        - ours.explained_variance[:k]).max()),
        "max_abs_diff_scores": float(np.abs(sk.transform(X)[:, :k] * signs
                                            - ours.transform(X, k)).max()),
    }
