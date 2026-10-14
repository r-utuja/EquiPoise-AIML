"""Hidden Markov model of visit-to-visit disease activity, with forward and Viterbi from scratch.

We assume each visit is in one of three hidden states (e.g. improving, active, stable) that
we cannot see directly; we only see the change in vision and in CST since the previous visit,
drawn from a Gaussian whose mean depends on the state. The state moves between visits
according to a transition matrix. hmmlearn estimates the means, variances and transitions
with the EM (Baum-Welch) algorithm. The forward algorithm sums the probability of the
observations over every possible state path (the likelihood); Viterbi finds the single most
likely path. We implement both in numpy (in log space, so long sequences do not underflow)
and check them against hmmlearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from equipoise.config import Config

FEATURES = ("delta_va", "delta_cst")


@dataclass
class Sequences:
    X: np.ndarray            # stacked observations (n_obs, 2)
    lengths: list[int]       # per patient
    frame: pd.DataFrame      # one row per observation: pt_id, treatment, week, injected


def visit_sequences(visits: pd.DataFrame, pt_ids: np.ndarray) -> Sequences:
    """Per-visit changes since the previous visit (study eye), visits with VA and CST only."""
    v = visits[visits.pt_id.isin(pt_ids)].dropna(subset=["va_letters", "cst_um"])
    v = v.sort_values(["pt_id", "days_from_rand"]).copy()
    v["delta_va"] = v.groupby("pt_id").va_letters.diff()
    v["delta_cst"] = v.groupby("pt_id").cst_um.diff()
    v = v.dropna(subset=["delta_va", "delta_cst"])
    lengths = v.groupby("pt_id", sort=True).size()
    lengths = lengths[lengths >= 2]
    v = v[v.pt_id.isin(lengths.index)]
    frame = v[["pt_id", "treatment", "days_from_rand", "injected"]].assign(
        week=(v.days_from_rand / 7).round(1)).reset_index(drop=True)
    return Sequences(v[list(FEATURES)].to_numpy(float), lengths.tolist(), frame)


def fit_hmm(seq: Sequences, cfg: Config) -> GaussianHMM:
    h = cfg.longitudinal["hmm"]
    model = GaussianHMM(n_components=int(h["n_states"]), covariance_type=str(h["covariance_type"]),
                        n_iter=int(h["n_iter"]), tol=float(h["tol"]), random_state=cfg.seed)
    return model.fit(seq.X, seq.lengths)


def label_states(model: GaussianHMM) -> dict[int, str]:
    """improving = highest mean vision change; of the other two, stable = smaller total
    variance, active = the other (bigger swings)."""
    means = model.means_[:, 0]
    improving = int(np.argmax(means))
    rest = [k for k in range(len(means)) if k != improving]
    var = [float(np.sum(model.covars_[k].diagonal() if model.covars_[k].ndim == 2
                        else model.covars_[k])) for k in rest]
    stable = rest[int(np.argmin(var))]
    labels = {improving: "improving", stable: "stable"}
    for k in rest:
        labels.setdefault(k, "active")
    return labels


def _log_emissions(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """log N(x_t | mean_k, diag cov_k) for every observation t and state k."""
    out = np.empty((len(X), model.n_components))
    for k in range(model.n_components):
        var = np.diag(model.covars_[k]) if model.covars_[k].ndim == 2 else model.covars_[k]
        out[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - model.means_[k]) ** 2 / var,
                                  axis=1)
    return out


def forward_loglik(model: GaussianHMM, X: np.ndarray) -> float:
    """Forward algorithm in log space: log P(x_1..x_T) for one sequence."""
    logB = _log_emissions(model, X)
    logA = np.log(model.transmat_)
    alpha = np.log(model.startprob_) + logB[0]
    for t in range(1, len(X)):
        alpha = logsumexp(alpha[:, None] + logA, axis=0) + logB[t]
    return float(logsumexp(alpha))


def viterbi(model: GaussianHMM, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Most likely state path and its log-probability (dynamic programming + backtracking)."""
    logB = _log_emissions(model, X)
    logA = np.log(model.transmat_)
    delta = np.log(model.startprob_) + logB[0]
    back = np.zeros((len(X), model.n_components), dtype=int)
    for t in range(1, len(X)):
        scores = delta[:, None] + logA
        back[t] = np.argmax(scores, axis=0)
        delta = scores[back[t], np.arange(model.n_components)] + logB[t]
    path = np.empty(len(X), dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(len(X) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(np.max(delta))


def split_sequences(seq: Sequences) -> list[np.ndarray]:
    bounds = np.cumsum([0, *seq.lengths])
    return [seq.X[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
