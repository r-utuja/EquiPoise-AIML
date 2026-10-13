"""Semi-synthetic benchmark: real patients' features, simulated outcomes with a known effect.

A patient's effect under the drugs they did not get is never observed, so we cannot score a
CATE learner on real outcomes. Instead we keep the real baseline features X, fit the
bevacizumab outcome surface f0(X) and the noise level to the real data, and simulate
Y(a) = f0(X) + tau_a(X) + noise with a treatment effect tau that we choose:
(a) null - the same effect for everyone (no heterogeneity), (b) published-like - aflibercept's
advantage concentrated in baseline vision < 69 letters, (c) nonlinear in baseline vision,
CST and HbA1c. Arms are re-randomized with the known probabilities in every replication.
PEHE (precision in estimating heterogeneous effects) is the RMSE between estimated and true
individual effects on patients not used for fitting. In (a) we also count how often a learner
"finds" heterogeneity that is not there (top-tertile minus bottom-tertile predicted effect
larger than the MCID).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

from equipoise.causal.data import CausalData, known_propensities
from equipoise.config import Config
from equipoise.preprocess.columns import CST, OCT_MACHINE
from equipoise.preprocess.harmonize import harmonizer_from_config

SCENARIOS = ("a_null", "b_published_like", "c_nonlinear")


@dataclass
class Simulator:
    """Outcome surface, noise and scenario effects fitted to the real training data."""

    f0: np.ndarray            # bevacizumab outcome surface for every patient
    noise_sd: float
    taus: dict[str, np.ndarray]   # scenario -> (n, n_comparisons) true effects


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def scenario_effects(frame: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """True effects (afl - bev, ran - bev) for each scenario, from raw baseline features."""
    s = cfg.causal["semi_synthetic"]
    arms = list(cfg.comparison_arms)
    n = len(frame)
    null = np.tile([s["null_effects"][a] for a in arms], (n, 1)).astype(float)

    below = (frame["X_va_below_69"] == 1).to_numpy()
    pub = s["published_effects"]
    published = np.where(below[:, None], [pub["va_below_69"][a] for a in arms],
                         [pub["va_69_or_better"][a] for a in arms]).astype(float)

    nl = s["nonlinear"]
    h = harmonizer_from_config(cfg.preprocess["cst"], "zscore").fit(frame[[CST, OCT_MACHINE]])
    cst_z = np.nan_to_num(h.transform(frame[[CST, OCT_MACHINE]])[:, 0])
    hba1c = frame["X_hba1c"].fillna(frame["X_hba1c"].median()).to_numpy(float)
    va = frame["X_va_letters"].to_numpy(float)
    afl = (float(nl["amplitude"]) * _sigmoid((float(nl["va_centre"]) - va) / float(nl["va_scale"]))
           * (0.5 + 0.5 * np.tanh(cst_z))
           + float(nl["hba1c_effect"]) * (hba1c > float(nl["hba1c_cut"])))
    nonlinear = np.column_stack([afl, float(nl["ranibizumab_share"]) * afl])
    return {"a_null": null, "b_published_like": published, "c_nonlinear": nonlinear}


def fit_simulator(data: CausalData, cfg: Config) -> Simulator:
    """f0 = Ridge fit on bevacizumab patients (applied to everyone); noise SD from its
    cross-validated residuals."""
    bev = data.T == 0
    alpha = float(cfg.causal["base_learners"]["ridge"]["alpha"])
    model = Ridge(alpha=alpha).fit(data.X[bev], data.Y[bev])
    resid = data.Y[bev] - cross_val_predict(Ridge(alpha=alpha), data.X[bev], data.Y[bev], cv=5)
    scale = float(cfg.causal["semi_synthetic"].get("noise_scale", 1.0))
    return Simulator(model.predict(data.X), scale * float(resid.std()),
                     scenario_effects(data.frame, cfg))


def simulate(sim: Simulator, scenario: str, rng: np.random.Generator, probs: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Randomize arms with the known probabilities; return (T, Y)."""
    n = len(sim.f0)
    T = rng.choice(len(probs), size=n, p=probs)
    tau = np.column_stack([np.zeros(n), sim.taus[scenario]])
    Y = sim.f0 + tau[np.arange(n), T] + rng.normal(0, sim.noise_sd, n)
    return T, Y


def tertile_gap(effect: np.ndarray) -> float:
    """Mean predicted effect in the top tertile minus the bottom tertile."""
    lo, hi = np.quantile(effect, [1 / 3, 2 / 3])
    top, bottom = effect[effect >= hi], effect[effect <= lo]
    return float(top.mean() - bottom.mean()) if len(top) and len(bottom) else 0.0


def run_benchmark(data: CausalData, learners: dict[str, Callable], cfg: Config
                  ) -> tuple[pd.DataFrame, Simulator]:
    """One row per scenario x replication x learner x comparison arm."""
    s = cfg.causal["semi_synthetic"]
    sim = fit_simulator(data, cfg)
    probs = known_propensities(cfg)
    mcid = float(cfg.mcid_letters)
    arms = list(cfg.comparison_arms)
    rows = []
    for si, scenario in enumerate(SCENARIOS):
        for rep in range(int(s["replications"])):
            rng = np.random.default_rng(cfg.seed + 1000 * si + rep)
            T, Y = simulate(sim, scenario, rng, probs)
            idx = rng.permutation(len(Y))
            n_eval = int(round(float(s["eval_fraction"]) * len(Y)))
            ev, fit = idx[:n_eval], idx[n_eval:]
            truth = sim.taus[scenario][ev]
            for name, factory in learners.items():
                est = factory(cfg).fit(data.X[fit], T[fit], Y[fit]).effect(data.X[ev])
                for k, arm in enumerate(arms):
                    err = est[:, k] - truth[:, k]
                    gap = tertile_gap(est[:, k])
                    rows.append({
                        "scenario": scenario, "replication": rep, "learner": name,
                        "comparison": f"{arm} - {cfg.reference_arm}",
                        "pehe": float(np.sqrt(np.mean(err**2))),
                        "ate_bias": float(est[:, k].mean() - truth[:, k].mean()),
                        "predicted_effect_sd": float(est[:, k].std()),
                        "tertile_gap": gap, "finds_heterogeneity": gap > mcid})
    return pd.DataFrame(rows), sim


def summarize(bench: pd.DataFrame) -> pd.DataFrame:
    """Mean (SD) PEHE per scenario x learner, averaged over both comparisons."""
    per = bench.groupby(["scenario", "learner", "replication"], as_index=False).agg(
        pehe=("pehe", "mean"), ate_bias=("ate_bias", "mean"),
        finds_heterogeneity=("finds_heterogeneity", "max"))
    return per.groupby(["scenario", "learner"], as_index=False).agg(
        pehe_mean=("pehe", "mean"), pehe_sd=("pehe", "std"),
        ate_bias_mean=("ate_bias", "mean"),
        false_heterogeneity_rate=("finds_heterogeneity", "mean"),
        replications=("replication", "nunique"))
