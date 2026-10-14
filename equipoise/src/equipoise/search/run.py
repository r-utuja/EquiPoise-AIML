"""M4 stage: filter / embedded / wrapper rankings and the metaheuristic search arena.

Training patients only. The selected subset is reported, and its held-out RMSE is shown once
for final reporting; it is not forced onto M6 (M6 reports whether using it changes results).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equipoise import tracking
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.splits import get_split
from equipoise.plotting import INK, INK_2, NEUTRAL, SEQUENTIAL, SERIES
from equipoise.preprocess.encode import build_preprocessor
from equipoise.search.algorithms import ALGORITHMS, SearchResult, run_algorithm
from equipoise.search.objective import (
    BudgetExhausted,
    Evaluator,
    column_owner,
    make_evaluator,
    ridge_fit_predict,
)
from equipoise.search.selection import rank_features

STAGE = "search"


def _short(f: str) -> str:
    return f.replace("X_", "")


def _convergence_plot(results: list[SearchResult], budget: int, baseline: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, name in enumerate(ALGORITHMS):
        runs = [r for r in results if r.algorithm == name]
        curves = np.full((len(runs), budget), np.nan)
        for j, r in enumerate(runs):
            h = np.asarray(r.history)
            curves[j, :len(h)] = h
            curves[j, len(h):] = h[-1]          # finished early (converged / cycling)
            ax.plot(np.arange(1, budget + 1), curves[j], color=SERIES[i], lw=0.7, alpha=0.35)
        ax.plot(np.arange(1, budget + 1), np.median(curves, axis=0), color=SERIES[i], lw=2,
                label=name)
    ax.axhline(baseline, color=NEUTRAL, ls="--", lw=1)
    ax.text(budget, baseline, "all features ", ha="right", va="bottom", fontsize=8,
            color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Unique objective evaluations (log scale)")
    ax.set_ylabel("Best objective so far\n(CV RMSE + penalty x features)")
    ax.set_title("Search arena: convergence (thick = median over seeds, thin = each seed)")
    ax.legend(ncol=2, fontsize=8)
    lo = min(min(r.history) for r in results)
    ax.set_ylim(lo - 0.05, baseline + 0.4)
    fig.tight_layout()
    return fig


def _hill_climb_plot(diag: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for _, r in diag.iterrows():
        ax.plot([0, 1], [r.start_value, r.local_optimum], color=SERIES[0], alpha=0.5, lw=1)
    ax.scatter(np.zeros(len(diag)), diag.start_value, color=NEUTRAL, s=20, zorder=3,
               label="random start")
    ax.scatter(np.ones(len(diag)), diag.local_optimum, color=SERIES[1], s=30, zorder=3,
               label="local optimum reached")
    ax.set_xticks([0, 1], ["start", "end of climb"])
    ax.set_ylabel("Objective (lower is better)")
    n_distinct = diag.local_optimum.round(6).nunique()
    ax.set_title(f"Hill climbing from {len(diag)} random starts: "
                 f"{n_distinct} distinct local optima")
    ax.legend()
    fig.tight_layout()
    return fig


def _rank_heatmap(ranks: pd.DataFrame, top: int) -> plt.Figure:
    cols = [c for c in ranks.columns if c.endswith("(rank)")]
    sub = ranks.head(top)
    M = sub[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 0.32 * top + 1.6))
    ax.imshow(M, cmap=SEQUENTIAL.reversed(), aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                color="white" if v <= 5 else INK)
    ax.set_xticks(range(len(cols)), [c.replace(" (rank)", "").replace("_", " ") for c in cols],
                  rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(sub)), [_short(f) for f in sub.feature], fontsize=8)
    ax.grid(False)
    ax.set_title(f"Feature ranks by method (1 = most important), top {top} by mean rank\n"
                 "RFE: every feature kept in its final subset shares rank 1", fontsize=10)
    fig.tight_layout()
    return fig


def _lasso_path_plot(extras: dict, top_features: list[str]) -> plt.Figure:
    alphas, coefs, owners = extras["lasso_alphas"], extras["lasso_coefs"], extras["owners"]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for j, own in enumerate(owners):
        if own in top_features[:6] and np.abs(coefs[j]).max() > 0:
            color = SERIES[top_features.index(own) % len(SERIES)]
            ax.plot(alphas, coefs[j], color=color, lw=1.8)
            col = extras["column_names"][j].split("__", 1)[1]
            ax.text(alphas[-1], coefs[j][-1], f" {_short(col)}", fontsize=7, color=INK_2,
                    va="center")

        else:
            ax.plot(alphas, coefs[j], color=NEUTRAL, lw=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Lasso penalty alpha (log scale; weaker penalty to the right)")
    ax.set_ylabel("Coefficient (letters per SD)")
    ax.set_title("Lasso path: features entering the model as the penalty weakens")
    fig.tight_layout()
    return fig


def _hill_climb_diagnostic(ev: Evaluator, n_starts: int, seed: int) -> pd.DataFrame:
    """Independent climbs from random starts (each with its own budget) to expose local
    optima: where does steepest descent stop?"""
    rows = []
    rng = np.random.default_rng(seed)
    for k in range(n_starts):
        e = ev.fresh()
        start = (rng.random(e.n_features) < 0.5).astype(np.int8)
        start_val = e(start)
        cur, val, steps = start, start_val, 0
        try:
            while True:
                vals = []
                for i in range(e.n_features):
                    m = cur.copy()
                    m[i] ^= 1
                    vals.append((e(m), m))
                best_val, best_m = min(vals, key=lambda t: t[0])
                if best_val >= val:
                    break
                cur, val, steps = best_m, best_val, steps + 1
        except BudgetExhausted:
            pass
        rows.append({"start": k, "start_value": start_val, "local_optimum": val,
                     "n_features": int(cur.sum()), "steps": steps,
                     "features": ", ".join(_short(f) for f, b in
                                           zip(ev.data.features, cur, strict=True) if b)})
    return pd.DataFrame(rows)


def _holdout_rmse(train: pd.DataFrame, test: pd.DataFrame, mask: np.ndarray, ev: Evaluator,
                  cfg: Config) -> float:
    tr = train.dropna(subset=[cfg.primary_outcome])
    te = test.dropna(subset=[cfg.primary_outcome])
    pre = build_preprocessor(cfg, imputer="median", include_treatment=True).fit(tr)
    owners = np.array(column_owner(list(pre.get_feature_names_out()), ev.data.features),
                      dtype=object)
    keep = {f for f, b in zip(ev.data.features, mask, strict=True) if b}
    cols = np.flatnonzero([o is None or o in keep for o in owners])
    pred = ridge_fit_predict(pre.transform(tr)[:, cols], tr[cfg.primary_outcome].to_numpy(float),
                             pre.transform(te)[:, cols], ev.alpha)
    return float(np.sqrt(np.mean((te[cfg.primary_outcome].to_numpy(float) - pred) ** 2)))


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M4 and return headline metrics."""
    s = cfg.search
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    train, test = split.train(df), split.test(df)
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"search": s, "n_train": len(train)})

        # --- filter / embedded / wrapper ---------------------------------------------
        ranks, extras = rank_features(train, cfg)
        tracking.log_table(cfg, STAGE, ranks, "feature_rankings")
        tracking.log_figure(cfg, STAGE, _rank_heatmap(ranks, 15), "feature_rankings")
        tracking.log_figure(cfg, STAGE, _lasso_path_plot(extras, list(ranks.feature)),
                            "lasso_path")
        top_k = int(s["filter_top_k"])
        tracking.log_table(cfg, STAGE, pd.DataFrame({
            m: list(ranks.sort_values(f"{m} (rank)").feature.head(top_k).map(_short))
            for m in ["mutual_information", "abs_correlation", "chi_square_gain15",
                      "lasso_entry_alpha", "rf_importance", "rfe_rank"]}), "top_features_by_method")

        # --- search arena --------------------------------------------------------------
        ev = make_evaluator(train, cfg)
        full = np.ones(ev.n_features, np.int8)
        ref = ev.fresh()
        all_value, none_value = ref(full), ref(np.zeros(ev.n_features, np.int8))
        results: list[SearchResult] = []
        for name in ALGORITHMS:
            for off in s["seed_offsets"]:
                r = run_algorithm(name, ev, cfg.seed + int(off), s)
                results.append(r)
                with tracking.child_run(f"search: {name} (seed {r.seed})",
                                        {"algorithm": name, "seed": r.seed}):
                    tracking.log_metrics({"best_objective": r.best_value,
                                          "n_features": int(r.best_mask.sum()),
                                          "n_evals": r.n_evals})
        rows = []
        for r in results:
            hist = np.asarray(r.history)
            rows.append({
                "algorithm": r.algorithm, "seed": r.seed, "best_objective": r.best_value,
                "cv_rmse": ev.fresh().cv_rmse(r.best_mask), "n_features": int(r.best_mask.sum()),
                "evals_used": r.n_evals,
                "evals_to_best": int(np.argmax(hist <= r.best_value + 1e-12) + 1),
                "stopped_by": r.diagnostics.get("stopped_by"),
                "features": ", ".join(_short(f) for f, b in
                                      zip(ev.data.features, r.best_mask, strict=True) if b)})
        runs = pd.DataFrame(rows)
        tracking.log_table(cfg, STAGE, runs, "search_runs")
        summary = runs.groupby("algorithm", sort=False).agg(
            best_objective_min=("best_objective", "min"),
            best_objective_mean=("best_objective", "mean"),
            cv_rmse_of_best=("cv_rmse", "min"), n_features_mean=("n_features", "mean"),
            evals_to_best_median=("evals_to_best", "median"),
            evals_used_mean=("evals_used", "mean")).reset_index()
        summary.insert(1, "budget", int(s["budget"]))
        tracking.log_table(cfg, STAGE, summary, "search_summary")
        tracking.log_figure(cfg, STAGE, _convergence_plot(results, int(s["budget"]), all_value),
                            "convergence")

        hc = _hill_climb_diagnostic(ev, 10, cfg.seed)
        tracking.log_table(cfg, STAGE, hc, "hill_climbing_restarts")
        tracking.log_figure(cfg, STAGE, _hill_climb_plot(hc), "hill_climbing_restarts")

        # --- selected subset -------------------------------------------------------------
        best = min(results, key=lambda r: (round(r.best_value, 10), int(r.best_mask.sum())))
        freq = pd.DataFrame({"feature": [_short(f) for f in ev.data.features],
                             "selected_in_best_subset": best.best_mask.astype(bool),
                             "share_of_runs_selecting": np.mean(
                                 [r.best_mask for r in results], axis=0)})
        freq = freq.sort_values(["selected_in_best_subset", "share_of_runs_selecting"],
                                ascending=False)
        tracking.log_table(cfg, STAGE, freq, "selected_features")
        comp = pd.DataFrame([
            {"subset": "all baseline features", "n_features": ev.n_features,
             "cv_objective": all_value, "cv_rmse": ev.fresh().cv_rmse(full),
             "heldout_rmse": _holdout_rmse(train, test, full, ev, cfg)},
            {"subset": f"selected ({best.algorithm})", "n_features": int(best.best_mask.sum()),
             "cv_objective": best.best_value, "cv_rmse": ev.fresh().cv_rmse(best.best_mask),
             "heldout_rmse": _holdout_rmse(train, test, best.best_mask, ev, cfg)},
            {"subset": "none (arm only)", "n_features": 0, "cv_objective": none_value,
             "cv_rmse": none_value, "heldout_rmse": _holdout_rmse(
                 train, test, np.zeros(ev.n_features, np.int8), ev, cfg)},
        ])
        tracking.log_table(cfg, STAGE, comp, "selected_vs_all")
        metrics.update({
            "objective_all_features": all_value, "objective_best": best.best_value,
            "n_selected": int(best.best_mask.sum()),
            "heldout_rmse_all": float(comp.heldout_rmse.iloc[0]),
            "heldout_rmse_selected": float(comp.heldout_rmse.iloc[1]),
            "hill_climb_distinct_optima": int(hc.local_optimum.round(6).nunique())})
        for _, r in summary.iterrows():
            metrics[f"best_{r.algorithm.lower().replace(' ', '_')}"] = r.best_objective_min
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
