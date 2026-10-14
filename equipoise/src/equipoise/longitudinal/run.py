"""M7 stage: trajectories, interpolation, finite differences, time series, HMM, early response.

Descriptive trajectories use all randomized patients (like M2). Everything that is fitted -
the HMM, trajectory clusters, the early-response model - uses training patients; the
early-response model is scored once on the held-out test set.
"""

from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange as scipy_lagrange

from equipoise import tracking
from equipoise.causal.bundle import EARLY_MODEL_NAME
from equipoise.config import Config
from equipoise.data.load import load_analysis, load_visits, visits_path
from equipoise.data.splits import get_split
from equipoise.longitudinal.clusters import cluster_trajectories
from equipoise.longitudinal.derivatives import SCHEMES, central_difference, error_orders
from equipoise.longitudinal.early_response import compare, early_features
from equipoise.longitudinal.hmm import (
    fit_hmm,
    forward_loglik,
    label_states,
    split_sequences,
    visit_sequences,
    viterbi,
)
from equipoise.longitudinal.tensor import build_tensor, lagrange_coefficients, lagrange_eval
from equipoise.longitudinal.timeseries import (
    adf_table,
    arm_mean_series,
    fit_arima_by_aic,
    moving_average,
)
from equipoise.plotting import ARM_COLORS, INK, INK_2, NEUTRAL, SERIES

STAGE = "longitudinal"
STATE_COLORS = {"improving": SERIES[2], "active": SERIES[1], "stable": SERIES[0]}


def _mean_band(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.sum(~np.isnan(x), axis=0)
    m = np.nanmean(x, axis=0)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, m - 1.96 * se, m + 1.96 * se


def _trajectory_plot(tensor, cfg: Config) -> tuple[plt.Figure, pd.DataFrame]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rows = []
    for ax, name, label in [(axes[0], "va_letters", "Vision change (letters)"),
                            (axes[1], "cst_um", "CST change (um, machine units)")]:
        ch = tensor.change_from_baseline(name)
        for arm in cfg.arms:
            m, lo, hi = _mean_band(ch[tensor.treatment == arm])
            ax.plot(tensor.weeks, m, color=ARM_COLORS[arm], label=arm)
            ax.fill_between(tensor.weeks, lo, hi, color=ARM_COLORS[arm], alpha=0.15, lw=0)
            for w in (12, 24, 52, 104):
                rows.append({"measure": name, "arm": arm, "week": w,
                             "mean_change": m[w], "ci_low": lo[w], "ci_high": hi[w]})
        ax.axhline(0, color=NEUTRAL, lw=1)
        ax.set_xlabel("Weeks since randomization")
        ax.set_ylabel(label)
    axes[0].legend(title="Arm")
    fig.suptitle("Mean trajectories per arm with 95% bands (linear interpolation, all patients)",
                 fontweight="bold")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def _interpolation_plot(visits: pd.DataFrame, pid: int, cfg: Config
                        ) -> tuple[plt.Figure, pd.DataFrame]:
    g = visits[(visits.pt_id == pid) & visits.va_letters.notna()].sort_values("days_from_rand")
    w = g.days_from_rand.to_numpy(float) / 7
    y = g.va_letters.to_numpy(float)
    grid = np.linspace(0, w.max(), 400)
    lc = cfg.longitudinal["lagrange"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(grid, np.interp(grid, w, y), color=SERIES[0], lw=1.5, label="linear")
    axes[0].plot(grid, CubicSpline(w, y, bc_type="natural")(grid), color=SERIES[1], lw=1.5,
                 label="cubic spline")
    axes[0].scatter(w, y, color=INK, s=18, zorder=3, label="visits")
    axes[0].set_title("Linear vs cubic-spline interpolation")
    rows = []
    for n, color in ((int(lc["short_points"]), SERIES[2]), (int(lc["long_points"]), SERIES[3])):
        xs, ys = w[:n], y[:n]
        t = np.linspace(xs[0], xs[-1], 400)
        axes[1].plot(t, lagrange_eval(xs, ys, t), color=color, lw=1.6,
                     label=f"Lagrange, {n} points (degree {n - 1})")
        ours = lagrange_coefficients(xs - xs.mean(), ys)
        ref = scipy_lagrange(xs - xs.mean(), ys).coeffs
        rows.append({"points": n, "degree": n - 1, "max_abs_coef_diff_vs_scipy":
                     float(np.max(np.abs(ours - ref[-len(ours):]))),
                     "fitted_range_letters": float(np.ptp(lagrange_eval(xs, ys, t))),
                     "observed_range_letters": float(np.ptp(ys))})
    n_long = int(lc["long_points"])
    axes[1].scatter(w[:n_long], y[:n_long], color=INK, s=18, zorder=3, label="visits")
    lo, hi = y[:n_long].min(), y[:n_long].max()
    axes[1].set_ylim(lo - 0.8 * (hi - lo) - 5, hi + 0.8 * (hi - lo) + 5)
    axes[1].set_title("Lagrange polynomials: high degree oscillates (Runge)")
    for ax in axes:
        ax.set_xlabel("Weeks since randomization")
        ax.set_ylabel("Visual acuity (letters)")
        ax.legend(fontsize=8)
    fig.suptitle("Interpolating one training patient's visits", fontweight="bold")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def _rates_plot(tensor, cfg: Config) -> tuple[plt.Figure, pd.DataFrame]:
    ch = tensor.change_from_baseline("va_letters")
    h = float(cfg.longitudinal["grid_step_weeks"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows = []
    for arm in cfg.arms:
        m = np.nanmean(ch[tensor.treatment == arm], axis=0)
        rate = central_difference(m, h)
        axes[0].plot(tensor.weeks, rate, color=ARM_COLORS[arm], label=arm)
        for w in (2, 6, 12, 24, 52):
            rows.append({"arm": arm, "week": w, **{s: float(fn(m, h)[w])
                                                  for s, fn in SCHEMES.items()}})
    axes[0].axhline(0, color=NEUTRAL, lw=1)
    axes[0].set_xlim(0, 60)
    axes[0].set_ylabel("Rate of vision change (letters / week)")
    axes[0].set_title("Central-difference rate of mean vision change")
    axes[0].legend(title="Arm")
    m = np.nanmean(ch[tensor.treatment == cfg.arms[0]], axis=0)
    for i, (s, fn) in enumerate(SCHEMES.items()):
        axes[1].plot(tensor.weeks, fn(m, h), color=SERIES[i], lw=1.4, label=s)
    axes[1].set_xlim(0, 20)
    axes[1].set_title(f"Forward vs backward vs central ({cfg.arms[0]}, weeks 0-20)")
    axes[1].set_ylabel("Letters / week")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Weeks since randomization")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def _ts_plot(results: dict, cfg: Config) -> plt.Figure:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, (arm, r) in zip(np.atleast_1d(axes), results.items(), strict=True):
        ax.plot(r["weeks"], r["y"], "o", color=ARM_COLORS[arm], ms=4, label="observed mean")
        ax.plot(r["weeks"], r["ma"], color=INK_2, lw=1, label="moving average")
        ax.plot(r["fc_weeks"], r["fc"], color=ARM_COLORS[arm], lw=2, ls="--",
                label=f"ARIMA{r['order']} forecast")
        ax.fill_between(r["fc_weeks"], r["fc_lo"], r["fc_hi"], color=ARM_COLORS[arm], alpha=0.15)
        ax.axvline(int(cfg.longitudinal["ts_fit_until_week"]), color=NEUTRAL, ls=":", lw=1)
        span = np.nanmax(np.abs(r["y"]))
        ax.set_ylim(-0.5 * span, 2.0 * span)     # forecast bands are clipped; see table
        ax.set_title(arm)
        ax.set_xlabel("Week")
    np.atleast_1d(axes)[0].set_ylabel("Mean vision change (letters)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Arm-mean series: fit on weeks 0-52, forecast 56-104 (population-level "
                 "demonstration)", fontweight="bold")
    fig.tight_layout()
    return fig


def _hmm_plot(seq, paths: list[np.ndarray], labels: dict[int, str], example: int
              ) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bounds = np.cumsum([0, *seq.lengths])
    a, b = bounds[example], bounds[example + 1]
    weeks = seq.frame.week.to_numpy()[a:b]
    axes[0].plot(weeks, seq.X[a:b, 0], color=INK_2, lw=1, marker="o", ms=3)
    for s, name in labels.items():
        m = paths[example] == s
        axes[0].scatter(weeks[m], seq.X[a:b, 0][m], color=STATE_COLORS[name], s=40, zorder=3,
                        label=name)
    axes[0].axhline(0, color=NEUTRAL, lw=1)
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Vision change since previous visit (letters)")
    axes[0].set_title("Viterbi-decoded states for one training patient")
    axes[0].legend()
    allp = np.concatenate(paths)
    wk = seq.frame.week.to_numpy()
    bins = np.arange(0, 108, 8)
    centres = (bins[:-1] + bins[1:]) / 2
    bottom = np.zeros(len(centres))
    for s, name in labels.items():
        share = [np.mean(allp[(wk >= lo) & (wk < hi)] == s) if np.any((wk >= lo) & (wk < hi))
                 else np.nan for lo, hi in zip(bins[:-1], bins[1:], strict=True)]
        axes[1].bar(centres, share, width=7, bottom=bottom, color=STATE_COLORS[name], label=name)
        bottom += np.nan_to_num(share)
    axes[1].set_xlabel("Week (8-week bins)")
    axes[1].set_ylabel("Share of visits")
    axes[1].set_title("Decoded state occupancy over time (training patients)")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


def _clusters_plot(tc) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (name, curve) in enumerate(tc.centres.items()):
        n = int(tc.summary.set_index("cluster").loc[name, "n"])
        ax.plot(tc.weeks, curve, color=SERIES[i], lw=2, label=f"{name} (n={n})")
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xlabel("Weeks since randomization")
    ax.set_ylabel("Mean vision change (letters)")
    ax.set_title("k-means clusters of vision trajectories (training patients)")
    ax.legend()
    fig.tight_layout()
    return fig


def _early_plot(preds: dict) -> plt.Figure:
    names = [k for k in preds if k != "y_test"]
    y = preds["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, name in zip(axes, names, strict=True):
        ax.scatter(y, preds[name], s=14, color=SERIES[0], alpha=0.6, linewidths=0)
        lim = [min(y.min(), preds[name].min()) - 2, max(y.max(), preds[name].max()) + 2]
        ax.plot(lim, lim, color=NEUTRAL, ls="--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Observed letters gained at week 52 (held-out)")
    axes[0].set_ylabel("Predicted")
    fig.suptitle("Early-response forecasting: does knowing weeks 4-12 help?", fontweight="bold")
    fig.tight_layout()
    return fig


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M7 and return headline metrics."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    analysis = load_analysis(cfg, synthetic=synthetic)
    visits = load_visits(cfg, synthetic=synthetic)
    split = get_split(cfg, analysis, synthetic)
    train_ids = np.array(split.train_ids)
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, visits_path(cfg, synthetic), synthetic):
        tracking.log_params({"longitudinal": cfg.longitudinal, "n_visits": len(visits)})

        # tensor + interpolation --------------------------------------------------------
        tensor = build_tensor(visits, cfg, "linear")
        spline = build_tensor(visits, cfg, "spline")
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "shape": str(tensor.values.shape), "axes": "patients x weeks x measures",
            "measures": "va_letters, cst_um, injected",
            "share_cells_observed_va": float(np.mean(~np.isnan(tensor.measure("va_letters")))),
            "mean_abs_diff_linear_vs_spline_va": float(np.nanmean(np.abs(
                tensor.measure("va_letters") - spline.measure("va_letters"))))}]), "tensor")
        fig, traj = _trajectory_plot(tensor, cfg)
        tracking.log_figure(cfg, STAGE, fig, "mean_trajectories")
        tracking.log_table(cfg, STAGE, traj, "mean_trajectories")
        counts = visits[visits.pt_id.isin(train_ids)].groupby("pt_id").size()
        example = int(counts[counts >= int(cfg.longitudinal["lagrange"]["long_points"])]
                      .index[0])
        fig, lag = _interpolation_plot(visits, example, cfg)
        tracking.log_figure(cfg, STAGE, fig, "interpolation")
        tracking.log_table(cfg, STAGE, lag, "lagrange_check")

        # finite differences ----------------------------------------------------------
        fig, rates = _rates_plot(tensor, cfg)
        tracking.log_figure(cfg, STAGE, fig, "rates_of_change")
        tracking.log_table(cfg, STAGE, rates, "rates_of_change")
        orders = error_orders()
        tracking.log_table(cfg, STAGE, orders, "finite_difference_orders")
        metrics["central_difference_order"] = float(
            orders[orders.scheme == "central"].observed_order.dropna().iloc[-1])

        # time series -----------------------------------------------------------------
        series = arm_mean_series(tensor, cfg)
        tracking.log_table(cfg, STAGE, series, "arm_mean_series")
        fit_until = int(cfg.longitudinal["ts_fit_until_week"])
        ys = {a: series[series.arm == a].sort_values("week") for a in cfg.arms}
        tracking.log_table(cfg, STAGE, adf_table(
            {a: s[s.week <= fit_until].mean_change.to_numpy() for a, s in ys.items()}), "adf")
        ts_results, fc_rows, aic_tables = {}, [], []
        for arm, s in ys.items():
            fit_part = s[s.week <= fit_until]
            future = s[s.week > fit_until]
            order, table, fit = fit_arima_by_aic(fit_part.mean_change.to_numpy(), cfg)
            aic_tables.append(table.assign(arm=arm))
            fc = fit.get_forecast(len(future))
            mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
            err = future.mean_change.to_numpy() - mean
            fc_rows.append({"arm": arm, "order": str(order), "aic": float(fit.aic),
                            "forecast_rmse": float(np.sqrt(np.mean(err**2))),
                            "naive_last_value_rmse": float(np.sqrt(np.mean(
                                (future.mean_change.to_numpy() - fit_part.mean_change.iloc[-1])
                                ** 2))), "n_fit_points": len(fit_part)})
            ts_results[arm] = {"weeks": s.week.to_numpy(), "y": s.mean_change.to_numpy(),
                               "ma": moving_average(s.mean_change.to_numpy(),
                                                    int(cfg.longitudinal["ma_window"])),
                               "fc_weeks": future.week.to_numpy(), "fc": mean,
                               "fc_lo": ci[:, 0], "fc_hi": ci[:, 1], "order": order}
        tracking.log_table(cfg, STAGE, pd.concat(aic_tables), "arima_aic_grid")
        tracking.log_table(cfg, STAGE, pd.DataFrame(fc_rows), "arima_forecast")
        tracking.log_figure(cfg, STAGE, _ts_plot(ts_results, cfg), "time_series")

        # HMM -------------------------------------------------------------------------
        seq = visit_sequences(visits, train_ids)
        hmm = fit_hmm(seq, cfg)
        labels = label_states(hmm)
        seqs = split_sequences(seq)
        ours = np.array([forward_loglik(hmm, x) for x in seqs])
        ref = np.array([hmm.score(x) for x in seqs])
        paths, agree = [], []
        for x in seqs:
            p, _ = viterbi(hmm, x)
            paths.append(p)
            agree.append(np.array_equal(p, hmm.predict(x)))
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "sequences": len(seqs), "observations": len(seq.X),
            # pass/fail only: the exact difference (~1e-13) is float noise that varies by run
            "forward_matches_hmmlearn_within_1e-8": bool(np.max(np.abs(ours - ref)) < 1e-8),
            "forward_matches_hmmlearn_within_1e-12": bool(np.max(np.abs(ours - ref)) < 1e-12),
            "viterbi_paths_identical_to_hmmlearn": float(np.mean(agree)),
            "converged": bool(hmm.monitor_.converged)}]), "hmm_checks")
        states = pd.DataFrame([{
            "state": labels[k], "mean_delta_va": hmm.means_[k, 0],
            "mean_delta_cst": hmm.means_[k, 1],
            "sd_delta_va": float(np.sqrt(hmm.covars_[k][0, 0])),
            "sd_delta_cst": float(np.sqrt(hmm.covars_[k][1, 1])),
            "start_prob": hmm.startprob_[k],
            "share_of_visits": float(np.mean(np.concatenate(paths) == k))}
            for k in range(hmm.n_components)])
        tracking.log_table(cfg, STAGE, states, "hmm_states")
        trans = pd.DataFrame(hmm.transmat_, columns=[f"to {labels[k]}" for k in range(len(labels))])
        trans.insert(0, "from", [labels[k] for k in range(len(labels))])
        tracking.log_table(cfg, STAGE, trans, "hmm_transitions")
        state_name = np.array([labels[k] for k in np.concatenate(paths)])
        tab = pd.crosstab(state_name, seq.frame.injected.to_numpy())
        chi2, p, _, _ = stats.chi2_contingency(tab)
        inj = pd.DataFrame({"state": tab.index, "visits": tab.sum(axis=1).to_numpy(),
                            "share_injected_at_visit": (tab[1] / tab.sum(axis=1)).to_numpy()
                            if 1 in tab.columns else 0.0})
        inj["chi_square_p"] = p
        tracking.log_table(cfg, STAGE, inj, "hmm_state_vs_injection")
        tracking.log_figure(cfg, STAGE, _hmm_plot(seq, paths, labels, 0), "hmm")
        metrics.update({"hmm_forward_matches": float(np.max(np.abs(ours - ref)) < 1e-8),
                        "hmm_viterbi_agreement": float(np.mean(agree)),
                        "hmm_state_injection_p": float(p)})

        # trajectory clusters -----------------------------------------------------------
        tc = cluster_trajectories(build_tensor(visits, cfg, "linear", train_ids), cfg)
        tracking.log_table(cfg, STAGE, tc.summary, "trajectory_clusters")
        tracking.log_table(cfg, STAGE, tc.by_arm, "trajectory_clusters_by_arm")
        tracking.log_figure(cfg, STAGE, _clusters_plot(tc), "trajectory_clusters")

        # early response ------------------------------------------------------------------
        early = early_features(visits, analysis, cfg)
        df = analysis.merge(early, on="pt_id", how="left")
        table, model, preds = compare(split.train(df), split.test(df), cfg)
        tracking.log_table(cfg, STAGE, table, "early_response")
        tracking.log_figure(cfg, STAGE, _early_plot(preds), "early_response")
        t = table.set_index("Model")
        base, withe = t.iloc[0], t.iloc[1]
        metrics.update({"early_cv_rmse_baseline": base["CV RMSE"],
                        "early_cv_rmse_with_early": withe["CV RMSE"],
                        "early_test_rmse_baseline": base["Test RMSE"],
                        "early_test_rmse_with_early": withe["Test RMSE"],
                        "early_test_r2_with_early": withe["Test R2"]})
        models_dir = cfg.paths.models_dir / ("synthetic" if synthetic else "")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / EARLY_MODEL_NAME, compress=3)

        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
