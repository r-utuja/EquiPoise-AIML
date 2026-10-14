"""M6 stage: counterfactual engine.

Learners are fitted on training patients (complete cases for week-52 vision). The final
learner is chosen by semi-synthetic PEHE (lowest mean over the three scenarios, excluding the
constant-effect yardstick); held-out policy values are reported alongside. The held-out test
set is used only for final reporting: policy values, calibration by predicted-effect
tertile, and the rediscovery checks.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from equipoise import tracking
from equipoise.causal import automl
from equipoise.causal.bundle import ModelBundle, register_bundle, save_bundle
from equipoise.causal.data import arm_codes, known_propensities, prepare
from equipoise.causal.learners import CausalForest, EconMLMeta, base_learner, learner_zoo
from equipoise.causal.policy import evaluate_policies
from equipoise.causal.semisynth import SCENARIOS, run_benchmark, simulate, summarize
from equipoise.causal.tarnet import OPTIMIZERS, TARNetLearner, train_tarnet
from equipoise.causal.tuning import compare_tuners
from equipoise.config import Config, load_prices
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.schema import ANALYSIS_COLUMNS
from equipoise.data.splits import get_split
from equipoise.plotting import ARM_COLORS, INK_2, NEUTRAL, SERIES
from equipoise.preprocess.columns import feature_groups
from equipoise.search.objective import column_owner
from equipoise.stats.replication import welch_ci
from equipoise.supervised.models import classification_models, count_models

STAGE = "causal"
CONSTANT = "Constant effect (difference in means)"


# ----------------------------------------------------------------------------- figures
def _tarnet_curves(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, (name, r) in enumerate(results.items()):
        ep = np.arange(1, len(r.train_loss) + 1)
        axes[0].plot(ep, r.train_loss, color=SERIES[i], label=name)
        axes[1].plot(ep, r.val_loss, color=SERIES[i],
                     label=f"{name} (best epoch {r.best_epoch + 1})")
        axes[1].scatter([r.best_epoch + 1], [r.best_val_loss], color=SERIES[i], s=30, zorder=3)
    axes[0].set_title("Training loss (factual outcomes)")
    axes[1].set_title("Validation loss (early stopping)")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Mean squared error (standardized letters)")
    axes[1].legend(title="Optimizer")
    fig.suptitle("TARNet trained with SGD (momentum), Adam and RMSprop - same seed, data and "
                 "initial weights", fontweight="bold")
    fig.tight_layout()
    return fig


def _pehe_plot(summary: pd.DataFrame, chosen: str) -> plt.Figure:
    learners = (summary.groupby("learner").pehe_mean.mean().sort_values(ascending=False)
                .index.tolist())
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(learners) + 1.5))
    labels = {"a_null": "(a) null", "b_published_like": "(b) published-like",
              "c_nonlinear": "(c) nonlinear"}
    for i, sc in enumerate(SCENARIOS):
        s = summary[summary.scenario == sc].set_index("learner").loc[learners]
        y = np.arange(len(learners)) + (i - 1) * 0.22
        ax.errorbar(s.pehe_mean, y, xerr=s.pehe_sd, fmt="o", color=SERIES[i], ms=5, capsize=2,
                    label=labels[sc])
    ax.set_yticks(range(len(learners)),
                  [f"{n}  <- chosen" if n == chosen else n for n in learners], fontsize=8)
    ax.set_xlabel("PEHE: RMSE of estimated vs true individual effect (letters; mean ± SD)")
    ax.set_title("Semi-synthetic benchmark (lower is better)")
    ax.legend(title="Scenario", loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _policy_plot(pol: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(pol) + 1.5))
    y = np.arange(len(pol))[::-1]
    ax.errorbar(pol.ipw_value, y + 0.12, xerr=[pol.ipw_value - pol.ipw_ci_low,
                                               pol.ipw_ci_high - pol.ipw_value],
                fmt="o", color=SERIES[0], capsize=3, label="IPW (known propensities)")
    ax.errorbar(pol.aipw_value, y - 0.12, xerr=[pol.aipw_value - pol.aipw_ci_low,
                                                pol.aipw_ci_high - pol.aipw_value],
                fmt="s", color=SERIES[1], capsize=3, label="Doubly robust (AIPW)")
    ax.set_yticks(y, pol.policy, fontsize=9)
    ax.set_xlabel("Expected letters gained at week 52 if patients followed the policy (95% CI)")
    ax.set_title("Policy value on held-out patients")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def _tertile_plot(cal: pd.DataFrame, comparison: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cal))
    ax.errorbar(x - 0.08, cal.observed_difference, yerr=[cal.observed_difference - cal.ci_low,
                                                         cal.ci_high - cal.observed_difference],
                fmt="o", color=SERIES[0], capsize=4, label="observed difference (Welch 95% CI)")
    ax.scatter(x + 0.08, cal.mean_predicted_effect, color=SERIES[1], marker="D", s=40,
               label="mean predicted effect", zorder=3)
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xticks(x, [f"{t}\n(n={n})" for t, n in zip(cal.tertile, cal.n, strict=True)])
    ax.set_ylabel(f"{comparison} (letters)")
    ax.set_title("Calibration by predicted-effect tertile (held-out patients)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _tuning_plot(hist: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (m, h) in enumerate(hist.groupby("method", sort=False)):
        ax.plot(h.trial, h.best_so_far, color=SERIES[i], marker="o", ms=3, label=m)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best CV RMSE so far (letters)")
    ax.set_title("Hyperparameter tuning of the per-arm outcome model (same budget)")
    ax.legend()
    fig.tight_layout()
    return fig


def _effect_distribution(effects: dict[str, np.ndarray], frame: pd.DataFrame, cfg: Config
                         ) -> plt.Figure:
    below = (frame.X_va_below_69 == 1).to_numpy()
    fig, axes = plt.subplots(1, len(cfg.comparison_arms), figsize=(11, 3.8), sharey=True)
    for ax, (k, arm) in zip(np.atleast_1d(axes), enumerate(cfg.comparison_arms), strict=True):
        e = effects["chosen"][:, k]
        bins = np.linspace(e.min() - 1, e.max() + 1, 25)
        ax.hist(e[below], bins=bins, color=ARM_COLORS[arm], alpha=0.85,
                label="baseline < 69 letters (20/50 or worse)")
        ax.hist(e[~below], bins=bins, histtype="step", lw=2, color=INK_2,
                label="baseline >= 69 letters")
        ax.axvline(cfg.mcid_letters, color=NEUTRAL, ls="--", lw=1)
        ax.set_xlabel(f"Predicted {arm} - {cfg.reference_arm} (letters)")
        ax.set_title(arm)
    np.atleast_1d(axes)[0].set_ylabel("Held-out patients")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("Predicted individual effects of the chosen learner (dashed = MCID)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- helpers
def _fit_outcome_models(tr, cfg: Config, params: dict) -> list:
    """Per-arm GB outcome models (tuned) on preprocessed X."""
    return [base_learner("gb", cfg, params).fit(tr.X[tr.T == a], tr.Y[tr.T == a])
            for a in range(len(cfg.arms))]


def _mu(models: list, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models])


def _chosen_factory(name: str, cfg: Config, gb_params: dict):
    """Factory for the chosen learner, with tuned GB parameters where it uses GB."""
    if name.startswith(("S-", "T-", "X-", "DR-")):
        kind = name.split("-")[0]
        base = "gb" if "(GB)" in name else "ridge"
        params = gb_params if base == "gb" else None
        return lambda c: EconMLMeta(c, kind, base, params)
    if name.startswith("TARNet"):
        return lambda c: TARNetLearner(c, name.split("(")[1].rstrip(")"))
    return learner_zoo(cfg)[name]


def select_learner(summary: pd.DataFrame, train_effect_sd: dict[str, float], cfg: Config
                   ) -> pd.DataFrame:
    """Apply the configured selection rule; one row per learner with the reason it was kept
    or excluded. Uses only semi-synthetic results and training-patient predictions."""
    rule = cfg.causal["selection"]
    piv = summary.pivot(index="learner", columns="scenario", values="pehe_mean")
    bias = summary.pivot(index="learner", columns="scenario", values="ate_bias_mean")
    out = pd.DataFrame({"learner": piv.index})
    for sc in SCENARIOS:
        out[f"pehe_{sc}"] = piv[sc].to_numpy()
    out["worst_case_pehe"] = piv.max(axis=1).to_numpy()
    out["mean_pehe"] = piv.mean(axis=1).to_numpy()
    out["max_abs_ate_bias"] = bias.abs().max(axis=1).to_numpy()
    out["train_effect_sd"] = out.learner.map(train_effect_sd)
    reasons = []
    for _, r in out.iterrows():
        if r.train_effect_sd <= float(rule["constant_effect_sd"]):
            reasons.append("excluded: effect identical for every patient by construction")
        elif r.max_abs_ate_bias > float(rule["max_abs_ate_bias"]):
            reasons.append(f"excluded: average-effect bias > {rule['max_abs_ate_bias']} letters")
        else:
            reasons.append("candidate")
    out["status"] = reasons
    cand = out[out.status == "candidate"]
    best = cand.loc[cand.worst_case_pehe.idxmin(), "learner"] if len(cand) else \
        out.loc[out.worst_case_pehe.idxmin(), "learner"]
    out["chosen"] = out.learner == best
    out.loc[out.chosen, "status"] = "chosen: lowest worst-case PEHE among candidates"
    return out.sort_values(["chosen", "worst_case_pehe"], ascending=[False, True])


def _calibration(eff: np.ndarray, te, cfg: Config, k: int) -> pd.DataFrame:
    if np.ptp(eff[:, k]) < 1e-9:   # constant predictions: tertiles are undefined
        return pd.DataFrame([{"tertile": "all (no heterogeneity predicted)", "n": len(eff),
                              "mean_predicted_effect": float(eff[0, k])}])
    level = float(cfg.stats["ci_level"])
    q = np.quantile(eff[:, k], [1 / 3, 2 / 3])
    groups = np.digitize(eff[:, k], q)
    rows = []
    for g, label in enumerate(["low", "middle", "high"]):
        m = groups == g
        a = pd.Series(te.Y[m & (te.T == k + 1)])
        b = pd.Series(te.Y[m & (te.T == 0)])
        d, lo, hi, p = welch_ci(a, b, level) if min(len(a), len(b)) >= 3 else (np.nan,) * 4
        rows.append({"tertile": label, "n": int(m.sum()), "n_arm": len(a), "n_reference": len(b),
                     "mean_predicted_effect": float(eff[m, k].mean()),
                     "observed_difference": d, "ci_low": lo, "ci_high": hi, "p": p})
    return pd.DataFrame(rows)


def _bootstrap(factory, tr, cfg: Config, reps: int) -> list:
    """Refit the learner on bootstrap resamples drawn within each arm (keeps arm sizes)."""
    rng = np.random.default_rng(cfg.seed)
    models = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(np.flatnonzero(tr.T == a), (tr.T == a).sum())
                              for a in np.unique(tr.T)])
        models.append(factory(cfg).fit(tr.X[idx], tr.T[idx], tr.Y[idx]))
    return models


def _importance_rank(effect: np.ndarray, X: np.ndarray, names: list[str], cfg: Config
                     ) -> pd.DataFrame:
    """Which raw features drive the predicted aflibercept effect? Surrogate random forest on
    the predicted effects; importances summed per raw feature."""
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=cfg.seed,
                               n_jobs=1).fit(X, effect)
    owners = column_owner([f"x__{n}" for n in names], list(feature_groups(cfg).all))
    imp = pd.Series(rf.feature_importances_, index=owners).groupby(level=0).sum()
    out = (imp.sort_values(ascending=False).rename("importance").rename_axis("feature")
           .reset_index())
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ----------------------------------------------------------------------------- stage
def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run M6 and return headline metrics."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    c = cfg.causal
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    tr, te, pre = prepare(split.train(df), split.test(df), cfg)
    codes = arm_codes(cfg)
    arms = list(codes)                                   # reference first
    probs = known_propensities(cfg)
    comparisons = [f"{a} - {cfg.reference_arm}" for a in cfg.comparison_arms]
    metrics: dict[str, float] = {}

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        tracking.log_params({"causal": c, "n_train": len(tr), "n_test": len(te),
                             "propensities": dict(zip(arms, probs.round(4), strict=True))})

        # 1. TARNet optimizers ----------------------------------------------------------
        curves = {o: train_tarnet(tr.X, tr.T, tr.Y, cfg, o, len(arms))[1] for o in OPTIMIZERS}
        tracking.log_figure(cfg, STAGE, _tarnet_curves(curves), "tarnet_optimizers")
        tracking.log_table(cfg, STAGE, pd.DataFrame([
            {"optimizer": o, "epochs_run": len(r.train_loss), "best_epoch": r.best_epoch + 1,
             "best_val_loss": r.best_val_loss, "final_train_loss": r.train_loss[-1]}
            for o, r in curves.items()]), "tarnet_optimizers")
        best_opt = min(curves, key=lambda o: curves[o].best_val_loss)
        metrics["tarnet_best_val_loss"] = curves[best_opt].best_val_loss

        # 2. all learners on real training data ------------------------------------------
        zoo = learner_zoo(cfg, include_tarnet=False)
        zoo[f"TARNet ({best_opt})"] = lambda cc, o=best_opt: TARNetLearner(cc, o)
        fitted = {name: f(cfg).fit(tr.X, tr.T, tr.Y) for name, f in zoo.items()}
        test_effects = {name: m.effect(te.X) for name, m in fitted.items()}
        below = (te.frame.X_va_below_69 == 1).to_numpy()
        rows = []
        for name, e in test_effects.items():
            for k, comp in enumerate(comparisons):
                rows.append({"learner": name, "comparison": comp,
                             "mean_effect": float(e[:, k].mean()),
                             "sd_effect": float(e[:, k].std()),
                             "mean_effect_va_below_69": float(e[below, k].mean()),
                             "mean_effect_va_69_or_better": float(e[~below, k].mean())})
        tracking.log_table(cfg, STAGE, pd.DataFrame(rows), "learner_effects_heldout")

        # 3-4. semi-synthetic benchmark and selection -----------------------------------
        bench, sim = run_benchmark(tr, zoo, cfg)
        summary = summarize(bench)
        summary = summary.rename(columns={"false_heterogeneity_rate": "heterogeneity_found_rate"})
        tracking.log_table(cfg, STAGE, summary, "semisynthetic_pehe")
        train_sd = {name: float(m.effect(tr.X).std(axis=0).max()) for name, m in fitted.items()}
        selection = select_learner(summary, train_sd, cfg)
        chosen = str(selection.loc[selection.chosen, "learner"].iloc[0])
        tracking.log_table(cfg, STAGE, selection, "learner_selection")
        tracking.log_figure(cfg, STAGE, _pehe_plot(summary, chosen), "semisynthetic_pehe")
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "f0": "Ridge on bevacizumab patients", "noise_sd_letters": sim.noise_sd,
            "replications": int(c["semi_synthetic"]["replications"]),
            "eval_fraction": float(c["semi_synthetic"]["eval_fraction"])}]), "semisynthetic_setup")
        s_by = summary.set_index(["scenario", "learner"])
        metrics.update({
            "pehe_chosen_worst_case": float(
                selection.loc[selection.chosen, "worst_case_pehe"].iloc[0]),
            "pehe_constant_b": float(s_by.loc[("b_published_like", CONSTANT), "pehe_mean"]),
            "pehe_chosen_b": float(s_by.loc[("b_published_like", chosen), "pehe_mean"]),
            "false_heterogeneity_chosen_a": float(
                s_by.loc[("a_null", chosen), "heterogeneity_found_rate"])})

        # 5. tuning of the per-arm GB outcome model --------------------------------------
        hist, tune_summary, gb_params = compare_tuners(tr, cfg, "gb")
        tracking.log_table(cfg, STAGE, hist, "tuning_history")
        tracking.log_table(cfg, STAGE, tune_summary, "tuning_summary")
        tracking.log_figure(cfg, STAGE, _tuning_plot(hist), "tuning")

        # 6. final learner ---------------------------------------------------------------
        factory = _chosen_factory(chosen, cfg, gb_params)
        final = factory(cfg).fit(tr.X, tr.T, tr.Y)
        eff_te = final.effect(te.X)
        outcome_models = _fit_outcome_models(tr, cfg, gb_params)
        mu_te = _mu(outcome_models, te.X)

        # 7. policies --------------------------------------------------------------------
        prices = load_prices()
        raw_train, raw_test = split.train(df), te.frame
        inj_model = count_models(cfg)["Poisson GLM"].fit(raw_train, raw_train["Y_n_inj_yr1"])
        laser_model = classification_models(cfg)["Logistic regression"].fit(
            raw_train, raw_train["Y_laser_yr1"].astype(int))
        inj = np.column_stack([inj_model.predict(raw_test.assign(treatment=a)) for a in arms])
        p_laser = np.column_stack([laser_model.predict_proba(raw_test.assign(treatment=a))[:, 1]
                                   for a in arms])
        cost = inj * np.array([prices.per_injection[a] for a in arms]) \
            + p_laser * prices.laser_session
        lam = float(c["policy"]["lambda_letters_per_cost_unit"])
        gain = np.column_stack([np.zeros(len(te)), eff_te])
        n = len(te)
        policies = {f"All {a}": np.full(n, codes[a]) for a in arms}
        policies["Published rule (<69 letters -> aflibercept)"] = np.where(
            below, codes["Aflibercept"], codes[cfg.reference_arm])
        policies[f"Model: most letters ({chosen})"] = np.argmax(gain, axis=1)
        policies[f"Model: letters - lambda x cost ({prices.label})"] = np.argmax(
            gain - lam * (cost - cost[:, [0]]), axis=1)
        pol = evaluate_policies(policies, te.T, te.Y, mu_te, probs, cfg, arms)
        pol["prices"] = prices.label
        tracking.log_table(cfg, STAGE, pol, "policy_values")
        tracking.log_figure(cfg, STAGE, _policy_plot(pol), "policy_values")
        per_learner = {}
        for name, e in test_effects.items():
            assign = np.argmax(np.column_stack([np.zeros(n), e]), axis=1)
            per_learner[f"Model: most letters ({name})"] = assign
        pol_l = evaluate_policies(per_learner, te.T, te.Y, mu_te, probs, cfg, arms)
        tracking.log_table(cfg, STAGE, pol_l, "policy_values_by_learner")
        pv = pol.set_index("policy")
        metrics.update({"policy_all_bev_ipw": pv.loc[f"All {cfg.reference_arm}", "ipw_value"],
                        "policy_all_afl_ipw": pv.loc["All Aflibercept", "ipw_value"],
                        "policy_model_ipw": pv.iloc[4]["ipw_value"],
                        "policy_model_aipw": pv.iloc[4]["aipw_value"],
                        "policy_published_ipw": pv.iloc[3]["ipw_value"],
                        "prices_demo": float(prices.is_demo)})

        # 8. calibration and rediscovery ------------------------------------------------
        for k, comp in enumerate(comparisons):
            cal = _calibration(eff_te, te, cfg, k)
            slug = cfg.comparison_arms[k].lower()
            tracking.log_table(cfg, STAGE, cal, f"calibration_{slug}")
            if len(cal) == 3:
                tracking.log_figure(cfg, STAGE, _tertile_plot(cal, comp), f"calibration_{slug}")
        eff_tr = final.effect(tr.X)
        if np.ptp(eff_tr[:, 0]) > 1e-9:
            # rounding removes ~1e-14 float noise from the forest's parallel tree building,
            # which would otherwise flip tie-breaks in the surrogate between runs
            imp = _importance_rank(np.round(eff_tr[:, 0], 6), tr.X, tr.feature_names, cfg)
        else:   # importance of a constant is undefined
            imp = pd.DataFrame([{"feature": "none (no heterogeneity predicted)",
                                 "importance": np.nan, "rank": np.nan}])
        tracking.log_table(cfg, STAGE, imp, "effect_importance_aflibercept")
        tr_below = (tr.frame.X_va_below_69 == 1).to_numpy()
        redis = pd.DataFrame([{
            "check": "predicted afl-bev larger when baseline < 69 letters (held-out)",
            "below_69": float(eff_te[below, 0].mean()),
            "69_or_better": float(eff_te[~below, 0].mean()),
            "passes": bool(eff_te[below, 0].mean() > eff_te[~below, 0].mean())}, {
            "check": "same on training patients",
            "below_69": float(eff_tr[tr_below, 0].mean()),
            "69_or_better": float(eff_tr[~tr_below, 0].mean()),
            "passes": bool(eff_tr[tr_below, 0].mean() > eff_tr[~tr_below, 0].mean())}, {
            "check": "baseline vision (va_letters or va_below_69) is the top driver of the "
                     "predicted aflibercept effect",
            "below_69": np.nan, "69_or_better": np.nan,
            "passes": bool(imp.feature.iloc[0] in ("X_va_letters", "X_va_below_69"))}])
        redis["top_feature"] = imp.feature.iloc[0]
        tracking.log_table(cfg, STAGE, redis, "rediscovery")
        tracking.log_figure(cfg, STAGE, _effect_distribution({"chosen": eff_te}, te.frame, cfg),
                            "effect_distribution")
        metrics["rediscovery_va_passes"] = float(redis.passes.iloc[0])
        metrics["top_driver_is_vision"] = float(redis.passes.iloc[2])

        # 9. uncertainty -----------------------------------------------------------------
        boot = _bootstrap(factory, tr, cfg, int(c["bootstrap_reps"]))
        B = np.stack([m.effect(te.X) for m in boot])
        lo, hi = np.quantile(B, 0.025, axis=0), np.quantile(B, 0.975, axis=0)
        p_mcid = (B > cfg.mcid_letters).mean(axis=0)
        unc_rows = []
        for k, comp in enumerate(comparisons):
            excl0 = (lo[:, k] > 0) | (hi[:, k] < 0)
            unc_rows.append({
                "comparison": comp, "bootstrap_reps": len(boot),
                "median_interval_width": float(np.median(hi[:, k] - lo[:, k])),
                "share_interval_excludes_0": float(np.mean(excl0)),
                "share_p_benefit_gt_mcid_above_0.8": float(np.mean(p_mcid[:, k] > 0.8)),
                "share_equipoise_ci_includes_below_mcid": float(
                    np.mean(lo[:, k] < cfg.mcid_letters)),
                "mean_p_benefit_gt_mcid": float(p_mcid[:, k].mean())})
        cf = fitted[CausalForest.name]
        cflo, cfhi = cf.effect_interval(te.X)
        for k, row in enumerate(unc_rows):
            row["causal_forest_median_interval_width"] = float(np.median(cfhi[:, k] - cflo[:, k]))
        tracking.log_table(cfg, STAGE, pd.DataFrame(unc_rows), "uncertainty_summary")
        cfg.paths.derived_dir.mkdir(parents=True, exist_ok=True)
        per_pt = pd.DataFrame({"pt_id": te.frame.pt_id})
        for k, arm in enumerate(cfg.comparison_arms):
            per_pt[f"effect_{arm}"] = eff_te[:, k]
            per_pt[f"low_{arm}"], per_pt[f"high_{arm}"] = lo[:, k], hi[:, k]
            per_pt[f"p_benefit_gt_mcid_{arm}"] = p_mcid[:, k]
            per_pt[f"cf_low_{arm}"], per_pt[f"cf_high_{arm}"] = cflo[:, k], cfhi[:, k]
        per_pt.to_csv(cfg.paths.derived_dir / "m6_heldout_effects.csv", index=False,
                      lineterminator="\n")
        metrics["share_equipoise_afl"] = unc_rows[0]["share_equipoise_ci_includes_below_mcid"]

        # 10. does M4's selected subset change things? -----------------------------------
        sel_path = cfg.paths.tables_dir / "m4_selected_features.csv"
        if sel_path.is_file():
            sel = pd.read_csv(sel_path)
            feats = ["X_" + f for f in sel.loc[sel.selected_in_best_subset, "feature"]]
            tr_s, te_s, _ = prepare(split.train(df), split.test(df), cfg, features=feats)
            m_s = factory(cfg).fit(tr_s.X, tr_s.T, tr_s.Y)
            e_s = m_s.effect(te_s.X)
            assign_s = np.argmax(np.column_stack([np.zeros(n), e_s]), axis=1)
            pol_s = evaluate_policies({"Model (M4 subset)": assign_s}, te.T, te.Y, mu_te, probs,
                                      cfg, arms)
            tracking.log_table(cfg, STAGE, pd.DataFrame([{
                "features": ", ".join(f.replace("X_", "") for f in feats),
                "corr_with_full_effect_afl": float(np.corrcoef(e_s[:, 0], eff_te[:, 0])[0, 1]),
                "mean_effect_afl_full": float(eff_te[:, 0].mean()),
                "mean_effect_afl_subset": float(e_s[:, 0].mean()),
                "policy_ipw_full": float(pv.iloc[4]["ipw_value"]),
                "policy_ipw_subset": float(pol_s.ipw_value.iloc[0]),
                "same_assignment_share": float(np.mean(assign_s == policies[
                    f"Model: most letters ({chosen})"]))}]), "m4_subset_effect")

        # 11. AutoGluon T-learner ------------------------------------------------------
        if automl.available() and c["automl"].get("enabled", True):
            ag = automl.AutoGluonTLearner(cfg, cfg.paths.models_dir / "autogluon" /
                                          ("synthetic" if synthetic else "real"))
            ag.fit(tr.X, tr.T, tr.Y)
            lb = ag.leaderboard()
            lb = lb[[c for c in lb.columns if "time" not in c]]   # timings vary run to run
            tracking.log_table(cfg, STAGE, lb, "autogluon_leaderboard")
            rng = np.random.default_rng(cfg.seed + 7)
            T_s, Y_s = simulate(sim, "b_published_like", rng, probs)
            idx = rng.permutation(len(Y_s))
            n_ev = int(round(float(c["semi_synthetic"]["eval_fraction"]) * len(Y_s)))
            ev, fit = idx[:n_ev], idx[n_ev:]
            truth = sim.taus["b_published_like"][ev]
            comp_rows = []
            ag_sim = automl.AutoGluonTLearner(cfg, cfg.paths.models_dir / "autogluon" / "semisynth")
            for name, model in [("T-learner (AutoGluon)", ag_sim),
                                ("T-learner (GB, tuned)", EconMLMeta(cfg, "T", "gb", gb_params)),
                                ("T-learner (GB, default)", EconMLMeta(cfg, "T", "gb")),
                                ("T-learner (Ridge)", EconMLMeta(cfg, "T", "ridge"))]:
                est = model.fit(tr.X[fit], T_s[fit], Y_s[fit]).effect(tr.X[ev])
                comp_rows.append({"learner": name, "scenario": "b_published_like (1 replication)",
                                  "pehe": float(np.sqrt(np.mean((est - truth) ** 2)))})
            ag_assign = np.argmax(np.column_stack([np.zeros(n), ag.effect(te.X)]), axis=1)
            ag_pol = evaluate_policies({"Model: most letters (T-learner AutoGluon)": ag_assign},
                                       te.T, te.Y, mu_te, probs, cfg, arms)
            out = pd.DataFrame(comp_rows)
            out["heldout_policy_ipw_autogluon"] = float(ag_pol.ipw_value.iloc[0])
            tracking.log_table(cfg, STAGE, out, "autogluon_comparison")
            metrics["pehe_b_autogluon_1rep"] = float(out.pehe.iloc[0])
        else:
            tracking.log_table(cfg, STAGE, pd.DataFrame([{
                "status": "skipped: install the [automl] extra (pip install -e .[automl])"}]),
                "autogluon_comparison")

        # 12. bundle ---------------------------------------------------------------------
        kept = boot[:int(c["bundle_bootstrap_models"])]
        rule = c["selection"]
        selection_text = (
            f"exclude effects constant by construction (train SD <= {rule['constant_effect_sd']}); "
            f"exclude |average-effect bias| > {rule['max_abs_ate_bias']} letters in any "
            f"scenario; then lowest worst-case semi-synthetic PEHE")
        schema = [{"name": s.name, "dtype": s.dtype, "min": s.min, "max": s.max,
                   "categories": list(s.categories) if s.categories else None,
                   "nullable": s.nullable, "description": s.description}
                  for s in ANALYSIS_COLUMNS if s.role == "feature"]
        bundle = ModelBundle(
            arms=arms, reference_arm=cfg.reference_arm, mcid_letters=cfg.mcid_letters,
            preprocessor=pre, effect_model=final, effect_model_name=chosen,
            bootstrap_models=kept, outcome_models=outcome_models, injection_model=inj_model,
            laser_model=laser_model, feature_schema=schema,
            training_data_hash=tracking.file_hash(analysis_path(cfg, synthetic), 64),
            config_snapshot=cfg.raw, synthetic=synthetic,
            extras={"tuned_gb_params": gb_params, "propensities": probs.tolist()})
        models_dir = cfg.paths.models_dir / ("synthetic" if synthetic else "")
        path = save_bundle(bundle, models_dir)
        version = register_bundle(path, chosen) if not synthetic else None
        if version:
            import mlflow

            mlflow.set_tag("registered_model_version", version)
        tracking.log_table(cfg, STAGE, pd.DataFrame([{
            "chosen_learner": chosen, "selection_rule": selection_text,
            "bundle": str(path.name), "bootstrap_models_reported": len(boot),
            "bootstrap_models_in_bundle": len(kept),
            "registered_in_mlflow": bool(version),
            "tuned_gb_params": str(gb_params)}]), "chosen_model")
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
