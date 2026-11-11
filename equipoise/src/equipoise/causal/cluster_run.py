"""Stage `unsupervised`: compare the cluster-then-look-up model with the supervised causal model.

Fits `ClusterEffectModel` on training patients (baseline features only), then evaluates it once
on the held-out test patients next to results the earlier stages already produced: the Lasso and
arm-mean regression rows (M5), the chosen causal model's per-patient effects and intervals (M6,
read from data/derived) and its semi-synthetic benchmark. Nothing supervised is retrained.
Aggregate results go to reports/; per-patient estimates go to data/derived/ (never committed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from equipoise import tracking
from equipoise.causal.cluster_effect import ClusterEffectModel
from equipoise.causal.data import arm_codes, known_propensities, prepare
from equipoise.causal.policy import evaluate_policies
from equipoise.causal.semisynth import run_benchmark, summarize
from equipoise.config import Config
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.splits import get_split
from equipoise.preprocess.discretize import gain_item
from equipoise.supervised.evaluate import classification_metrics

STAGE = "unsupervised"
EQUIPOISE = "Equipoise"
MIN_CELL_N = 20
CLUSTER = ClusterEffectModel.name
PER_PATIENT = "m6_unsupervised_heldout.csv"


def verdicts(low: np.ndarray, effect: np.ndarray, arms: list[str], mcid: float) -> np.ndarray:
    """Per patient: the comparison drug with the largest effect among those whose lower
    interval end is >= MCID, else 'Equipoise' (the causal model's rule)."""
    clear = low >= mcid
    best = np.argmax(np.where(clear, effect, -np.inf), axis=1)
    return np.where(clear.any(axis=1), np.asarray(arms)[best], EQUIPOISE)


def _rmse_r2(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    sse = float(np.sum((y - pred) ** 2))
    return float(np.sqrt(sse / len(y))), 1 - sse / float(np.sum((y - y.mean()) ** 2))


def _cells(model: ClusterEffectModel, cfg: Config, level: float) -> pd.DataFrame:
    """Aggregate cluster x arm table: n, mean outcome, effect vs reference with interval."""
    B = model._bootstrap_effects()
    a = (1 - level) / 2
    lo, hi = np.quantile(B, a, axis=0), np.quantile(B, 1 - a, axis=0)
    arms = list(arm_codes(cfg))   # code order: reference first
    rows = []
    for c in range(model.k):
        for j, arm in enumerate(arms):
            row = {"cluster": c, "arm": arm, "n": int(model.n_[c, j]),
                   "mean_va_change_wk52": float(model.mean_[c, j])}
            if j > 0:
                eff = model.mean_[c, j] - model.mean_[c, 0]
                row.update({"effect_vs_reference": float(eff), "ci_low": float(lo[c, j - 1]),
                            "ci_high": float(hi[c, j - 1]),
                            "p_benefit_gt_mcid": float((B[:, c, j - 1] > cfg.mcid_letters).mean()),
                            "clearly_better": bool(lo[c, j - 1] >= cfg.mcid_letters)})
            row["note"] = f"n < {MIN_CELL_N}" if model.n_[c, j] < MIN_CELL_N else ""
            rows.append(row)
    return pd.DataFrame(rows)


def _m5_rows(cfg: Config) -> dict[str, tuple[float, float]]:
    path = cfg.paths.tables_dir / "m5_regression_comparison.csv"
    if not path.is_file():
        return {}
    t = pd.read_csv(path).set_index("Model")
    out = {}
    for label, key in [("Lasso (CV alpha)", "Lasso (M5, best by CV)"),
                       ("Baseline: arm mean", "Arm-mean baseline (M5)")]:
        if label in t.index:
            out[key] = (float(t.loc[label, "Test RMSE"]), float(t.loc[label, "Test R2"]))
    return out


def _chosen_name(cfg: Config) -> str:
    path = cfg.paths.tables_dir / "m6_chosen_model.csv"
    return str(pd.read_csv(path).chosen_learner.iloc[0]) if path.is_file() else "Causal model"


def _mu(cfg: Config, synthetic: bool, X: np.ndarray, model: ClusterEffectModel) -> np.ndarray:
    """Outcome predictions under every arm for AIPW: the saved M6 per-arm models if the bundle
    exists (same as M6's policy table), else this model's own cell means."""
    from equipoise.causal.bundle import load_bundle
    from equipoise.decision.recommend import default_bundle_path

    path = default_bundle_path(cfg, synthetic=synthetic)
    if path.is_file():
        return np.column_stack([m.predict(X) for m in load_bundle(path).outcome_models])
    return np.column_stack([model.predict_outcome(X, np.full(len(X), a))
                            for a in range(len(cfg.arms))])


def _method(model: str) -> str:
    """Label each row: supervised, unsupervised, a no-learning baseline, or a comparison."""
    if model == CLUSTER or model.startswith(("Cluster:", "Treat by cluster")):
        return "Unsupervised"
    if "vs" in model:
        return "Comparison"
    if model.startswith(("Arm-mean", "Constant", "All ", "Published")) or "baseline" in model:
        return "Baseline"
    return "Supervised"


def _metric_info(metric: str) -> tuple[str, str]:
    """(unit, which direction is better) for the value column."""
    if metric.startswith("RMSE") or metric.startswith("PEHE"):
        return "letters", "lower"
    if "bias" in metric:
        return "letters", "closer to 0"
    if metric.endswith("value"):
        return "letters gained", "higher"
    if metric.startswith("Share in equipoise") or "agreement" in metric:
        return "%", "n/a"
    return "score 0-1" if metric in CLASS_METRICS else "ratio", "higher"


CLASS_METRICS = ("Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC", "PR-AUC")


def run(cfg: Config, synthetic: bool) -> dict[str, float]:
    """Run the unsupervised comparison and return headline metrics."""
    df = load_analysis(cfg, synthetic=synthetic)
    split = get_split(cfg, df, synthetic)
    tr, te, _ = prepare(split.train(df), split.test(df), cfg)
    codes, probs = arm_codes(cfg), known_propensities(cfg)
    arms, comps = list(codes), list(cfg.comparison_arms)
    mcid, level = float(cfg.mcid_letters), float(cfg.stats["ci_level"])
    cf_name = _chosen_name(cfg)
    metrics: dict[str, float] = {}
    rows: list[dict] = []

    def add(section: str, metric: str, model: str, value: float, note: str = "") -> None:
        unit, better = _metric_info(metric)
        rows.append({"section": section, "metric": metric, "model": model,
                     "method": _method(model), "value": value, "unit": unit,
                     "better": better, "note": note})

    with tracking.start_run(cfg, STAGE, analysis_path(cfg, synthetic), synthetic):
        model = ClusterEffectModel(cfg).fit(tr.X, tr.T, tr.Y)
        tracking.log_params({"k": model.k, "n_train": len(tr), "n_test": len(te)})
        cells = _cells(model, cfg, level)
        tracking.log_table(cfg, STAGE, cells, "unsupervised_cells")
        metrics["cells_n_below_20"] = float((model.n_ < MIN_CELL_N).sum())

        # a) predicting week-52 change ------------------------------------------------
        rmse, r2 = _rmse_r2(te.Y, model.predict_outcome(te.X, te.T))
        pred = {CLUSTER: (rmse, r2), **_m5_rows(cfg)}
        for name, (r, q) in pred.items():
            add("Predicting week-52 letters (test)", "RMSE (letters)", name, r)
            add("Predicting week-52 letters (test)", "R2", name, q)
        metrics.update({"cluster_test_rmse": rmse, "cluster_test_r2": r2})

        # a2) classifying a gain of >= 15 letters (same task, threshold and test rows as M5)
        gain = gain_item(df, cfg).reindex(df.index)
        gain_by_id = pd.Series(gain.to_numpy(), index=df.pt_id.to_numpy())
        g_tr = gain_by_id.reindex(tr.frame.pt_id).to_numpy()
        g_te = gain_by_id.reindex(te.frame.pt_id).to_numpy()
        ok_tr, ok_te = ~np.isnan(g_tr), ~np.isnan(g_te)
        rate = np.array([[g_tr[ok_tr & (model.labels_ == c) & (tr.T == a)].mean()
                          for a in range(len(arms))] for c in range(model.k)])
        proba = rate[model.cluster_of(te.X), te.T][ok_te]
        thr = float(cfg.supervised["classification_threshold"])
        cls = {CLUSTER: classification_metrics(g_te[ok_te].astype(int), proba, thr)}
        m5c = cfg.paths.tables_dir / "m5_classification_gain15_full.csv"
        if m5c.is_file():
            t5 = pd.read_csv(m5c).set_index("Model")
            for src, label in [("Logistic regression", "Logistic regression (M5)"),
                               ("Baseline: arm event rate", "Arm-rate baseline (M5)")]:
                if src in t5.index:
                    cls[label] = {k: float(t5.loc[src, f"Test {k}"]) for k in CLASS_METRICS}
        for name, vals in cls.items():
            for k in CLASS_METRICS:
                add("Classifying gain >= 15 letters (test)", k, name, float(vals[k]))
        metrics.update({"cluster_gain15_f1": float(cls[CLUSTER]["F1"]),
                        "cluster_gain15_auc": float(cls[CLUSTER]["AUC"])})

        # per-patient estimates ---------------------------------------------------------
        lo, hi = model.effect_interval(te.X, alpha=1 - level)
        eff, p_ben = model.predict_effect(te.X), model.p_benefit(te.X, mcid)
        per = pd.DataFrame({"pt_id": te.frame.pt_id.to_numpy(), "cluster": model.cluster_of(te.X)})
        for k, arm in enumerate(comps):
            per[f"effect_{arm}"] = eff[:, k]
            per[f"low_{arm}"], per[f"high_{arm}"] = lo[:, k], hi[:, k]
            per[f"p_benefit_gt_mcid_{arm}"] = p_ben[:, k]
        v_cl = verdicts(lo, eff, comps, mcid)
        per["verdict_cluster"] = v_cl
        metrics["cluster_share_equipoise"] = float(np.mean(v_cl == EQUIPOISE))
        add("Verdicts (test)", "Share in equipoise (%)", CLUSTER,
            100 * metrics["cluster_share_equipoise"])

        # b) verdicts against the causal model ------------------------------------------
        cf_path = cfg.paths.derived_dir / "m6_heldout_effects.csv"
        cf_eff = cf_lo = None
        if cf_path.is_file():
            cf = pd.read_csv(cf_path).set_index("pt_id").reindex(per.pt_id)
            if not cf.isna().any().any():
                cf_eff = cf[[f"effect_{a}" for a in comps]].to_numpy()
                cf_lo = cf[[f"low_{a}" for a in comps]].to_numpy()
                for k, arm in enumerate(comps):
                    per[f"cf_effect_{arm}"], per[f"cf_low_{arm}"] = cf_eff[:, k], cf_lo[:, k]
                    per[f"cf_high_{arm}"] = cf[f"high_{arm}"].to_numpy()
        v_cf = None
        if cf_eff is not None:
            v_cf = verdicts(cf_lo, cf_eff, comps, mcid)
            per["verdict_causal"] = v_cf
            rho = spearmanr(cf_eff[:, 0], eff[:, 0]).statistic
            metrics.update({"causal_share_equipoise": float(np.mean(v_cf == EQUIPOISE)),
                            "verdict_agreement": float(np.mean(v_cf == v_cl)),
                            "spearman_afl_benefit": float(rho)})
            add("Verdicts (test)", "Share in equipoise (%)", cf_name,
                100 * metrics["causal_share_equipoise"])
            add("Verdicts (test)", "Verdict agreement (%)", "Cluster vs causal",
                100 * metrics["verdict_agreement"])
            add("Verdicts (test)", f"Spearman, {comps[0]} benefit", "Cluster vs causal",
                float(rho), "cluster estimates take only k distinct values (ties)")
        else:
            add("Verdicts (test)", "Comparison with causal model", "-", np.nan,
                "m6_heldout_effects.csv not found: run `python -m equipoise causal` first")
        cfg.paths.derived_dir.mkdir(parents=True, exist_ok=True)
        per.to_csv(cfg.paths.derived_dir / PER_PATIENT, index=False, lineterminator="\n")

        # c) policy value (known propensities) -------------------------------------------
        mu = _mu(cfg, synthetic, te.X, model)
        n = len(te)
        policies = {f"All {a}": np.full(n, codes[a]) for a in arms}
        below = (te.frame.X_va_below_69 == 1).to_numpy()
        policies["Published rule (<69 letters -> aflibercept)"] = np.where(
            below, codes["Aflibercept"], codes[cfg.reference_arm])

        def assigned(v: np.ndarray) -> np.ndarray:   # equipoise -> the reference (cheapest) arm
            return np.array([codes.get(x, codes[cfg.reference_arm]) for x in v])

        policies["Treat by cluster verdict"] = assigned(v_cl)
        policies["Cluster: most letters"] = np.argmax(np.column_stack([np.zeros(n), eff]), axis=1)
        if v_cf is not None:
            policies[f"Treat by causal verdict ({cf_name})"] = assigned(v_cf)
            policies[f"Causal: most letters ({cf_name})"] = np.argmax(
                np.column_stack([np.zeros(n), cf_eff]), axis=1)
        pol = evaluate_policies(policies, te.T, te.Y, mu, probs, cfg, arms)
        tracking.log_table(cfg, STAGE, pol, "unsupervised_policy_values")
        for _, r in pol.iterrows():
            add("Policy value (letters gained, test)", "IPW value", r.policy, r.ipw_value)
            add("Policy value (letters gained, test)", "AIPW value", r.policy, r.aipw_value)
        pv = pol.set_index("policy")
        metrics.update({"policy_cluster_verdict_ipw": float(pv.loc["Treat by cluster verdict",
                                                                    "ipw_value"]),
                        "policy_cluster_verdict_aipw": float(pv.loc["Treat by cluster verdict",
                                                                     "aipw_value"])})

        # d) semi-synthetic PEHE / bias (same benchmark as M6) ---------------------------
        summ = summarize(run_benchmark(tr, {CLUSTER: ClusterEffectModel}, cfg)[0])
        ref_path = cfg.paths.tables_dir / "m6_semisynthetic_pehe.csv"
        if ref_path.is_file():
            ref = pd.read_csv(ref_path)
            ref = ref[ref.learner.isin([cf_name, "Constant effect (difference in means)"])]
            summ = pd.concat([summ, ref[summ.columns.intersection(ref.columns)]])
        tracking.log_table(cfg, STAGE, summ[["scenario", "learner", "pehe_mean", "ate_bias_mean"]],
                           "unsupervised_semisynthetic")
        for _, r in summ.iterrows():
            add(f"Semi-synthetic ({r.scenario})", "PEHE (letters)", r.learner, r.pehe_mean)
            add(f"Semi-synthetic ({r.scenario})", "Average-effect bias (letters)", r.learner,
                r.ate_bias_mean)
        own = summ[summ.learner == CLUSTER].set_index("scenario")
        metrics["pehe_cluster_worst_case"] = float(own.pehe_mean.max())

        tracking.log_table(cfg, STAGE, pd.DataFrame(rows), "unsupervised_comparison")
        tracking.write_metrics(cfg, STAGE, metrics)
    return metrics
