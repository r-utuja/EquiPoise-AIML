"""Report-template tables generated from the code and the experiment log, never typed in.

- Feature description: every analysis-table column, its stored type, description and role
  straight from `data.schema`.
- Experiment table: one row per model run read back from MLflow (latest real-data run of
  each stage plus its child runs), with the parameters and metrics MLflow recorded.
- Tech stack: the third-party libraries actually imported by `src/`, `api/` and `app/`
  (found by parsing the code with `ast`), with the names imported from each.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd

from equipoise.config import Config
from equipoise.data.schema import ANALYSIS_COLUMNS, ColumnSpec
from equipoise.tracking import STAGES, experiment_name, tracking_uri

ROLE_TEXT = {"id": "identifier / randomized arm", "feature": "model input (baseline X_)",
             "outcome": "target only (Y_)", "meta": "bookkeeping (M_)"}


def _dtype(spec: ColumnSpec) -> str:
    if spec.categories:
        return "categorical (" + ", ".join(spec.categories[:3]) + \
            (", ..." if len(spec.categories) > 3 else "") + ")"
    if spec.binary:
        return "binary (0/1)"
    rng = f" [{spec.min:g}, {spec.max:g}]" if spec.min is not None and spec.max is not None \
        else ""
    return ("integer" if spec.integer else "continuous") + rng


def feature_description() -> pd.DataFrame:
    """Feature | Data Type | Description | Role | Missing allowed."""
    return pd.DataFrame([{"Feature": c.name, "Data Type": _dtype(c),
                          "Description": c.description, "Role": ROLE_TEXT[c.role],
                          "Missing allowed": "yes" if c.nullable else "no"}
                         for c in ANALYSIS_COLUMNS])


# ------------------------------------------------------------------ experiments (MLflow)
_MODEL_PARAMS = {   # config keys that describe each M5 model (shown in the Parameters column)
    "Random forest": "supervised.random_forest.", "Gradient boosting (Poisson": "supervised."
    "hist_gb_poisson.", "Gradient boosting": "supervised.gradient_boosting.",
    "Decision tree": "supervised.tree.", "kNN": "supervised.knn_neighbors",
    "Ridge": "supervised.ridge_alphas", "Lasso": "supervised.lasso_n_alphas",
    "IRLS": "supervised.irls.", "Logistic regression": "supervised.logistic_C",
    "Poisson GLM": "supervised.poisson_alpha",
}
_TASK_METRICS = {
    "regression": ["CV_RMSE", "Test_RMSE", "Test_MAE", "Test_R2"],
    "classification": ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1",
                       "Test_AUC"],
    "count": ["CV_Poisson_deviance", "Test_Poisson_deviance", "Test_MAE"],
    "search": ["best_objective", "n_features", "n_evals"],
}
_SELECT_ON = {"regression": ("CV_RMSE", min), "classification": ("CV_AUC", max),
              "count": ("CV_Poisson_deviance", min), "search": ("best_objective", min)}
_STAGE_METRICS = {   # headline metrics per stage run (keys as logged)
    "preprocess": ["diff_mice_aflibercept", "diff_complete-case_aflibercept", "nrmse_mice"],
    "stats": ["mean_gain_yr1_aflibercept", "mean_gain_yr1_bevacizumab", "anova_yr1_p",
              "p_interaction_va_below_69"],
    "represent": ["pca_components_for_target", "k_best", "silhouette_best"],
    "causal": ["pehe_chosen_worst_case", "policy_all_bev_ipw", "policy_model_ipw",
               "policy_published_ipw"],
    "longitudinal": ["early_test_rmse_baseline", "early_test_rmse_with_early",
                     "hmm_forward_matches"],
    "decision": ["share_equipoise", "newton_brentq_max_diff_benefit"],
    "explain": ["shap_rank_va_letters", "n_unreliable_subgroups"],
    "vision": ["resnet_test_accuracy", "resnet_test_macro_f1", "yolo_test_accuracy",
               "yolo_test_macro_f1"],
}


def _fmt_params(params: dict[str, str], prefix: str | None, limit: int = 4) -> str:
    keys = [k for k in params if prefix and k.startswith(prefix)][:limit]
    return "; ".join(f"{k.split('.', 1)[1]}={params[k]}" for k in keys) or "-"


def _fmt_metrics(metrics: dict[str, float], keys: list[str]) -> str:
    return "; ".join(f"{k.replace('_', ' ')}={metrics[k]:.3g}" for k in keys if k in metrics)


def experiments_table(cfg: Config) -> pd.DataFrame:
    """Exp-NN | Module | Model | Parameters | Metrics | Remarks, from the latest real runs."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri(cfg))
    client = MlflowClient()
    rows: list[dict[str, str]] = []
    for stage, (code, _) in STAGES.items():
        exp = client.get_experiment_by_name(experiment_name(cfg, stage))
        if exp is None or stage in ("monitor", "report-assets"):
            continue
        parents = [r for r in client.search_runs(
            [exp.experiment_id], "tags.data_source = 'real' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"], max_results=20)
            if "mlflow.parentRunId" not in r.data.tags]
        if not parents:
            continue
        parent = parents[0]
        kids = client.search_runs([exp.experiment_id],
                                  f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
                                  max_results=500)
        if not kids:
            rows.append({"Module": code.upper(), "Model": stage,
                         "Parameters": _fmt_params(parent.data.params, f"{stage}."),
                         "Metrics": _fmt_metrics(parent.data.metrics,
                                                 _STAGE_METRICS.get(stage, [])),
                         "Remarks": f"stage run {parent.info.run_id[:8]}"})
            continue
        groups: dict[str, list] = defaultdict(list)
        for k in kids:
            task = k.data.params.get("task") or ("search" if "algorithm" in k.data.params
                                                 else "other")
            groups[task].append(k)
        for task, runs in sorted(groups.items()):
            family = task.split()[0]
            sel_metric, better = _SELECT_ON.get(family, (None, None))
            scored = [r for r in runs if sel_metric in r.data.metrics]
            best = better(scored, key=lambda r: r.data.metrics[sel_metric]) if scored else None
            for r in sorted(runs, key=lambda r: r.info.run_name or ""):
                model = r.data.params.get("model") or r.data.params.get("algorithm", "?")
                if "seed" in r.data.params:
                    model += f" (seed {r.data.params['seed']})"
                prefix = next((v for k, v in _MODEL_PARAMS.items() if k in model), None)
                params = _fmt_params(parent.data.params, prefix) if family != "search" else \
                    _fmt_params(parent.data.params, "search.", 3)
                remark = "baseline" if model.startswith("Baseline") else ""
                if r is best:
                    remark = (remark + "; " if remark else "") + \
                        f"best {sel_metric.replace('_', ' ')} in {task}"
                rows.append({"Module": code.upper(), "Model": f"{task}: {model}",
                             "Parameters": params,
                             "Metrics": _fmt_metrics(r.data.metrics,
                                                     _TASK_METRICS.get(family, [])),
                             "Remarks": remark or "-"})
    out = pd.DataFrame(rows)
    out.insert(0, "Exp", [f"Exp-{i:02d}" for i in range(1, len(out) + 1)])
    return out


# ------------------------------------------------------------------ tech stack (ast)
_PURPOSE = {
    "numpy": "arrays, linear algebra (SVD, normal equations), 3-D trajectory tensor",
    "pandas": "tabular data, joins, report tables",
    "scipy": "statistical tests, interpolation/splines, sparse matrices, brentq",
    "sklearn": "preprocessing pipelines, models, CV, clustering, metrics",
    "statsmodels": "OLS/GLM inference, ARIMA, ADF test",
    "mlxtend": "Apriori and association rules",
    "econml": "meta-learners and causal forest (CATE)",
    "torch": "TARNet neural network (SGD / Adam / RMSprop)",
    "optuna": "TPE hyperparameter search",
    "hmmlearn": "Gaussian HMM reference implementation",
    "skfuzzy": "fuzzy inference system",
    "shap": "SHAP explanations of the effect model",
    "networkx": "causal DAG drawing",
    "matplotlib": "figures",
    "typer": "command-line interface",
    "pydantic": "validated API / recommendation schemas",
    "joblib": "model bundle persistence",
    "mlflow": "experiment tracking and model registry",
    "fastapi": "REST API",
    "streamlit": "clinician web app",
    "evidently": "data-drift monitoring",
    "requests": "app-to-API HTTP client",
    "yaml": "configuration files (PyYAML)",
    "autogluon": "AutoML base learner (optional extra)",
}
_DIST_NAME = {"sklearn": "scikit-learn", "skfuzzy": "scikit-fuzzy", "yaml": "PyYAML",
              "autogluon": "autogluon.tabular"}


def _chains(tree: ast.AST, alias: dict[str, str]) -> list[tuple[str, str]]:
    """(library, dotted attribute) for every outermost `alias.a.b` chain, e.g. np.linalg.svd."""
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in inner:
            parts, cur = [], node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id in alias:
                out.append((alias[cur.id], ".".join(reversed(parts))))
    return out


def tech_stack(roots: list[Path]) -> pd.DataFrame:
    """Library | Purpose | Key functions used | Files, for third-party imports in `roots`.

    Names imported with `from lib import f` are listed; for `import lib as x` the most used
    `x.something` attributes are listed instead.
    """
    import sys
    from collections import Counter

    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    names: dict[str, set[str]] = defaultdict(set)
    files: dict[str, set[str]] = defaultdict(set)
    calls: dict[str, Counter] = defaultdict(Counter)
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            alias: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        top = a.name.split(".")[0]
                        alias[a.asname or top] = top
                        files[top].add(path.name)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    sub = node.module.split(".", 1)[1] if "." in node.module else ""
                    for a in node.names:
                        names[top].add(f"{sub}.{a.name}" if sub else a.name)
                    files[top].add(path.name)
            for lib, chain in _chains(tree, alias):
                calls[lib][chain] += 1
    rows = []
    for lib in sorted(files):
        if lib in stdlib or lib in ("equipoise", "api", "app"):
            continue
        used = sorted(names[lib]) or [n for n, _ in calls[lib].most_common(8)]
        rows.append({"Library": _DIST_NAME.get(lib, lib), "Purpose": _PURPOSE.get(lib, ""),
                     "Key functions used": ", ".join(used[:8]) + (" ..." if len(used) > 8
                                                                  else ""),
                     "Files": len(files[lib])})
    return pd.DataFrame(rows)
