"""Single entry point for configuration.

`load_config()` reads `config/default.yaml` and returns a frozen `Config` whose paths are
absolute (resolved against the repository root). `load_prices()` reads `config/costs.yaml` and
returns real prices only when every price is filled in; otherwise it returns an explicit
override or the clearly labelled DEMO set, with `is_demo=True` so callers can warn the user.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT_ENV_VAR = "EQUIPOISE_ROOT"


def project_root() -> Path:
    """Locate the repository root (the folder that holds `config/default.yaml`).

    Order: the `EQUIPOISE_ROOT` environment variable, then the current directory and its
    parents, then the location of this source file (editable install).
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "config" / "default.yaml").is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find config/default.yaml; set {ROOT_ENV_VAR} to the repository root."
    )


@dataclass(frozen=True)
class Paths:
    """Absolute project paths."""

    root: Path
    raw_dir: Path
    analysis_table: Path
    visits_table: Path
    synthetic_dir: Path
    splits_file: Path
    cst_quality_table: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    derived_dir: Path
    metrics_dir: Path

    @property
    def synthetic_analysis_table(self) -> Path:
        return self.synthetic_dir / self.analysis_table.name

    @property
    def synthetic_visits_table(self) -> Path:
        return self.synthetic_dir / self.visits_table.name

    @property
    def synthetic_cst_quality_table(self) -> Path:
        return self.synthetic_dir / self.cst_quality_table.name

    @property
    def synthetic_truth_table(self) -> Path:
        return self.synthetic_dir / "protocolT_truth.csv"


@dataclass(frozen=True)
class Config:
    """Typed view of `config/default.yaml`."""

    seed: int
    paths: Paths
    test_fraction: float
    stratify_by: str
    n_folds: int
    arms: tuple[str, ...]
    arm_sizes: dict[str, int]
    n_patients: int
    n_missing_wk52: int
    reference_arm: str
    primary_outcome: str
    va_eligibility_letters: tuple[int, int]
    mcid_letters: float
    bootstrap_reps: int
    mlflow_tracking_uri: str
    mlflow_artifact_dir: Path
    mlflow_experiment_prefix: str
    synthetic: dict[str, Any] = field(default_factory=dict)
    visits: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    represent: dict[str, Any] = field(default_factory=dict)
    supervised: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    causal: dict[str, Any] = field(default_factory=dict)
    longitudinal: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def comparison_arms(self) -> tuple[str, ...]:
        """Arms compared against the reference arm, e.g. (Aflibercept, Ranibizumab)."""
        return tuple(a for a in self.arms if a != self.reference_arm)


def for_data_source(cfg: Config, synthetic: bool) -> Config:
    """Route outputs of synthetic runs to reports/synthetic/ and data/derived/synthetic/.

    Synthetic runs (tests, CI, demos) must never overwrite the real report assets.
    """
    if not synthetic:
        return cfg
    from dataclasses import replace

    base = cfg.paths.figures_dir.parent / "synthetic"
    paths = replace(cfg.paths, figures_dir=base / "figures", tables_dir=base / "tables",
                    metrics_dir=base / "metrics", derived_dir=cfg.paths.derived_dir / "synthetic")
    return replace(cfg, paths=paths)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: Path | None = None) -> Config:
    """Load `config/default.yaml` (or `path`) into a `Config`."""
    root = project_root()
    raw = _read_yaml(path or root / "config" / "default.yaml")
    p = {k: (root / v).resolve() for k, v in raw["paths"].items()}
    trial = raw["trial"]
    return Config(
        seed=int(raw["seed"]),
        paths=Paths(root=root, **p),
        test_fraction=float(raw["split"]["test_fraction"]),
        stratify_by=str(raw["split"]["stratify_by"]),
        n_folds=int(raw["cv"]["n_folds"]),
        arms=tuple(trial["arms"]),
        arm_sizes={k: int(v) for k, v in trial["arm_sizes"].items()},
        n_patients=int(trial["n_patients"]),
        n_missing_wk52=int(trial["n_missing_wk52"]),
        reference_arm=str(trial["reference_arm"]),
        primary_outcome=str(trial["primary_outcome"]),
        va_eligibility_letters=tuple(trial["va_eligibility_letters"]),
        mcid_letters=float(raw["effects"]["mcid_letters"]),
        bootstrap_reps=int(raw["bootstrap"]["n_reps"]),
        mlflow_tracking_uri=str(raw["mlflow"]["tracking_uri"]),
        mlflow_artifact_dir=(root / raw["mlflow"]["artifact_dir"]).resolve(),
        mlflow_experiment_prefix=str(raw["mlflow"]["experiment_prefix"]),
        synthetic=dict(raw.get("synthetic", {})),
        visits=dict(raw.get("visits", {})),
        preprocess=dict(raw.get("preprocess", {})),
        stats=dict(raw.get("stats", {})),
        represent=dict(raw.get("represent", {})),
        supervised=dict(raw.get("supervised", {})),
        search=dict(raw.get("search", {})),
        causal=dict(raw.get("causal", {})),
        longitudinal=dict(raw.get("longitudinal", {})),
        decision=dict(raw.get("decision", {})),
        explain=dict(raw.get("explain", {})),
        raw=raw,
    )


@dataclass(frozen=True)
class Prices:
    """Per-injection drug prices and laser-session price.

    `is_demo` is True when the DEMO relative units are in use; any output built from these
    prices must display `label` as a visible warning.
    """

    per_injection: dict[str, float]
    laser_session: float
    is_demo: bool
    label: str
    sources: tuple[str, ...] = ()
    willingness_to_pay: float | None = None   # per unit of value-of-vision utility


DEMO_WARNING = "DEMO prices in use (relative units, not real rupees). Fill config/costs.yaml."


def load_prices(
    override: dict[str, Any] | None = None, path: Path | None = None
) -> Prices:
    """Resolve prices: explicit `override` > fully filled `costs.yaml` > DEMO set.

    `override` uses the same keys as costs.yaml: `price_per_injection_inr` (dict per drug)
    and `laser_session_inr`. A partially filled costs.yaml is treated as unfilled, because
    mixing real rupees with relative demo units would be meaningless.
    """
    raw = _read_yaml(path or project_root() / "config" / "costs.yaml")
    if override is not None:
        per = override["price_per_injection_inr"]
        laser = override["laser_session_inr"]
        if any(v is None for v in per.values()) or laser is None:
            raise ValueError("A --prices override must give every drug price and the laser price.")
        wtp = override.get("willingness_to_pay")
        return Prices({k: float(v) for k, v in per.items()}, float(laser), False, "override",
                      willingness_to_pay=None if wtp is None else float(wtp))

    per = raw.get("price_per_injection_inr") or {}
    laser = raw.get("laser_session_inr")
    if per and all(v is not None for v in per.values()) and laser is not None:
        return Prices(
            {k: float(v) for k, v in per.items()},
            float(laser),
            False,
            f"{raw.get('currency', 'INR')} (config/costs.yaml)",
            tuple(raw.get("sources") or ()),
            None if raw.get("willingness_to_pay_inr_per_utility") is None
            else float(raw["willingness_to_pay_inr_per_utility"]),
        )

    demo = raw["demo"]
    warnings.warn(DEMO_WARNING, UserWarning, stacklevel=2)
    return Prices(
        {k: float(v) for k, v in demo["price_per_injection"].items()},
        float(demo["laser_session"]),
        True,
        str(demo["label"]),
        willingness_to_pay=float(demo["willingness_to_pay_per_utility"]),
    )
