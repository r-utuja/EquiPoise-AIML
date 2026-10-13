"""Phase 9: drift monitoring on logged requests, registry promotion, CLI wiring."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.decision.recommend import FEATURES

pytest.importorskip("evidently")


def _log(path: Path, rows: pd.DataFrame) -> None:
    from equipoise.serving.service import append_request_log

    for _, r in rows.iterrows():
        feats = {f: (None if pd.isna(r[f]) else (r[f].item() if hasattr(r[f], "item") else r[f]))
                 for f in FEATURES}
        append_request_log(path, "test", feats)


@pytest.fixture()
def split_frames(cfg: Config, synthetic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = get_split(cfg, synthetic, True)
    return s.train(synthetic), s.test(synthetic)


def test_read_request_log_roundtrip(tmp_path: Path, split_frames) -> None:
    from equipoise.monitor import read_request_log

    _, test = split_frames
    log = tmp_path / "r.jsonl"
    _log(log, test.head(10))
    back = read_request_log(log)
    assert list(back.columns) == FEATURES and len(back) == 10
    np.testing.assert_allclose(back.X_va_letters, test.head(10).X_va_letters)


def test_drift_detects_planted_shift_only(cfg: Config, split_frames) -> None:
    from equipoise.monitor import drift_report

    train, test = split_frames
    same, _ = drift_report(train[FEATURES], test[FEATURES], cfg)
    shifted = test[FEATURES].copy()
    shifted["X_va_letters"] = (shifted.X_va_letters - 15).clip(24, 78)
    shifted["X_age"] = shifted.X_age + 15
    table, summary = drift_report(train[FEATURES], shifted, cfg)
    flagged = set(table[table.drifted].feature)
    assert {"X_va_letters", "X_age"} <= flagged
    assert not {"X_va_letters", "X_age"} & set(same[same.drifted].feature)
    assert 0 <= summary["share_drifted_features"] <= 1


def test_monitor_stage_writes_report(cfg: Config, split_frames, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from equipoise.monitor import run

    _, test = split_frames
    log = tmp_path / "requests.jsonl"
    _log(log, pd.concat([test] * 2).head(cfg.raw["monitor"]["min_requests"] + 5))
    monkeypatch.setenv(cfg.raw["serving"]["request_log_env_var"], str(log))
    paths = replace(cfg.paths, figures_dir=tmp_path / "rep" / "figures")
    small = replace(cfg, paths=paths,
                    mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                    mlflow_artifact_dir=tmp_path / "mlruns")
    out = run(small, synthetic=True)
    assert Path(out["report"]).is_file()
    assert (tmp_path / "rep" / "monitoring" / "drift_by_feature.csv").is_file()
    monkeypatch.setenv(cfg.raw["serving"]["request_log_env_var"], str(tmp_path / "none.jsonl"))
    with pytest.raises(FileNotFoundError, match="simulate_requests"):
        run(small, synthetic=True)


def test_promote_sets_champion_alias(tmp_path: Path) -> None:
    import mlflow
    from mlflow import MlflowClient

    from equipoise.causal.bundle import REGISTERED_MODEL
    from equipoise.registry import CHAMPION_ALIAS, promote_bundle

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    client.create_registered_model(REGISTERED_MODEL)
    for _ in range(2):
        client.create_model_version(REGISTERED_MODEL, source=str(tmp_path))
    assert promote_bundle(uri) == "2"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version) == "2"
    assert promote_bundle(uri, version="1") == "1"
    with pytest.raises(LookupError):
        promote_bundle(uri, version="9")


def test_cli_registers_monitor_and_promote() -> None:
    from typer.testing import CliRunner

    from equipoise.cli import _runner, app

    assert _runner("monitor") is not None
    out = CliRunner().invoke(app, ["--help"]).output
    assert "monitor" in out and "promote" in out
