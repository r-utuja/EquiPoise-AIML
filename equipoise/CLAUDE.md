# CLAUDE.md: working rules for AI coding assistants

Equipoise is a student AIML mini-project: a cost-aware, explainable DME treatment-choice
assistant learned from the DRCR **Protocol T** randomized trial. `README.md` is the
specification. **Research prototype, not for clinical use.**

Ignore the `self learning/` folder outside this repo; it is not part of the project.

## Process

- Work **one phase at a time**. At the end of a phase: run `pytest` and the phase's commands,
  fix failures, commit (`feat(mN): ...`), summarize what was built, which commands to run,
  which outputs to look at and what is uncertain, then **stop and wait for "continue"**.
- **Ask, don't guess, about raw data.** Inspect the actual DRCR files under
  `data/raw/Data Tables - Text files/` and `scripts/build_protocolT_table.py` for column names;
  check the CRFs / protocol PDF if present; otherwise ask.
- `scripts/build_protocolT_table.py` (M0) is done: **do not rewrite it**; reuse its
  study-eye (`pick_eye`, `RandEye`) and join logic.
- **Never** push to GitHub, delete files outside the repo, or commit anything under `data/`
  except `data/synthetic/`.

## Scientific integrity (non-negotiable)

- **Only `X_` columns are model inputs.** `Y_` = targets only; `M_` = bookkeeping;
  `pt_id`, `study_eye` = identifiers. All feature matrices go through
  `equipoise.features.baseline_feature_matrix` / `assert_no_leakage` (tested in
  `tests/test_leakage.py`). `treatment` is allowed only where a learner explicitly takes it
  as the treatment variable (`allow_treatment=True`).
- Post-randomization raw tables are **never features**: `TAdvEvent`, `TMedication`,
  `TMedUpFU`, `TFUExamQuestions`, `TOcularExamFU`, `TVisitComp`, `TNonStudyEyeInj`,
  `TStudyProcedure`, `TPostRandPtFinalStatus`.
  - Single exception: M7's early-response model may use vision up to week 12. It lives in
    its own clearly named module.
- **Fit every preprocessing step on training folds only** (imputers, scalers, encoders,
  harmonization parameters) via sklearn `Pipeline` / `ColumnTransformer`.
- **Splits:** one fixed held-out test set (20%, stratified by `treatment`, seed from config,
  IDs in `data/splits.json`), touched only for final reporting. CV / cross-fitting inside the
  80%. **Never tune on the test set.**
- **Trial-level inference vs prediction.** M1's missing-data comparison and all of M2
  (Table 1, replication, hypothesis tests) estimate trial quantities and use all 660
  randomized patients, as the published analysis did. Anything that learns parameters for a
  predictive model is fitted on training folds only.
- **CST quality.** Primary analyses keep CST as recorded (as M0 / published). The sensitivity
  analysis (`preprocess.cst.quality_policy: missing`) drops reading-centre "Unacceptable"
  grades and machine switches (`data/protocolT_cst_quality.csv`, M_ columns only).
- **Outputs.** `reports/` holds only aggregate results (committed). Patient-level derived
  files go to `data/derived/` (never committed). `--synthetic` runs write to
  `reports/synthetic/` so they never overwrite real report assets.
- **M6 learner selection** (`causal.selection` in config): exclude models whose effects
  cannot vary by patient, exclude learners with > 1 letter average-effect bias in any
  semi-synthetic scenario, then pick the lowest worst-case PEHE. Never select on test-set
  outcome fit. Propensities are the known randomization probabilities (`KnownPropensity`).
- **Weak per-patient signal (M6 finding).** With ~370 fitting patients, flexible CATE learners
  are noise-dominated and the chosen causal forest calls every held-out patient "equipoise" at
  MCID 5. Decision layer and app must say per-patient differences are weakly estimated, always
  show the published < 69-letter rule alongside the model, and report MCID 3 as sensitivity.
- **Never fabricate or tidy results.** No heterogeneity or no gain over baseline is a
  legitimate finding; report it plainly. Never write numbers into docs/README that the code
  did not produce.
- **Seeds everywhere** (numpy, random, torch, sklearn, optuna): call
  `equipoise.reproducibility.seed_everything(cfg.seed)` and pass `random_state` / `seed`.

## Domain facts

| Item | Value |
|---|---|
| Arms | aflibercept 224, bevacizumab 218, ranibizumab 218 (total 660) |
| Primary outcome | `Y_va_change_wk52` (ETDRS letters) |
| Missing week-52 vision | 44 patients |
| Reference arm | **bevacizumab** (cheapest): report aflibercept − bevacizumab, ranibizumab − bevacizumab |
| Equipoise threshold | MCID = 5 letters (configurable, `effects.mcid_letters`) |
| Propensities | Known from randomization: use empirical arm proportions (≈0.34/0.33/0.33); **never** estimate with a model |
| Replication targets | README §9.1 table (tested in phase 2) |

## Prices

**Never invent prices.** `config/costs.yaml` holds `null` prices (`# TEAM: fill with current
Indian prices and cite the source`). `equipoise.config.load_prices()` falls back to an explicit
`--prices` override, else the labelled **DEMO** set (relative units 1/5/10). Any output or app
page built on demo prices must show a visible warning.

## Code quality

- Package code in `src/equipoise/`; CLI `python -m equipoise <stage> [--synthetic]`; thin
  notebooks.
- Config only via `config/default.yaml` + `config/costs.yaml` through `equipoise/config.py`.
  No magic numbers in code.
- Cross-platform (team may use Windows): `pathlib`, no shell-only tricks, no `make`.
- Type hints, docstrings, `ruff check .` clean. Each module docstring explains its method in
  2–4 plain sentences (students must defend every method in a viva).
- Every stage writes figures to `reports/figures/<module>_*.png`, tables to
  `reports/tables/<module>_*.csv` **and** `.md`, and an MLflow run
  (`equipoise.tracking.start_run`, experiment `equipoise-mN-<stage>`).
- Lean core deps; heavy optional pieces in extras `[automl]`, `[cv]`, `[app]`.

## Serving and MLOps

- API and app are thin layers over `equipoise.serving.Service` → `decision.recommend()`;
  put logic in the package, never in `api/main.py` or `app/streamlit_app.py`.
- `serving.schemas.PatientFeatures` is generated from the schema's `X_` columns and forbids
  unknown fields; never widen it to `Y_`/`M_`/`treatment`/identifiers.
- Request logs (`logs/requests.jsonl`) hold baseline features only. `serving_assets.joblib`
  (M9) holds k-means centroids and medians only, never patient rows.
- Docker images never contain `data/` (except synthetic) or the DRCR-trained bundle; models
  are mounted. Public demos use the synthetic bundle (`EQUIPOISE_SYNTHETIC=1`). Never deploy.
- `python -m equipoise promote` sets the MLflow alias `champion`; `monitor` writes the
  Evidently report to `reports/monitoring/` (git-ignored).

## M10 OCT screening (optional)

- Kermany retinal OCT images (Kaggle `paultimothymooney/kermany2018`): **not linked to
  Protocol T patients**; a screening front-end demo, never an input to the treatment model.
  Say so on every output.
- Subset fetched by `scripts/download_oct_subset.py` into `data/external/kermany_oct/` (never committed).
  The Kaggle token is the user's: pass it via `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`,
  never print or commit it.
- Validation split is grouped by patient (file names carry patient IDs); the official test
  split is the held-out test set. `python -m equipoise vision` is not part of `all`.
- Ultralytics YOLO is AGPL-3.0; its usage analytics are switched off in code.

## Environment

- Python **3.11** in `.venv` (`py -V:3.11 -m venv .venv` on Windows). `pip install -e ".[dev]"`.
- pandas **2.2+ and 3.x** are both supported (the `[automl]` extra pins pandas < 3). Never
  `astype("str")` a column with missing values (pandas 2 turns NaN into "nan"); use
  `schema.numeric_casts` / `schema.dtype_name`.
- Avoid backslash escapes inside bash heredocs that run Python edit scripts (they get
  mangled); use the editor tool for such edits.
- Tests run on `data/synthetic/` (regenerate with `python scripts/make_synthetic_data.py`).
- Real-data tests skip when `data/protocolT_analysis.csv` is absent.
