# MLOps: Git, DVC, MLflow, Docker, CI, Evidently

- **Git** versions code, configs, synthetic data and aggregate reports. DRCR data is never
  committed (`.gitignore`, DRCR terms).
- **DVC** versions the data and the pipeline. `dvc.yaml` lists every stage with its
  dependencies, parameters (read from `config/default.yaml`) and outputs; `dvc.lock`
  records their hashes. `dvc repro` reruns only the stages whose inputs changed.
- **MLflow** records one run per stage (parameters, metrics, figures, tables, tags for git
  commit and data hash) and a model registry where the bundle is `equipoise-cate`, with
  alias `champion` (see `docs/experiments.md`).
- **Docker** packages the API and app reproducibly. Models are mounted at run time; data is
  never baked in.
- **GitHub Actions CI** runs ruff and 160+ tests on synthetic data, and builds the Docker
  images (no push), on every push.
- **Evidently** compares the features of incoming API requests with the training features,
  one statistical test per feature. On labelled *simulated* traffic in which half the
  requests were shifted (older, higher HbA1c, worse vision), it flagged age, HbA1c and
  vision as drifted (`python -m equipoise monitor`). It also flagged diabetes type and OCT
  machine, which were not shifted: with 120 requests resampled from 132 test patients,
  drift tests raise some false alarms too.

**Why.** Every number in the report can be traced to a code version, a data hash and a
logged run, and anyone can rebuild it with `dvc repro`.
