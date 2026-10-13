# Deployment

Equipoise ships as a FastAPI service plus a Streamlit app. Both are thin layers over
`equipoise.decision.recommend()`, so local runs, Docker and a hosted demo all give the same
answers for the same bundle. **Nothing here has been deployed by the tooling: deploying is a
team decision.**

> Research prototype for a college AIML project. Not for clinical use.

## 0. Before you publish anything

1. **Never upload raw or derived DRCR data.** `data/` (except `data/synthetic/`) is
   git-ignored and excluded from every Docker build context by `.dockerignore`. Keep it that
   way, and never copy `data/` into a Space or a cloud bucket.
2. **Check the DRCR terms before publishing a model trained on DRCR data.** The bundle in
   `models/equipoise_bundle.joblib` was fitted on Protocol T patients. Whether a model trained
   on it may be redistributed or served publicly is a question for the DRCR Retina Network's
   data-use terms (https://public.jaeb.org/drcrnet), not for us. Until the team has confirmed
   this in writing, deploy the **synthetic-data model** publicly (section 3) and keep the real
   model on your own machine.
3. **Keep the attribution.** Every app page carries the DRCR attribution in its footer and
   the "research prototype, not for clinical use" banner. Do not remove either.
4. **Prices.** `config/costs.yaml` ships with `null` prices, so the app runs on the labelled
   DEMO set and shows a warning. Fill it in only with cited Indian prices.

## 1. Run locally (no Docker)

```bash
pip install -e ".[dev,app]"
uvicorn api.main:app --reload            # http://localhost:8000/docs
streamlit run app/streamlit_app.py       # http://localhost:8501 (loads the bundle itself)
```

To make the app call the API instead of loading the bundle, set `API_URL` first.

| Variable | Effect |
|---|---|
| `API_URL` | the app calls this API (e.g. `http://localhost:8000`) instead of loading the bundle |
| `EQUIPOISE_SYNTHETIC=1` | serve the synthetic demo bundle in `models/synthetic/` |
| `EQUIPOISE_BUNDLE` | path to a specific bundle file |
| `EQUIPOISE_REQUEST_LOG` | request-log path; `off` disables logging (default `logs/requests.jsonl`) |

Requests are logged with **baseline features only**: no names, IDs, IP addresses or
outcomes. `python -m equipoise monitor` compares them with the training features (drift
report).

## 2. Docker Compose (API + app)

Prerequisites: Docker Desktop running, and the models built locally
(`python -m equipoise causal`, `longitudinal`, `explain`; or `dvc repro`).

```bash
docker compose -f docker/docker-compose.yml up --build
```

- API at http://localhost:8000/docs, app at http://localhost:8501.
- `models/` is mounted **read-only** into the API container and is not baked into any image.
  `logs/` is mounted for the request log. `config/costs.yaml` is mounted so price edits apply
  without a rebuild.
- To serve the synthetic model, put `EQUIPOISE_SYNTHETIC=1` in `docker/.env`, or prefix the
  command with it.

The images install CPU-only PyTorch and the pinned `requirements*.txt`, so a build downloads
roughly 1–2 GB the first time.

## 3. Public demo on Hugging Face Spaces (Docker SDK), synthetic model

This is the recommended public deployment: one container in which the app loads the
**synthetic-data** bundle itself. It holds no DRCR data and no DRCR-trained weights.

1. Build the synthetic models locally:
   ```bash
   python -m equipoise causal --synthetic
   python -m equipoise longitudinal --synthetic
   python -m equipoise explain --synthetic
   ```
   This writes `models/synthetic/`: the bundle, the early-response model and the serving
   assets.
2. Test the image locally:
   ```bash
   docker build -f docker/Dockerfile.space -t equipoise-space .
   docker run -p 7860:7860 equipoise-space
   ```
   Open http://localhost:7860. The sidebar must say the app is serving the synthetic-data
   model.
3. Create a new Space (SDK: **Docker**) and clone its repo.
4. Copy into the Space repo: `src/`, `app/`, `config/`, `reports/`, `data/synthetic/`,
   `models/synthetic/`, `pyproject.toml`, `README.md`, `requirements.txt`,
   `requirements-app.txt`, and `docker/Dockerfile.space` saved as `Dockerfile` at the
   Space root.
5. The bundle is larger than 10 MB, so track it with Git LFS:
   `git lfs track "*.joblib"`. Then commit and push to the Space.
6. Add this front-matter at the top of the Space's `README.md` (Spaces read it):
   ```yaml
   ---
   title: Equipoise (synthetic demo)
   sdk: docker
   app_port: 7860
   ---
   ```
7. In the Space description, state that this is a synthetic-data demo and include the DRCR
   attribution.

Never push `models/equipoise_bundle.joblib` (the DRCR-trained model) or anything under
`data/` except `data/synthetic/` to a Space.

## 4. Streamlit Community Cloud (alternative)

Community Cloud runs `app/streamlit_app.py` directly from a GitHub repo, with no Docker and
no API. It suits the synthetic demo only:

1. In a public demo repo or branch, add the files listed in section 3, step 4. Leave out the
   Dockerfile, and track `models/synthetic/*.joblib` with Git LFS.
2. Add a top-level `requirements.txt` that installs the package with its app extra, i.e. the
   lines `-r requirements-app.txt` and `.`. If the build runs out of memory, use CPU-only
   torch.
3. In the app's settings, set the secret/env `EQUIPOISE_SYNTHETIC = "1"` and Python 3.11.
4. Deploy, and check that the sidebar shows the synthetic-model warning.

Community Cloud has tight memory limits (check the current limits before deploying). The
causal-forest bundle plus PyTorch may not fit, so Hugging Face Spaces, whose free CPU tier
has much more RAM, is the safer choice.

## 5. OCT screening tab (M10, optional)

The app shows a sixth page, "OCT screening", only when the `[cv]` extra is installed and
`models/oct/` exists (`python -m equipoise vision`). The Docker images install only the core
and `[app]` dependencies, so the tab is hidden there unless you add `requirements-cv.txt`
to the image and mount `models/oct/`. The separate Gradio demo runs with
`python app/oct_gradio.py`. These models are trained on the public Kermany dataset (CC BY
4.0 on Mendeley Data; check the Kaggle page's terms too), not on DRCR data. Ultralytics YOLO
is AGPL-3.0, which matters if you publish an app that includes it.

## 6. What each endpoint does

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | – | status, model name, synthetic flag, price label |
| `GET /metadata` | – | arms, MCID, feature schema, form defaults, attribution |
| `POST /predict` | `{"patient": {...}, "prices": {...}?, "explain": true}` | `Recommendation` + SHAP contributions + imputed features |
| `POST /breakeven` | `{"patient": {...}, "prices": {...}?, "drug": "Aflibercept"}` | break-even price/benefit (Newton-Raphson) + net-benefit curve |
| `POST /early-response` | `{"patient": {...}, "treatment": "...", "va_wk4"/"va_wk8"/"va_wk12": ...}` | week-52 forecast + trajectory |

Patient fields are generated from the feature schema. Only `X_va_letters` is required.
Unknown fields (`Y_`, `M_`, `treatment`, `pt_id`) are rejected with HTTP 422, so outcomes
cannot leak in through the API.
