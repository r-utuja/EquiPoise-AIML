# Equipoise clinician app (Streamlit). Build from the repository root:
#   docker build -f docker/Dockerfile.app -t equipoise-app .
# With API_URL set (docker-compose) it calls the API; without it, it loads the bundle itself
# from /app/models (mount models/ there).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    EQUIPOISE_ROOT=/app MPLBACKEND=Agg \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-app.txt ./
RUN pip install "$(grep '^torch==' requirements.txt)" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -r requirements-app.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY config ./config
COPY app ./app
COPY reports ./reports
# synthetic patients for the 'load a test patient' demo; models/ is mounted at run time
COPY data/synthetic ./data/synthetic

RUN useradd --create-home equipoise && mkdir -p /app/logs && chown -R equipoise /app/logs
USER equipoise

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", \
     "--server.address=0.0.0.0", "--server.headless=true"]
