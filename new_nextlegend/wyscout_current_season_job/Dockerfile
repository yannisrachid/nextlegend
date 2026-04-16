FROM mcr.microsoft.com/playwright/python:v1.52.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install dependencies for both the scraper and the pipeline runner.
COPY wyscout_current_season_job/requirements.txt /tmp/requirements-scraper.txt
COPY jobs/pipeline/requirements.txt /tmp/requirements-pipeline.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-scraper.txt -r /tmp/requirements-pipeline.txt && \
    python -m playwright install chromium

WORKDIR /workspace/wyscout_current_season_job

ENTRYPOINT ["bash", "./run_current_season_e2e.sh"]
