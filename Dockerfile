# Matches runtime.txt (python-3.12.10) so local, CI, and container
# behavior stay in sync instead of drifting apart.
FROM python:3.12.10-slim

WORKDIR /app

# System deps for xgboost / scikit-learn wheels that need a compiler on
# some platforms; kept minimal and removed from the final layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY models/ models/

# Note: models/*.json and *.pkl (the trained model artifacts) are part of
# models/ and get copied above. If you regenerate them with
# `python -m models.train`, rebuild the image to pick up the new files.

EXPOSE 8000

# --host 0.0.0.0 is required for the server to be reachable from outside
# the container; --reload is a dev-only flag and is intentionally omitted
# here (see docker-compose.override or run locally without Docker for that).
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
