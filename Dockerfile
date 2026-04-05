# ── Stage: Build & Train ──────────────────────────────────────────────────────
FROM python:3.11-slim

# Install system deps needed by numpy/pandas/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source & dataset
COPY . .

# Train the model at build time → artifacts stored in /app/model/
RUN python train.py

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2", "app:app"]
