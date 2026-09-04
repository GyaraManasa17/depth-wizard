FROM python:3.11-slim

# rasterio (via GDAL) needs these system libraries, which the slim image doesn't include by default
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install backend dependencies first (better Docker layer caching -
# this step only re-runs if requirements.txt changes, not on every code edit)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements.txt

# Now copy the actual application code
COPY backend/ backend/
COPY frontend/ frontend/

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]