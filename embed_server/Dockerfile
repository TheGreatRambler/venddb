FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Clone siglip2 model via git+LFS during build so startup is fast.
# Downloading this model takes a long time as it is around 7.5GB.
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install
RUN GIT_LFS_PROGRESS=/dev/stderr git clone --progress \
    https://huggingface.co/google/siglip2-giant-opt-patch16-384 \
    /app/model 2>&1

COPY embed_service.py .

# Run with gunicorn. --preload loads the model once in the master before forking workers.
# --timeout is bumped to 300s because CPU inference for siglip2-giant can take ~30-60s per request.
CMD ["gunicorn", "--bind", "0.0.0.0:5936", "--workers", "1", \
    "--threads", "4", "--timeout", "300", "--preload", "embed_service:app"]
