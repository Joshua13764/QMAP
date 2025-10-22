# ---------- Dockerfile (MLtools BOULDERING, CPU PoC) ----------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps (GDAL/PROJ etc. for raster/shp tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    gdal-bin libgdal-dev libspatialindex-dev proj-bin libproj-dev \
    libgl1 libglib2.0-0 ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep build tooling compatible with older stacks
RUN python -m pip install --upgrade "pip<23" "setuptools<70" "wheel<0.41"

# --- bring in your version pins early ---
COPY constraints.txt /tmp/constraints.txt

# PyTorch trio (CPU) — exact pairing
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu -c /tmp/constraints.txt

# Detectron2 v0.6 from source
RUN git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git /opt/detectron2 && \
    pip install -e /opt/detectron2 -c /tmp/constraints.txt

# rastertools
RUN git clone https://github.com/astroNils/rastertools.git /opt/rastertools && \
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING && \
    pip install -r /opt/rastertools/requirements.txt -c /tmp/constraints.txt

# shptools
RUN git clone https://github.com/astroNils/shptools.git /opt/shptools && \
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING && \
    pip install -r /opt/shptools/requirements.txt -c /tmp/constraints.txt

# MLtools
RUN git clone https://github.com/astroNils/MLtools.git /opt/MLtools && \
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps MLtools_BOULDERING && \
    pip install -r /opt/MLtools/requirements.txt -c /tmp/constraints.txt

# Enforce Pillow 9.5.0 last (fixes Image.LINEAR issue in D2 v0.6)
RUN pip install "Pillow==9.5.0"

# Make the in-repo src importable
ENV PYTHONPATH=/opt/MLtools/src:$PYTHONPATH

WORKDIR /workspace

# PoC import check (no fallbacks)
CMD ["python", "-c", "from MLtools import inference; print('MLtools inference import OK')"]
