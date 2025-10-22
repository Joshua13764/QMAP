FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    gdal-bin libgdal-dev libspatialindex-dev proj-bin libproj-dev \
    libgl1 libglib2.0-0 ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<23" "setuptools<70" "wheel<0.41"

RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git /opt/detectron2 && \
    PIP_USE_PEP517=0 PIP_NO_BUILD_ISOLATION=1 pip install -e /opt/detectron2

RUN git clone https://github.com/astroNils/rastertools.git /opt/rastertools && \
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING && \
    pip install -r /opt/rastertools/requirements.txt

RUN git clone https://github.com/astroNils/shptools.git /opt/shptools && \
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING && \
    pip install -r /opt/shptools/requirements.txt

RUN git clone https://github.com/astroNils/MLtools.git /opt/MLtools && \
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps MLtools_BOULDERING && \
    pip install -r /opt/MLtools/requirements.txt

ENV PYTHONPATH=/opt/MLtools/src:$PYTHONPATH

WORKDIR /workspace

CMD ["python", "-c", "import MLtools as m; print('MLtools ready:', getattr(m,'__file__','ok'))"]
