# Container for running the pre-compute pass (scripts/build_webapp.py) as a
# SageMaker Processing job.
#
# CPU-only torch on purpose: the CUDA wheels add ~2.5 GB to the image, and the
# push from a laptop dominates the wall-clock cost of the whole exercise. The
# detection pass is minutes-scale on an ml.m5.2xlarge, which is cheap enough.
FROM python:3.11-slim

# opencv (even headless) needs libgl/libglib present at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# Torch first, from the CPU index, so the ultralytics install below resolves
# against it instead of pulling the default CUDA build.
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        numpy \
        polars \
        pyarrow \
        lightgbm \
        scikit-learn \
        opencv-python-headless \
        ultralytics

# Ultralytics phones home and writes config on first run; neither works well in
# a job container with no writable HOME.
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    ULTRALYTICS_NO_ANALYTICS=1 \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1

# The application: reasoning source, the trained phase-2 models (~1.2 MB), and
# the YOLO weights. Baking the weights in keeps the job hermetic — ultralytics
# would otherwise try to download yolo11l.pt on first use.
COPY src/                 /opt/program/src/
COPY scripts/             /opt/program/scripts/
COPY models/phase2/       /opt/program/models/phase2/
COPY yolo11l.pt           /opt/program/yolo11l.pt

COPY deploy/aws/sagemaker/entrypoint.py /opt/program/entrypoint.py

ENTRYPOINT ["python", "/opt/program/entrypoint.py"]
