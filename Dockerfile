FROM python:3.12-slim

ARG VIDEO_DEPTH_ANYTHING_REF=main
ARG INSTALL_DEPTH_MODEL_DEPS=true
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pull official Video-Depth-Anything source (no pip package available yet).
RUN git clone --depth 1 --branch "${VIDEO_DEPTH_ANYTHING_REF}" \
    https://github.com/DepthAnything/Video-Depth-Anything.git /opt/video-depth-anything

ENV PYTHONPATH="/opt/video-depth-anything"

COPY pyproject.toml .
COPY *.py ./
COPY README.md .

RUN pip install --no-cache-dir .

# Torch Linux default wheels can pull CUDA packages that are too large for CI.
# For deterministic edge builds we install CPU wheels by default.
RUN if [ "${INSTALL_DEPTH_MODEL_DEPS}" = "true" ]; then \
      pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision && \
      pip install --no-cache-dir "einops>=0.4.1" "easydict>=1.13" "onnxruntime>=1.18"; \
    fi

RUN mkdir -p /app/.cyberwave /app/checkpoints

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
