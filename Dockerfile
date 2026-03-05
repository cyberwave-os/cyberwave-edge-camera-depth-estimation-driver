FROM python:3.12-slim

ARG VIDEO_DEPTH_ANYTHING_REF=main

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

ENV PYTHONPATH="/opt/video-depth-anything:${PYTHONPATH}"

COPY pyproject.toml .
COPY *.py ./
COPY README.md .

RUN pip install --no-cache-dir .

RUN mkdir -p /app/.cyberwave /app/checkpoints

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
