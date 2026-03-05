"""Video-Depth-Anything integration and depth payload publishing."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

RELATIVE_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true",
}

METRIC_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth?download=true",
}


@dataclass
class VideoDepthAnythingConfig:
    """Model configuration loaded from environment variables."""

    encoder: str = "vits"
    metric: bool = False
    fp32: bool = False
    input_size: int = 518
    auto_download: bool = True
    repository_path: str = "/opt/video-depth-anything"
    checkpoint_dir: str = "/app/checkpoints"
    checkpoint_path: Optional[str] = None
    device: str = "auto"

    def checkpoint_filename(self) -> str:
        checkpoint_prefix = "metric_video_depth_anything" if self.metric else "video_depth_anything"
        return f"{checkpoint_prefix}_{self.encoder}.pth"

    def resolved_checkpoint_path(self) -> str:
        if self.checkpoint_path:
            return self.checkpoint_path
        return os.path.join(self.checkpoint_dir, self.checkpoint_filename())


class VideoDepthAnythingEstimator:
    """Frame-by-frame depth estimator backed by Video-Depth-Anything."""

    def __init__(self, config: VideoDepthAnythingConfig):
        self.config = config
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._model is not None

    def _download_checkpoint_if_needed(self, checkpoint_path: str) -> None:
        if os.path.exists(checkpoint_path):
            return
        if not self.config.auto_download:
            raise FileNotFoundError(
                f"Model checkpoint not found at '{checkpoint_path}' and auto-download is disabled."
            )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        urls = METRIC_CHECKPOINT_URLS if self.config.metric else RELATIVE_CHECKPOINT_URLS
        download_url = urls.get(self.config.encoder)
        if not download_url:
            raise ValueError(f"Unsupported encoder '{self.config.encoder}'")

        logger.info("Downloading Video-Depth-Anything checkpoint from %s", download_url)
        with urllib.request.urlopen(download_url, timeout=120) as response:
            with open(checkpoint_path, "wb") as checkpoint_file:
                shutil.copyfileobj(response, checkpoint_file)
        logger.info("Downloaded checkpoint to %s", checkpoint_path)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            import sys

            if self.config.repository_path not in sys.path:
                sys.path.insert(0, self.config.repository_path)

            try:
                import torch
                from video_depth_anything.video_depth_stream import (
                    VideoDepthAnything as VideoDepthAnythingModel,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import Video-Depth-Anything. "
                    "Ensure /opt/video-depth-anything is present and dependencies are installed."
                ) from exc

            if self.config.encoder not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unsupported encoder '{self.config.encoder}'. "
                    f"Use one of {sorted(MODEL_CONFIGS.keys())}."
                )

            checkpoint_path = self.config.resolved_checkpoint_path()
            self._download_checkpoint_if_needed(checkpoint_path)

            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device

            logger.info(
                "Loading Video-Depth-Anything model (encoder=%s, metric=%s, device=%s, checkpoint=%s)",
                self.config.encoder,
                self.config.metric,
                device,
                checkpoint_path,
            )
            model = VideoDepthAnythingModel(**MODEL_CONFIGS[self.config.encoder])
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
            model = model.to(device).eval()

            self._torch = torch
            self._model = model
            self._device = device
            logger.info("Video-Depth-Anything model loaded")

    def estimate_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth for a single BGR frame."""
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("Model failed to initialize")

        # Video-Depth-Anything expects RGB input.
        frame_rgb = frame_bgr[:, :, ::-1]
        with self._lock:
            depth = self._model.infer_video_depth_one(
                frame_rgb,
                input_size=self.config.input_size,
                device=self._device,
                fp32=self.config.fp32,
            )
        return np.asarray(depth, dtype=np.float32)


def depth_to_uint16(
    depth: np.ndarray,
    *,
    output_mode: str = "normalized_uint16",
    scale_factor: float = 1000.0,
) -> np.ndarray:
    """Convert floating-point depth map to uint16 payload."""
    clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if output_mode == "metric_mm":
        metric = np.clip(clean * scale_factor, 0, 65535)
        return metric.astype(np.uint16)

    if output_mode != "normalized_uint16":
        raise ValueError(
            f"Unsupported output_mode '{output_mode}'. "
            "Use 'normalized_uint16' or 'metric_mm'."
        )

    min_depth = float(np.min(clean))
    max_depth = float(np.max(clean))
    if max_depth - min_depth < 1e-8:
        return np.zeros_like(clean, dtype=np.uint16)

    normalized = (clean - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized * 65535.0, 0.0, 65535.0)
    return normalized.astype(np.uint16)


class DepthFramePublisher:
    """Publish AI-estimated depth frames in SDK-compatible depth payload format."""

    def __init__(
        self,
        mqtt_client: Any,
        twin_uuid: str,
        estimator: VideoDepthAnythingEstimator,
        *,
        publish_interval: int = 5,
        output_mode: str = "normalized_uint16",
        scale_factor: float = 1000.0,
    ):
        self.mqtt_client = mqtt_client
        self.twin_uuid = twin_uuid
        self.estimator = estimator
        self.publish_interval = max(1, int(publish_interval))
        self.output_mode = output_mode
        self.scale_factor = scale_factor
        self._has_logged_inference_error = False

    def __call__(self, frame_bgr: np.ndarray, frame_count: int) -> None:
        if frame_count % self.publish_interval != 0:
            return

        try:
            depth = self.estimator.estimate_depth(frame_bgr)
        except Exception:
            if not self._has_logged_inference_error:
                logger.exception("Depth estimation failed; continuing RGB-only stream")
                self._has_logged_inference_error = True
            return

        depth_u16 = depth_to_uint16(
            depth,
            output_mode=self.output_mode,
            scale_factor=self.scale_factor,
        )
        height, width = depth_u16.shape[:2]
        depth_binary = base64.b64encode(depth_u16.tobytes()).decode("utf-8")
        depth_data = {
            "depth_binary": depth_binary,
            "width": width,
            "height": height,
            "dtype": "uint16",
        }
        self.mqtt_client.publish_depth_frame(self.twin_uuid, depth_data, timestamp=time.time())
