"""Utility helpers for the camera depth-estimation driver."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

logger = logging.getLogger(__name__)


def get_first_env_value(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """Return first non-empty environment value from keys."""
    import os

    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def parse_camera_id(video_device: str) -> Union[int, str]:
    """Parse camera metadata into SDK-compatible camera_id."""
    try:
        return int(video_device)
    except ValueError:
        return video_device


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse bool-like strings from env vars."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def parse_int(
    value: Optional[str],
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Parse and clamp an integer-like env var value."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default

    if minimum is not None and parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def parse_float(
    value: Optional[str],
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Parse and clamp a float-like env var value."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default

    if minimum is not None and parsed < minimum:
        return minimum
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def list_cv2_cameras(max_devices: int = 10) -> list[str]:
    """List available OpenCV cameras by index."""
    indices: list[str] = []
    try:
        import cv2

        for index in range(max_devices):
            cap = cv2.VideoCapture(index)
            try:
                if cap.isOpened():
                    indices.append(str(index))
            finally:
                cap.release()
    except Exception:
        logger.exception("Failed to enumerate CV2 cameras")
    return indices
