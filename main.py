"""Camera depth-estimation driver.

Streams RGB video from a standard camera and publishes AI-estimated depth frames
to the same twin, emulating a depth camera data flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict

from cyberwave import Cyberwave

from depth_estimation import (
    DEFAULT_DEPTH_MODEL_BACKEND,
    DepthAnythingV2OnnxConfig,
    DepthFramePublisher,
    VideoDepthAnythingConfig,
    create_depth_estimator,
)
from driver_utils import (
    get_first_env_value,
    list_cv2_cameras,
    parse_bool,
    parse_camera_id,
    parse_float,
    parse_int,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("camera-depth-estimation-driver")


def _load_twin_data() -> Dict[str, Any]:
    twin_json_path = os.getenv("CYBERWAVE_TWIN_JSON_FILE")
    if not twin_json_path:
        return {}
    try:
        with open(twin_json_path, "r", encoding="utf-8") as twin_file:
            return json.load(twin_file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read twin JSON file at %s", twin_json_path)
        return {}


def _validate_twin_data(twin_data: Dict[str, Any]) -> str:
    asset = twin_data.get("asset") or {}
    asset_key = asset.get("registry_id") or ""
    if not asset_key:
        raise ValueError("No asset.registry_id found in twin JSON")

    capabilities = twin_data.get("capabilities") or {}
    sensors = capabilities.get("sensors") or []
    if not sensors:
        logger.warning(
            "Twin JSON does not include capabilities.sensors. "
            "Driver will continue with metadata-based configuration."
        )
        return asset_key

    has_depth_sensor = any(sensor.get("type") == "depth" for sensor in sensors)
    if not has_depth_sensor:
        logger.warning(
            "Twin does not declare a depth sensor. Driver will still publish depth payloads."
        )
    return asset_key


def _build_model_config() -> VideoDepthAnythingConfig:
    encoder = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_ENCODER", "CYBERWAVE_DEPTH_MODEL_ENCODER"],
        default="vits",
    ) or "vits"
    metric = parse_bool(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_METRIC", "CYBERWAVE_DEPTH_MODEL_METRIC"],
            default="false",
        ),
        default=False,
    )
    fp32 = parse_bool(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_FP32", "CYBERWAVE_DEPTH_MODEL_FP32"],
            default="false",
        ),
        default=False,
    )
    input_size = parse_int(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_INPUT_SIZE", "CYBERWAVE_DEPTH_MODEL_INPUT_SIZE"],
            default="518",
        ),
        default=518,
        minimum=196,
    )
    auto_download = parse_bool(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_AUTO_DOWNLOAD", "CYBERWAVE_DEPTH_MODEL_AUTO_DOWNLOAD"],
            default="true",
        ),
        default=True,
    )
    repository_path = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_REPO_PATH", "CYBERWAVE_DEPTH_MODEL_REPO_PATH"],
        default="/opt/video-depth-anything",
    ) or "/opt/video-depth-anything"
    checkpoint_dir = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_CHECKPOINT_DIR", "CYBERWAVE_DEPTH_MODEL_CHECKPOINT_DIR"],
        default="/app/checkpoints",
    ) or "/app/checkpoints"
    checkpoint_path = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_CHECKPOINT_PATH", "CYBERWAVE_DEPTH_MODEL_CHECKPOINT_PATH"],
        default=None,
    )
    device = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_DEVICE", "CYBERWAVE_DEPTH_MODEL_DEVICE"],
        default="auto",
    ) or "auto"

    return VideoDepthAnythingConfig(
        encoder=encoder,
        metric=metric,
        fp32=fp32,
        input_size=input_size,
        auto_download=auto_download,
        repository_path=repository_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def _build_depth_anything_v2_onnx_config() -> DepthAnythingV2OnnxConfig:
    encoder = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_ENCODER", "CYBERWAVE_DEPTH_MODEL_ENCODER"],
        default="vits",
    ) or "vits"
    auto_download = parse_bool(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_AUTO_DOWNLOAD", "CYBERWAVE_DEPTH_MODEL_AUTO_DOWNLOAD"],
            default="true",
        ),
        default=True,
    )
    model_dir = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_CHECKPOINT_DIR", "CYBERWAVE_DEPTH_MODEL_CHECKPOINT_DIR"],
        default="/app/checkpoints",
    ) or "/app/checkpoints"
    model_path = get_first_env_value(
        ["CYBERWAVE_METADATA_DEPTH_MODEL_ONNX_PATH", "CYBERWAVE_DEPTH_MODEL_ONNX_PATH"],
        default=None,
    )
    input_height = parse_int(
        get_first_env_value(
            [
                "CYBERWAVE_METADATA_DEPTH_MODEL_ONNX_INPUT_HEIGHT",
                "CYBERWAVE_DEPTH_MODEL_ONNX_INPUT_HEIGHT",
            ],
            default="320",
        ),
        default=320,
        minimum=96,
        maximum=1080,
    )
    input_width = parse_int(
        get_first_env_value(
            [
                "CYBERWAVE_METADATA_DEPTH_MODEL_ONNX_INPUT_WIDTH",
                "CYBERWAVE_DEPTH_MODEL_ONNX_INPUT_WIDTH",
            ],
            default="0",
        ),
        default=0,
        minimum=0,
        maximum=1920,
    )
    provider = (
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_ONNX_PROVIDER", "CYBERWAVE_DEPTH_MODEL_ONNX_PROVIDER"],
            default="cpu",
        )
        or "cpu"
    )
    return DepthAnythingV2OnnxConfig(
        model_path=model_path,
        model_dir=model_dir,
        encoder=encoder,
        auto_download=auto_download,
        input_height=input_height,
        input_width=input_width or None,
        provider=provider,
    )


async def main() -> None:
    token = os.getenv("CYBERWAVE_API_KEY")
    twin_uuid = os.getenv("CYBERWAVE_TWIN_UUID")

    if not token:
        logger.error("CYBERWAVE_API_KEY environment variable is required")
        sys.exit(1)
    if not twin_uuid:
        logger.error("CYBERWAVE_TWIN_UUID environment variable is required")
        sys.exit(1)

    twin_data = _load_twin_data()
    asset_key = _validate_twin_data(twin_data)

    video_device = get_first_env_value(
        ["CYBERWAVE_METADATA_VIDEO_DEVICE", "CYBERWAVE_METADATA_CAMERA_SOURCE"],
        default="0",
    ) or "0"
    camera_id = parse_camera_id(video_device)
    fps = parse_int(
        get_first_env_value(["CYBERWAVE_METADATA_CAMERA_FPS", "CYBERWAVE_METADATA_FPS"], default="15"),
        default=15,
        minimum=1,
        maximum=60,
    )
    width = parse_int(
        get_first_env_value(["CYBERWAVE_METADATA_CAMERA_WIDTH"], default="640"),
        default=640,
        minimum=160,
        maximum=3840,
    )
    height = parse_int(
        get_first_env_value(["CYBERWAVE_METADATA_CAMERA_HEIGHT"], default="480"),
        default=480,
        minimum=120,
        maximum=2160,
    )
    fourcc = get_first_env_value(["CYBERWAVE_METADATA_CAMERA_FOURCC"], default=None)

    publish_interval = parse_int(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_PUBLISH_INTERVAL", "CYBERWAVE_DEPTH_PUBLISH_INTERVAL"],
            default="5",
        ),
        default=5,
        minimum=1,
    )
    output_mode = (
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_OUTPUT_MODE", "CYBERWAVE_DEPTH_OUTPUT_MODE"],
            default="normalized_uint16",
        )
        or "normalized_uint16"
    )
    scale_factor = parse_float(
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_SCALE_FACTOR", "CYBERWAVE_DEPTH_SCALE_FACTOR"],
            default="1000.0",
        ),
        default=1000.0,
        minimum=0.001,
    )

    logger.info(
        "Initializing camera depth-estimation driver for twin=%s asset=%s camera=%s fps=%s",
        twin_uuid,
        asset_key,
        camera_id,
        fps,
    )

    model_backend = (
        get_first_env_value(
            ["CYBERWAVE_METADATA_DEPTH_MODEL_BACKEND", "CYBERWAVE_DEPTH_MODEL_BACKEND"],
            default=DEFAULT_DEPTH_MODEL_BACKEND,
        )
        or DEFAULT_DEPTH_MODEL_BACKEND
    )
    logger.info("Depth model backend selected: %s", model_backend)
    model_config = _build_model_config()
    onnx_config_placeholder = _build_depth_anything_v2_onnx_config()
    estimator = create_depth_estimator(
        model_backend=model_backend,
        video_depth_anything_config=model_config,
        depth_anything_v2_onnx_config=onnx_config_placeholder,
    )

    client = Cyberwave(api_key=token, source_type="edge")
    # Keep the same init behavior as other edge camera drivers.
    _ = client.twin(asset_key=asset_key, twin_id=twin_uuid)

    depth_callback = DepthFramePublisher(
        mqtt_client=client.mqtt,
        twin_uuid=twin_uuid,
        estimator=estimator,
        publish_interval=publish_interval,
        output_mode=output_mode,
        scale_factor=scale_factor,
    )

    stop_event = asyncio.Event()
    streamer = None

    def _handle_signal() -> None:
        logger.info("Shutdown signal received, stopping...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    async def _start_stream(selected_camera_id: Any):
        return_streamer = client.video_stream(
            twin_uuid=twin_uuid,
            camera_type="cv2",
            camera_id=selected_camera_id,
            fps=fps,
            resolution=(width, height),
            frame_callback=depth_callback,
            fourcc=fourcc,
        )
        await return_streamer.start()
        return return_streamer

    try:
        try:
            streamer = await _start_stream(camera_id)
        except Exception:
            logger.exception(
                "Camera stream failed with configured device '%s', trying auto-detect fallback",
                camera_id,
            )
            fallback_candidates = list_cv2_cameras()
            if not fallback_candidates:
                raise

            fallback_camera_id = parse_camera_id(fallback_candidates[0])
            logger.info(
                "Retrying camera stream using auto-detected fallback device '%s'",
                fallback_camera_id,
            )
            streamer = await _start_stream(fallback_camera_id)

        logger.info("Camera stream started. Waiting for shutdown signal...")
        await stop_event.wait()
    except Exception:
        logger.exception("Camera streaming failed")
    finally:
        if streamer is not None:
            logger.info("Stopping camera stream...")
            try:
                await streamer.stop()
            except Exception:
                logger.exception("Error while stopping camera streamer")
        client.disconnect()
        logger.info("Camera depth-estimation driver stopped.")


if __name__ == "__main__":
    asyncio.run(main())
