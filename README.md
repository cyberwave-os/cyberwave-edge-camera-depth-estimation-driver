# Cyberwave Camera Depth Estimation Driver

Edge driver that streams RGB video from a standard camera and publishes AI-estimated depth maps using [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything).

## Metadata config

```json
"drivers": {
  "default": {
    "docker_image": "cyberwaveos/camera-depth-estimation-driver",
    "params": ["--device /dev/video0:/dev/video0"]
  }
}
```

## Runtime env/metadata

Required:

- `CYBERWAVE_API_KEY`
- `CYBERWAVE_TWIN_UUID`
- `CYBERWAVE_TWIN_JSON_FILE`

Camera:

- `metadata.video_device` (default: `"0"`)
- `metadata.camera_fps` (default: `15`)
- `metadata.camera_width` / `metadata.camera_height` (default: `640x480`)
- `metadata.camera_fourcc` (optional, e.g. `MJPG`)

Depth estimation:

- `metadata.depth_model_backend` (default `video_depth_anything_stream`)
  - `video_depth_anything_stream`: current implementation (Video-Depth-Anything)
  - `depth_anything_v2_onnx`: realtime per-frame ONNX backend (Depth-Anything-V2)
- `metadata.depth_model_encoder` (`vits` | `vitb` | `vitl`, default `vits`)
- `metadata.depth_model_metric` (`true`/`false`, default `false`)
- `metadata.depth_model_input_size` (default `518`)
- `metadata.depth_model_checkpoint_path` (optional explicit path)
- `metadata.depth_model_checkpoint_dir` (default `/app/checkpoints`)
- `metadata.depth_model_auto_download` (default `true`)
- `metadata.depth_model_onnx_path` (optional explicit ONNX path)
- `metadata.depth_model_onnx_input_height` (default `320`; reduced resolution for realtime)
- `metadata.depth_model_onnx_input_width` (default auto from aspect ratio; rounded to multiple of 14)
- `metadata.depth_model_onnx_provider` (`cpu` | `cuda` | `auto`, default `cpu`)
- `metadata.depth_publish_interval` (publish every N RGB frames, default `5`)
- `metadata.depth_output_mode` (`normalized_uint16` or `metric_mm`, default `normalized_uint16`)
- `metadata.depth_scale_factor` (used in `metric_mm` mode, default `1000.0`)

## Notes

- Depth is published to `cyberwave/twin/{twin_uuid}/depth` using the SDK format:
  - `type = "depth_data"`
  - `data.depth_binary` (base64-encoded `uint16`)
  - `data.width`, `data.height`, `data.dtype = "uint16"`
- RGB video is streamed through standard Cyberwave WebRTC flow (`camera_type=cv2`).
- `vits` checkpoints are Apache-2.0; larger model licenses may differ in the upstream project.
- ONNX backend defaults to dynamic Depth-Anything-V2 models from `fabio-sim/Depth-Anything-ONNX` release `v2.0.0`.
