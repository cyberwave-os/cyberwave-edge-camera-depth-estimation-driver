#!/bin/bash
# =============================================================================
# Camera Depth Estimation Driver End-to-End Test
# =============================================================================
#
# This test validates the new camera depth-estimation driver end-to-end:
#   1. Starts local backend (or reuses an existing one) and seeds it
#   2. Builds the depth-estimation driver Docker image
#   3. Pushes it to a local Docker registry (localhost:5000)
#   4. Starts an RGB camera emulator (RTSP server + synthetic RGB publisher)
#   5. Builds a "Pi simulator" container (CLI + edge-core + SDK)
#   6. Inside the Pi simulator:
#      a. Logs in via CLI
#      b. Creates project/environment and a `the-robot-studio/so101` twin
#      c. Updates twin metadata to force the just-built driver image
#      d. Writes edge config files and runs edge-core driver discovery
#      e. Verifies driver container is running and producing startup logs
#
# Usage:
#   cd cyberwave-edge-nodes/cyberwave-edge-camera-depth-estimation
#   bash test-depth-estimation-e2e.sh
#   bash test-depth-estimation-e2e.sh --skip-build
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$REPO_ROOT/cyberwave-backend"

DRIVER_IMAGE_LOCAL="localhost:5000/camera-depth-estimation-e2e:latest"
DRIVER_IMAGE_BUILD="camera-depth-estimation-e2e:latest"
PI_SIM_IMAGE="camera-depth-estimation-pi-sim:latest"
EDGE_CONFIG_DIR_HOST="$REPO_ROOT/.cyberwave-depth-e2e-$$"

REGISTRY_CONTAINER="cyberwave-local-registry"
RTSP_SERVER_CONTAINER="cyberwave-rgb-rtsp-server"
RTSP_PUBLISHER_CONTAINER="cyberwave-rgb-rtsp-publisher"
SKIP_BUILD=false

TEST_EMAIL="admin@cyberwave.com"
TEST_PASSWORD="admin123"

DOCKER_SOCK="/var/run/docker.sock"
for _candidate in \
    "${DOCKER_HOST:-}" \
    "$HOME/.docker/run/docker.sock" \
    "/var/run/docker.sock"; do
    if [ "${_candidate#unix://}" != "$_candidate" ]; then
        _candidate="${_candidate#unix://}"
    fi
    if [ -z "$_candidate" ]; then
        continue
    fi
    if [ -S "$_candidate" ]; then
        DOCKER_SOCK="$_candidate"
        break
    fi
done

for arg in "$@"; do
    case "$arg" in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: bash test-depth-estimation-e2e.sh [--skip-build]"
            exit 0
            ;;
    esac
done

BACKEND_STARTED=false

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    docker rm -f "$RTSP_PUBLISHER_CONTAINER" 2>/dev/null || true
    docker rm -f "$RTSP_SERVER_CONTAINER" 2>/dev/null || true
    docker rm -f "$REGISTRY_CONTAINER" 2>/dev/null || true
    docker rm -f cyberwave-driver-* 2>/dev/null || true
    rm -rf "$EDGE_CONFIG_DIR_HOST" 2>/dev/null || true
    if [ "$BACKEND_STARTED" = true ]; then
        echo "Stopping backend..."
        cd "$BACKEND_DIR" && docker compose -f local.yml down --remove-orphans 2>/dev/null || true
    fi
    echo "Done."
}

trap cleanup EXIT

echo ""
echo "=========================================="
echo " Step 1: Starting backend locally"
echo "=========================================="

cd "$BACKEND_DIR"
if [ ! -f "$BACKEND_DIR/.env" ]; then
    if [ -f "$BACKEND_DIR/.env.example" ]; then
        cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
        echo "Created backend .env from .env.example for local/CI compose runs."
    else
        echo "ERROR: Missing $BACKEND_DIR/.env and .env.example"
        exit 1
    fi
fi
if curl -sf http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "Backend is already running, skipping startup."
else
    docker compose -f local.yml up -d
    BACKEND_STARTED=true
fi

echo ""
echo "=========================================="
echo " Step 2: Waiting for backend health check"
echo "=========================================="

MAX_RETRIES=120
RETRY_COUNT=0
until curl -sf http://localhost:8000/healthz > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
        echo "ERROR: Backend did not become healthy within ${MAX_RETRIES} seconds"
        docker compose -f local.yml logs --tail=100 django
        exit 1
    fi
    printf "\r  Waiting for backend... (%d/%d)" "$RETRY_COUNT" "$MAX_RETRIES"
    sleep 1
done
echo ""
echo "Backend is healthy!"

echo ""
echo "=========================================="
echo " Step 3: Seeding backend data"
echo "=========================================="

if [ "$BACKEND_STARTED" = true ]; then
    docker compose -f local.yml exec -T django python manage.py migrate --run-syncdb --no-input 2>/dev/null || true
    docker compose -f local.yml exec -T django python manage.py seed_data --skip-assets --skip-projects
else
    echo "Backend already running; skipping seed_data."
fi

echo ""
echo "=========================================="
echo " Step 4: Generating local Python REST SDK"
echo "=========================================="

cd "$REPO_ROOT/cyberwave-sdks"
./python-sdk-gen.sh sdk --host localhost:8000
echo "  ✅ Local SDK REST client generated"

echo ""
echo "=========================================="
echo " Step 5: Building driver image + local registry push"
echo "=========================================="

cd "$SCRIPT_DIR"
if [ "$SKIP_BUILD" = true ]; then
    if docker image inspect "$DRIVER_IMAGE_LOCAL" >/dev/null 2>&1; then
        echo "Skipping build: found $DRIVER_IMAGE_LOCAL"
    else
        echo "ERROR: --skip-build requested but '$DRIVER_IMAGE_LOCAL' not found"
        exit 1
    fi
else
    docker build --build-arg INSTALL_DEPTH_MODEL_DEPS=false -t "$DRIVER_IMAGE_BUILD" .
    docker rm -f "$REGISTRY_CONTAINER" 2>/dev/null || true
    docker run -d --name "$REGISTRY_CONTAINER" -p 5000:5000 registry:2
    docker tag "$DRIVER_IMAGE_BUILD" "$DRIVER_IMAGE_LOCAL"
    docker push "$DRIVER_IMAGE_LOCAL"
fi

echo ""
echo "=========================================="
echo " Step 6: Starting RGB camera emulator"
echo "=========================================="

docker rm -f "$RTSP_SERVER_CONTAINER" 2>/dev/null || true
docker rm -f "$RTSP_PUBLISHER_CONTAINER" 2>/dev/null || true

docker run -d --name "$RTSP_SERVER_CONTAINER" --network host bluenviron/mediamtx:latest
sleep 2
docker run -d --name "$RTSP_PUBLISHER_CONTAINER" --network host \
    jrottenberg/ffmpeg:6.0-alpine \
    -hide_banner -loglevel warning \
    -re -stream_loop -1 \
    -f lavfi -i "testsrc=size=640x480:rate=15" \
    -pix_fmt yuv420p \
    -c:v mpeg4 \
    -f rtsp -rtsp_transport tcp \
    rtsp://127.0.0.1:8554/camera
sleep 4

echo "  ✅ RGB RTSP camera emulator running at rtsp://127.0.0.1:8554/camera"

echo ""
echo "=========================================="
echo " Step 7: Building edge test (Pi simulator) image"
echo "=========================================="

docker build \
    --build-arg CYBERWAVE_SDK_REL_PATH=cyberwave-sdks/cyberwave-python \
    -t "$PI_SIM_IMAGE" \
    -f - "$REPO_ROOT" <<'DOCKERFILE'
FROM docker:cli AS docker-cli

FROM python:3.12-slim
ARG CYBERWAVE_SDK_REL_PATH

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV CLI_VENV=/opt/venvs/cli
ENV EDGE_VENV=/opt/venvs/edge-core
ENV PATH="$CLI_VENV/bin:$EDGE_VENV/bin:$PATH"

COPY ${CYBERWAVE_SDK_REL_PATH}/ /workspace/cyberwave-sdk/
COPY cyberwave-clis/cyberwave-python-cli/ /workspace/cyberwave-cli/
COPY cyberwave-edge-nodes/cyberwave-edge-core/ /workspace/cyberwave-edge-core/

RUN python -m venv "$CLI_VENV" && \
    "$CLI_VENV/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel && \
    "$CLI_VENV/bin/pip" install --no-cache-dir -e "/workspace/cyberwave-sdk" && \
    "$CLI_VENV/bin/pip" install --no-cache-dir -e "/workspace/cyberwave-cli/"

RUN python -m venv "$EDGE_VENV" && \
    "$EDGE_VENV/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel && \
    "$EDGE_VENV/bin/pip" install --no-cache-dir -e "/workspace/cyberwave-sdk" && \
    "$EDGE_VENV/bin/pip" install --no-cache-dir -e "/workspace/cyberwave-edge-core/"
DOCKERFILE

echo ""
echo "=========================================="
echo " Step 8: Edge-core E2E run (force driver metadata)"
echo "=========================================="

mkdir -p "$EDGE_CONFIG_DIR_HOST"
docker run --rm -i \
    --add-host=host.docker.internal:host-gateway \
    -e CYBERWAVE_BASE_URL=http://host.docker.internal:8000 \
    -e CYBERWAVE_MQTT_HOST=host.docker.internal \
    -e CYBERWAVE_ENVIRONMENT=local \
    -e CYBERWAVE_EDGE_CONFIG_DIR="$EDGE_CONFIG_DIR_HOST" \
    -e DRIVER_IMAGE="$DRIVER_IMAGE_LOCAL" \
    -v "$DOCKER_SOCK":/var/run/docker.sock \
    -v "$EDGE_CONFIG_DIR_HOST:$EDGE_CONFIG_DIR_HOST" \
    "$PI_SIM_IMAGE" \
    bash -c "
set -euo pipefail
export PATH=\"/opt/venvs/cli/bin:/opt/venvs/edge-core/bin:\$PATH\"

cyberwave login --email '$TEST_EMAIL' --password '$TEST_PASSWORD'

/opt/venvs/edge-core/bin/python3 - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
from cyberwave import Cyberwave
from cyberwave.fingerprint import generate_fingerprint
from cyberwave_edge_core.startup import fetch_and_run_twin_drivers

driver_image = os.environ['DRIVER_IMAGE']
base_url = os.environ.get('CYBERWAVE_BASE_URL', 'http://host.docker.internal:8000')
config_dir = Path(os.environ.get('CYBERWAVE_EDGE_CONFIG_DIR', '/etc/cyberwave'))
credentials_path = config_dir / 'credentials.json'
assert credentials_path.exists(), f'Missing credentials file at {credentials_path}'
credentials_payload = json.loads(credentials_path.read_text())
token = credentials_payload.get('token')
workspace_uuid = credentials_payload.get('workspace_uuid')
assert token, 'Missing CLI credentials/token after login'
assert workspace_uuid, 'Missing workspace UUID in CLI credentials'
credentials_envs = credentials_payload.get('envs', {})
if not isinstance(credentials_envs, dict):
    credentials_envs = {}
api_url = credentials_envs.get('CYBERWAVE_BASE_URL') or base_url

client = Cyberwave(base_url=api_url, api_key=token)
suffix = str(int(time.time()))
project = client.projects.create(
    name=f'Camera Depth Estimation E2E {suffix}',
    workspace_id=workspace_uuid,
    description='E2E test for camera depth estimation driver',
)
environment = client.environments.create(
    name=f'Camera Depth Estimation Env {suffix}',
    project_id=str(project.uuid),
    description='E2E test env',
)
env_uuid = str(environment.uuid)
twin = client.twin('the-robot-studio/so101', environment_id=env_uuid)
twin_uuid = str(twin.uuid)
fingerprint = generate_fingerprint()

headers = {
    'Authorization': f'Token {token}',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}
metadata = {
    'edge_fingerprint': fingerprint,
    'video_device': 'rtsp://127.0.0.1:8554/camera',
    'camera_fps': '10',
    'camera_width': '640',
    'camera_height': '480',
    # Keep test deterministic and lightweight: do not download model weights.
    # The driver should still run RGB stream and handle missing depth model gracefully.
    'depth_model_auto_download': 'false',
    'depth_model_checkpoint_path': '/app/checkpoints/missing-test-checkpoint.pth',
    'depth_publish_interval': '10',
    'drivers': {
        'default': {
            'docker_image': driver_image,
        }
    },
}
resp = httpx.put(
    f'{base_url}/api/v1/twins/{twin_uuid}',
    headers=headers,
    json={'metadata': metadata},
    timeout=20.0,
)
assert resp.status_code == 200, (
    f'Failed to update twin metadata: status={resp.status_code}, body={resp.text[:300]}'
)

config_dir.mkdir(parents=True, exist_ok=True)
(config_dir / 'fingerprint.json').write_text(json.dumps({'fingerprint': fingerprint}, indent=2) + '\\n')
(config_dir / 'environment.json').write_text(
    json.dumps({'uuid': env_uuid, 'workspace_uuid': workspace_uuid, 'twin_uuids': [twin_uuid]}, indent=2) + '\\n'
)

results = fetch_and_run_twin_drivers(token, env_uuid, fingerprint)
target = next((r for r in results if r.get('twin_uuid') == twin_uuid), None)
assert target is not None, f'No startup result found for test twin: {results}'
assert target.get('success') is True, f'Driver startup reported failure: {target}'

container_name = f'cyberwave-driver-{twin_uuid[:8]}'
running = False
for _ in range(30):
    ps = subprocess.run(
        ['docker', 'ps', '--format', '{{.Names}}'],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    names = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if container_name in names:
        running = True
        break
    time.sleep(1)
assert running, f'Driver container {container_name} is not running'

startup_log_seen = False
for _ in range(45):
    logs = subprocess.run(
        ['docker', 'logs', container_name],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    message = logs.stdout + '\\n' + logs.stderr
    if (
        'Initializing camera depth-estimation driver' in message
        or 'Camera stream started' in message
    ):
        startup_log_seen = True
        break
    time.sleep(1)
assert startup_log_seen, (
    f'Did not observe expected startup log in container {container_name}. '
    'Last logs:\\n' + message[-4000:]
)

print('✅ Driver started via edge-core and connected to RGB emulator')
print(f'   twin={twin_uuid}')
print(f'   container={container_name}')
PY
"

echo ""
echo "=========================================="
echo " ✅ Camera depth-estimation E2E passed!"
echo "=========================================="
