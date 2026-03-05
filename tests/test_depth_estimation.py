import base64

import numpy as np

from depth_estimation import DepthFramePublisher, depth_to_uint16


class _DummyEstimator:
    def __init__(self):
        self.calls = 0

    def estimate_depth(self, _frame):
        self.calls += 1
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class _DummyMQTTClient:
    def __init__(self):
        self.messages = []

    def publish_depth_frame(self, twin_uuid, depth_data, timestamp=None):
        self.messages.append(
            {
                "twin_uuid": twin_uuid,
                "depth_data": depth_data,
                "timestamp": timestamp,
            }
        )


def test_depth_to_uint16_normalized():
    depth = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    result = depth_to_uint16(depth, output_mode="normalized_uint16")

    assert result.dtype == np.uint16
    assert result.shape == depth.shape
    assert result.min() == 0
    assert result.max() == 65535


def test_depth_to_uint16_metric_mm():
    depth_meters = np.array([[0.0, 1.2], [65.0, 100.0]], dtype=np.float32)
    result = depth_to_uint16(depth_meters, output_mode="metric_mm", scale_factor=1000.0)

    assert result.dtype == np.uint16
    assert result[0, 0] == 0
    assert result[0, 1] == 1200
    assert result[1, 0] == 65000
    assert result[1, 1] == 65535  # clipped


def test_depth_frame_publisher_interval_and_payload():
    mqtt = _DummyMQTTClient()
    estimator = _DummyEstimator()
    publisher = DepthFramePublisher(
        mqtt_client=mqtt,
        twin_uuid="twin-123",
        estimator=estimator,
        publish_interval=2,
    )

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    publisher(frame, 0)  # publish
    publisher(frame, 1)  # skip
    publisher(frame, 2)  # publish

    assert estimator.calls == 2
    assert len(mqtt.messages) == 2

    depth_data = mqtt.messages[0]["depth_data"]
    assert depth_data["width"] == 2
    assert depth_data["height"] == 2
    assert depth_data["dtype"] == "uint16"

    decoded = base64.b64decode(depth_data["depth_binary"])
    decoded_depth = np.frombuffer(decoded, dtype=np.uint16).reshape(2, 2)
    assert decoded_depth.shape == (2, 2)
