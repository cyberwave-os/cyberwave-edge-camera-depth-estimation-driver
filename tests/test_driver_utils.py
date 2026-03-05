from driver_utils import parse_bool, parse_camera_id, parse_float, parse_int


def test_parse_camera_id():
    assert parse_camera_id("0") == 0
    assert parse_camera_id("2") == 2
    assert parse_camera_id("/dev/video2") == "/dev/video2"
    assert parse_camera_id("rtsp://camera/stream") == "rtsp://camera/stream"


def test_parse_bool():
    assert parse_bool("true") is True
    assert parse_bool("1") is True
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool("unknown", default=True) is True


def test_parse_int_clamp():
    assert parse_int("20", default=5) == 20
    assert parse_int("nope", default=5) == 5
    assert parse_int("1", default=5, minimum=2) == 2
    assert parse_int("200", default=5, maximum=100) == 100


def test_parse_float_clamp():
    assert parse_float("20.5", default=5.0) == 20.5
    assert parse_float("nope", default=5.0) == 5.0
    assert parse_float("0.1", default=5.0, minimum=0.5) == 0.5
    assert parse_float("20.0", default=5.0, maximum=10.0) == 10.0
