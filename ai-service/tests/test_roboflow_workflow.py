import importlib.util
from pathlib import Path
from threading import Condition, Event
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np

module_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
spec = importlib.util.spec_from_file_location("ai_app_main", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


VisionEngine = module.VisionEngine


def test_extracts_predictions_from_workflow_output():
    payload = {
        "outputs": [
            {
                "predictions": [
                    {
                        "x": 110,
                        "y": 120,
                        "width": 80,
                        "height": 60,
                        "confidence": 0.86,
                        "class_name": "occupied chair",
                    }
                ]
            }
        ]
    }

    predictions = VisionEngine._extract_roboflow_predictions(payload)

    assert len(predictions) == 1
    assert predictions[0]["class_name"] == "occupied chair"
    assert predictions[0]["bbox"] == [70.0, 90.0, 150.0, 150.0]


def test_vision_engine_does_not_create_roboflow_client_without_sdk(monkeypatch):
    monkeypatch.setattr(module, "InferenceHTTPClient", None)
    monkeypatch.setattr(module, "InferenceConfiguration", None)
    monkeypatch.setattr(module, "ROBOFLOW_API_KEY", "demo-key")
    monkeypatch.setattr(module, "ROBOFLOW_MODEL_ID", "demo-model")
    monkeypatch.setattr(module, "ROBOFLOW_WORKSPACE", "demo-workspace")
    monkeypatch.setattr(module, "ROBOFLOW_WORKFLOW_ID", "demo-workflow")

    engine = VisionEngine()

    assert engine.roboflow_workflow is True
    assert engine.roboflow_client is None


def test_configured_video_source_builds_rtsp_url_with_escaped_credentials(monkeypatch):
    monkeypatch.delenv("CAMERA_STREAM_URL", raising=False)
    monkeypatch.delenv("RTSP_URL", raising=False)
    monkeypatch.setenv("VIDEO_SOURCE", "0")
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.10")
    monkeypatch.setenv("CAMERA_PORT", "8554")
    monkeypatch.setenv("CAMERA_PROTOCOL", "rtsp")
    monkeypatch.setenv("CAMERA_STREAM_PATH", "/live/main")
    monkeypatch.setenv("CAMERA_USERNAME", "library user")
    monkeypatch.setenv("CAMERA_PASSWORD", "pass/word")

    source = module.configured_video_source()

    assert source == "rtsp://library%20user:pass%2Fword@192.0.2.10:8554/live/main"
    assert urlsplit(source).password == "pass%2Fword"


def test_configured_video_source_uses_http_mjpeg_camera_defaults(monkeypatch):
    monkeypatch.delenv("CAMERA_STREAM_URL", raising=False)
    monkeypatch.delenv("RTSP_URL", raising=False)
    monkeypatch.setenv("VIDEO_SOURCE", "0")
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.10")
    monkeypatch.delenv("CAMERA_PROTOCOL", raising=False)
    monkeypatch.delenv("CAMERA_PORT", raising=False)
    monkeypatch.delenv("CAMERA_STREAM_PATH", raising=False)
    monkeypatch.setenv("CAMERA_USERNAME", "")
    monkeypatch.setenv("CAMERA_PASSWORD", "")

    assert module.configured_video_source() == "http://192.0.2.10:80/cgi-bin/mjpg/video.cgi"


def test_parse_camera_channels_defaults_to_seven_distinct_channels():
    assert module.parse_camera_channels("1,2,3,4,5,6,7") == (1, 2, 3, 4, 5, 6, 7)
    assert module.parse_camera_channels("2, 2, 7") == (2, 7)


def test_parse_camera_channels_rejects_invalid_channel_numbers():
    import pytest

    with pytest.raises(ValueError, match="positive channel"):
        module.parse_camera_channels("1,0,2")
    with pytest.raises(ValueError, match="positive integers"):
        module.parse_camera_channels("1,two")


def test_configured_camera_sources_preserve_rtsp_settings_and_separate_channels():
    source = "rtsp://user:password@192.0.2.10:554/cam/realmonitor?channel=2&subtype=0"

    cameras = module.configured_camera_sources(source, (1, 2, 7))

    assert [camera["name"] for camera in cameras] == ["Library-1", "Library-2", "Cabin-2"]
    assert [camera["channel"] for camera in cameras] == [1, 2, 7]
    assert [urlsplit(camera["source"]).query for camera in cameras] == [
        "channel=1&subtype=0",
        "channel=2&subtype=0",
        "channel=7&subtype=0",
    ]
    assert all(urlsplit(camera["source"]).password == "password" for camera in cameras)


def test_camera_frame_endpoint_returns_the_latest_annotated_jpeg(monkeypatch):
    frame = b"\xff\xd8annotated-camera-frame\xff\xd9"
    feed = SimpleNamespace(condition=Condition(), latest_annotated_frame=frame, latest_source_frame=b"raw-frame")
    monkeypatch.setattr(module, "camera_monitor", SimpleNamespace(get_feed=lambda _camera_id: feed))

    response = module.camera_frame(camera_id="cam-2", annotated=True)

    assert response.media_type == "image/jpeg"
    assert response.body == frame
    assert response.headers["Cache-Control"] == "no-cache, no-store"


def test_camera_frame_endpoint_reports_when_no_frame_is_available(monkeypatch):
    import pytest
    from fastapi import HTTPException

    feed = SimpleNamespace(condition=Condition(), latest_annotated_frame=None, latest_source_frame=None)
    monkeypatch.setattr(module, "camera_monitor", SimpleNamespace(get_feed=lambda _camera_id: feed))

    with pytest.raises(HTTPException, match="not produced a frame yet"):
        module.camera_frame(camera_id="cam-2", annotated=True)


def test_http_camera_credentials_are_not_added_to_stream_url(monkeypatch):
    monkeypatch.delenv("CAMERA_STREAM_URL", raising=False)
    monkeypatch.delenv("RTSP_URL", raising=False)
    monkeypatch.setenv("VIDEO_SOURCE", "0")
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.10")
    monkeypatch.setenv("CAMERA_USERNAME", "camera user")
    monkeypatch.setenv("CAMERA_PASSWORD", "camera password")

    source = module.configured_video_source()

    assert source == "http://192.0.2.10:80/cgi-bin/mjpg/video.cgi"
    assert "camera" not in source


def test_iter_mjpeg_frames_extracts_frames_across_chunk_boundaries():
    first_frame = b"\xff\xd8first-frame\xff\xd9"
    second_frame = b"\xff\xd8second-frame\xff\xd9"
    payload = b"--boundary\r\nContent-Type: image/jpeg\r\n\r\n" + first_frame + b"\r\n--boundary\r\nContent-Type: image/jpeg\r\n\r\n" + second_frame

    class ChunkedStream:
        def __init__(self, data):
            self.data = data

        def read(self, size):
            chunk, self.data = self.data[:7], self.data[7:]
            return chunk

    frames = list(VisionEngine.iter_mjpeg_frames(ChunkedStream(payload), Event()))

    assert frames == [first_frame, second_frame]


def test_iter_mjpeg_frames_uses_available_http_chunks_without_waiting_for_full_buffer():
    frame = b"\xff\xd8camera-frame\xff\xd9"
    payload = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame

    class BufferedStream:
        def __init__(self, data):
            self.data = data

        def read1(self, size):
            chunk, self.data = self.data[:11], self.data[11:]
            return chunk

        def read(self, size):
            raise AssertionError("read() can block waiting to fill the requested buffer")

    frames = list(VisionEngine.iter_mjpeg_frames(BufferedStream(payload), Event()))

    assert frames == [frame]


def test_engine_caches_jpeg_frames_for_live_preview():
    engine = VisionEngine()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    engine._cache_frame(frame)

    assert engine.latest_frame is not None
    assert engine.latest_frame.startswith(b"\xff\xd8")
    assert engine.last_frame_at is not None


def test_engine_reuses_same_model_for_person_and_generic_fallback_paths(tmp_path, monkeypatch):
    model_path = tmp_path / "yolo11n.pt"
    model_path.touch()
    model = SimpleNamespace(names={0: "person", 56: "chair"})
    loaded_paths = []
    monkeypatch.setattr(module, "YOLO", lambda path: loaded_paths.append(path) or model)
    monkeypatch.setattr(module, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(module, "FALLBACK_MODEL_PATH", str(model_path))
    monkeypatch.setattr(module, "SEAT_MODEL_PATH", "")
    monkeypatch.setattr(module, "ROBOFLOW_API_KEY", "")
    monkeypatch.setenv("AI_MODE", "real")

    engine = VisionEngine()

    assert engine.person_model is model
    assert engine.fallback_model is model
    assert loaded_paths == [str(model_path)]


def test_process_falls_back_to_detected_people_when_chair_state_is_missing(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def fake_predict(self, model, frame, classes, confidence, tiled=None, imgsz=640):
        return [
            {"bbox": [10, 10, 30, 30], "confidence": 0.9, "class_id": 0},
            {"bbox": [40, 10, 60, 30], "confidence": 0.85, "class_id": 0},
        ]

    monkeypatch.setattr(engine, "person_model", object())
    monkeypatch.setattr(engine, "fallback_model", None)
    monkeypatch.setattr(engine, "roboflow_seat_model", False)
    monkeypatch.setattr(engine, "detect_seats", lambda _frame, _people: (4, 0, 0))
    monkeypatch.setattr(VisionEngine, "_predict_detections", fake_predict)

    result = engine.process(frame)

    assert result.people_count == 2
    assert result.occupied_seats == 2
    assert result.empty_seats == 2
    assert result.occupancy_percentage == 50.0


def test_process_uses_detected_people_when_total_seat_count_is_zero(monkeypatch):
    engine = VisionEngine()
    engine.total_seats = 0
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def fake_predict(self, model, frame, classes, confidence, tiled=None, imgsz=640):
        return [
            {"bbox": [10, 10, 30, 30], "confidence": 0.9, "class_id": 0},
            {"bbox": [40, 10, 60, 30], "confidence": 0.85, "class_id": 0},
        ]

    monkeypatch.setattr(engine, "person_model", object())
    monkeypatch.setattr(engine, "fallback_model", None)
    monkeypatch.setattr(engine, "roboflow_seat_model", False)
    monkeypatch.setattr(engine, "detect_seats", lambda _frame, _people: (0, 0, 0))
    monkeypatch.setattr(VisionEngine, "_predict_detections", fake_predict)

    result = engine.process(frame)

    assert result.people_count == 2
    assert result.occupied_seats == 2
    assert result.empty_seats == 0
    assert result.occupancy_percentage == 100.0


def test_process_keeps_valid_chair_state_even_when_people_count_is_higher(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def fake_predict(self, model, frame, classes, confidence, tiled=None, imgsz=640):
        return [
            {"bbox": [10, 10, 30, 30], "confidence": 0.9, "class_id": 0},
            {"bbox": [40, 10, 60, 30], "confidence": 0.85, "class_id": 0},
            {"bbox": [70, 10, 90, 30], "confidence": 0.8, "class_id": 0},
            {"bbox": [100, 10, 120, 30], "confidence": 0.75, "class_id": 0},
            {"bbox": [130, 10, 150, 30], "confidence": 0.7, "class_id": 0},
        ]

    monkeypatch.setattr(engine, "person_model", object())
    monkeypatch.setattr(engine, "fallback_model", None)
    monkeypatch.setattr(engine, "roboflow_seat_model", False)
    monkeypatch.setattr(engine, "detect_seats", lambda _frame, _people: (13, 2, 11))
    monkeypatch.setattr(VisionEngine, "_predict_detections", fake_predict)

    result = engine.process(frame)

    assert result.people_count == 5
    assert result.occupied_seats == 2
    assert result.empty_seats == 11
    assert result.occupancy_percentage == 15.4


def test_process_runs_generic_fallback_when_specialized_seat_detection_succeeds(monkeypatch):
    engine = VisionEngine()
    person_model = object()
    fallback_model = SimpleNamespace(names={0: "person", 56: "chair"})
    engine.person_model = person_model
    engine.fallback_model = fallback_model
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    predictions = []

    def fake_predict(self, model, frame, classes, confidence, tiled=None, imgsz=640):
        predictions.append((model, classes))
        return []

    def detected_seats(_frame, _people):
        engine.last_annotated_image = "c2VhdCBhbm5vdGF0aW9ucw=="
        return 4, 1, 3

    monkeypatch.setattr(engine, "detect_seats", detected_seats)
    monkeypatch.setattr(VisionEngine, "_predict_detections", fake_predict)

    result = engine.process(frame)

    assert result.occupied_seats == 1
    assert predictions == [
        (fallback_model, [0, 56]),
        (person_model, [0]),
    ]


def test_detect_seats_prefers_local_seat_model_over_roboflow(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    engine.seat_model = SimpleNamespace(names={1: "occupied chair", 2: "empty chair"})
    engine.roboflow_seat_model = True

    monkeypatch.setattr(engine, "_predict_detections", lambda *_args, **_kwargs: [{"bbox": [10, 10, 30, 30], "confidence": 0.9, "class_id": 1}, {"bbox": [40, 10, 60, 30], "confidence": 0.8, "class_id": 2}])
    monkeypatch.setattr(engine, "_is_seat_region", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(engine, "_annotate_local_detections", lambda *_args, **_kwargs: "")

    result = engine.detect_seats(frame, [])

    assert result == (2, 1, 1)


def test_detect_seats_keeps_trained_seat_detections_in_the_back_of_the_frame(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    engine.seat_model = SimpleNamespace(names={1: "occupied chair", 2: "empty chair"})

    monkeypatch.setattr(engine, "_predict_detections", lambda *_args, **_kwargs: [
        {"bbox": [10, 10, 30, 30], "confidence": 0.9, "class_id": 1},
        {"bbox": [40, 25, 60, 45], "confidence": 0.8, "class_id": 2},
    ])
    monkeypatch.setattr(engine, "_annotate_local_detections", lambda *_args, **_kwargs: "")

    assert engine.detect_seats(frame, []) == (2, 1, 1)


def test_detect_seats_uses_camera_roi_to_exclude_detections_outside_the_seating_area(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    engine.seat_model = SimpleNamespace(names={1: "occupied chair", 2: "empty chair"})
    detections = [
        {"bbox": [10, 80, 30, 120], "confidence": 0.9, "class_id": 1},
        {"bbox": [160, 80, 180, 120], "confidence": 0.8, "class_id": 2},
    ]

    monkeypatch.setattr(engine, "_predict_detections", lambda *_args, **_kwargs: detections)
    monkeypatch.setattr(engine, "_annotate_local_detections", lambda *_args, **_kwargs: "")

    assert engine.detect_seats(
        frame,
        [],
        seat_roi=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
    ) == (1, 1, 0)


def test_process_does_not_assume_default_seat_count_when_calibrated_roi_has_no_seats(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    engine.person_model = object()
    engine.seat_model = SimpleNamespace(names={1: "occupied chair", 2: "empty chair"})
    monkeypatch.setattr(module, "GENERIC_SEAT_FALLBACK", False)
    monkeypatch.setattr(engine, "_predict_detections", lambda *_args, **_kwargs: [])

    result = engine.process(
        frame,
        seat_roi=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
    )

    assert result.people_count == 0
    assert result.occupied_seats == 0
    assert result.empty_seats == 0


def test_process_uses_person_fallback_and_roi_filtered_generic_chairs_alongside_seat_model(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    person_model = object()
    fallback_model = SimpleNamespace(names={0: "person", 56: "chair"})
    seat_model = object()
    engine.person_model = person_model
    engine.fallback_model = fallback_model
    engine.seat_model = seat_model
    monkeypatch.setattr(module, "GENERIC_SEAT_FALLBACK", True)

    def predict(model, _frame, classes, *_args, **_kwargs):
        if model is person_model:
            return [{"bbox": [10, 20, 30, 70], "confidence": 0.8, "class_id": 0}]
        if model is seat_model:
            return []
        return [
            {"bbox": [120, 20, 140, 70], "confidence": 0.75, "class_id": 0},
            {"bbox": [10, 10, 25, 35], "confidence": 0.8, "class_id": 56},
            {"bbox": [35, 10, 50, 35], "confidence": 0.8, "class_id": 56},
            {"bbox": [60, 10, 75, 35], "confidence": 0.8, "class_id": 56},
            {"bbox": [80, 10, 95, 35], "confidence": 0.8, "class_id": 56},
            {"bbox": [150, 10, 175, 35], "confidence": 0.9, "class_id": 56},
        ]

    def detect_seats(_frame, _people, _seat_roi=None):
        engine.last_annotated_image = ""
        return 2, 1, 1

    monkeypatch.setattr(engine, "_predict_detections", predict)
    monkeypatch.setattr(engine, "detect_seats", detect_seats)
    monkeypatch.setattr(engine, "_annotate_local_detections", lambda *_args, **_kwargs: "")

    result = engine.process(
        frame,
        seat_roi=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
    )

    assert result.people_count == 2
    assert result.occupied_seats == 1
    assert result.empty_seats == 3


def test_camera_calibration_persists_per_camera_and_can_be_cleared(tmp_path):
    cameras = [{"id": "cam-test", "name": "Test camera", "channel": 1, "source": "rtsp://camera"}]
    calibration_path = tmp_path / "data" / "seat-areas.json"
    monitor = module.CameraMonitor(VisionEngine(), cameras, calibration_path)
    points = [(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)]

    monitor.set_calibration(monitor.get_feed("cam-test"), points)
    restored = module.CameraMonitor(VisionEngine(), cameras, calibration_path)

    assert restored.get_feed("cam-test").seat_roi == points
    restored.set_calibration(restored.get_feed("cam-test"), None)
    cleared = module.CameraMonitor(VisionEngine(), cameras, calibration_path)
    assert cleared.get_feed("cam-test").seat_roi is None


def test_camera_roi_api_allows_browser_put_and_delete_preflight():
    from fastapi.testclient import TestClient

    client = TestClient(module.app)
    headers = {
        "Origin": "http://localhost:3001",
        "Access-Control-Request-Method": "PUT",
        "Access-Control-Request-Headers": "content-type",
    }

    response = client.options("/ai/cameras/cam-4/roi", headers=headers)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"
    assert "PUT" in response.headers["access-control-allow-methods"]
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_process_clamps_occupied_count_to_detected_people_and_keeps_remaining_chairs_empty(monkeypatch):
    engine = VisionEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def fake_predict(self, model, frame, classes, confidence, tiled=None, imgsz=640):
        return [
            {"bbox": [10, 10, 30, 30], "confidence": 0.9, "class_id": 0},
            {"bbox": [40, 10, 60, 30], "confidence": 0.85, "class_id": 0},
        ]

    monkeypatch.setattr(engine, "person_model", object())
    monkeypatch.setattr(engine, "fallback_model", None)
    monkeypatch.setattr(engine, "roboflow_seat_model", False)
    monkeypatch.setattr(engine, "detect_seats", lambda _frame, _people: (13, 13, 0))
    monkeypatch.setattr(VisionEngine, "_predict_detections", fake_predict)

    result = engine.process(frame)

    assert result.people_count == 2
    assert result.occupied_seats == 2
    assert result.empty_seats == 11
    assert result.occupancy_percentage == 15.4
