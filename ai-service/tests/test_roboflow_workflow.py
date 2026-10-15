import importlib.util
from pathlib import Path
from threading import Event
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


def test_engine_caches_jpeg_frames_for_live_preview():
    engine = VisionEngine()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    engine._cache_frame(frame)

    assert engine.latest_frame is not None
    assert engine.latest_frame.startswith(b"\xff\xd8")
    assert engine.last_frame_at is not None


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
