import importlib.util
from pathlib import Path

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


def test_vision_engine_initializes_roboflow_client_without_sdk_config_error(monkeypatch):
    monkeypatch.setattr(module, "ROBOFLOW_API_KEY", "demo-key")
    monkeypatch.setattr(module, "ROBOFLOW_MODEL_ID", "demo-model")
    monkeypatch.setattr(module, "ROBOFLOW_WORKSPACE", "demo-workspace")
    monkeypatch.setattr(module, "ROBOFLOW_WORKFLOW_ID", "demo-workflow")

    engine = VisionEngine()

    assert engine.roboflow_workflow is True
    assert engine.roboflow_client is not None


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


def test_process_uses_people_count_when_chair_labels_understate_occupancy(monkeypatch):
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
    monkeypatch.setattr(engine, "detect_seats", lambda _frame, _people: (13, 0, 13))
    monkeypatch.setattr(VisionEngine, "_predict_detections", fake_predict)

    result = engine.process(frame)

    assert result.people_count == 2
    assert result.occupied_seats == 2
    assert result.empty_seats == 11
    assert result.occupancy_percentage == 15.4


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
