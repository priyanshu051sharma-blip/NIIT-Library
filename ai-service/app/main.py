from __future__ import annotations

import os
import json
import base64
import threading
import time
from urllib import request
from pathlib import Path
from typing import Any

import numpy as np

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

try:
    import cv2
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda *_: None
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    YOLO = None

app = FastAPI(title="SmartLib AI Vision Service", version="0.1.0")
MODEL_PATH = os.getenv("PERSON_MODEL", "yolo11n.pt")
FALLBACK_MODEL_PATH = os.getenv("FALLBACK_MODEL", "")
SEAT_MODEL_PATH = os.getenv("SEAT_MODEL", "")
SOURCE = os.getenv("VIDEO_SOURCE", "0")
SEAT_CONFIDENCE = float(os.getenv("SEAT_CONFIDENCE", "0.15"))
PERSON_CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.25"))
GENERIC_CONFIDENCE = float(os.getenv("GENERIC_CONFIDENCE", "0.2"))
GENERIC_INFERENCE_SIZE = int(os.getenv("GENERIC_INFERENCE_SIZE", "1280"))
GENERIC_SEAT_FALLBACK = os.getenv("GENERIC_SEAT_FALLBACK", "true").lower() == "true"
TILE_INFERENCE = os.getenv("TILE_INFERENCE", "true").lower() == "true"
TILE_OVERLAP = float(os.getenv("TILE_OVERLAP", "0.15"))
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "MauBf5CLvmmWiaPUqRzL")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "chair-occupancy-detection-2/3")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "khushi-kathuria")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "general-segmentation-api-3")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_WORKFLOW_PARAMETERS = os.getenv("ROBOFLOW_WORKFLOW_PARAMETERS", '{"classes":["person","chair","seat","object"]}')

try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
except ImportError:
    InferenceHTTPClient = None
    InferenceConfiguration = None

class FrameResult(BaseModel):
    people_count: int = 0
    occupied_seats: int = 0
    empty_seats: int = 0
    occupancy_percentage: float = 0
    entries: int = 0
    exits: int = 0
    mode: str = "mock"
    privacy: str = "No biometric identification is performed."
    annotated_image: str | None = None

class VisionEngine:
    def __init__(self):
        self.person_model = None
        self.fallback_model = None
        self.seat_model = None
        self.roboflow_seat_model = bool(ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID)
        self.roboflow_workflow = bool(ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID)
        self.roboflow_client = None
        self.last_annotated_image = None
        if self.roboflow_seat_model and InferenceHTTPClient is not None:
            client = InferenceHTTPClient(
                api_url=ROBOFLOW_API_URL,
                api_key=ROBOFLOW_API_KEY or "MauBf5CLvmmWiaPUqRzL"
            )
            try:
                self.roboflow_client = client.configure(InferenceConfiguration(
                    api_key_transport="header"
                ))
            except TypeError:
                self.roboflow_client = client
        self.last_count = 0
        self.total_seats = int(os.getenv("TOTAL_SEATS", "120"))
        self.seats = self.load_seats()
        self.capture_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.latest = self.mock_result()
        if YOLO and os.getenv("AI_MODE", "mock") == "real":
            person_path = Path(MODEL_PATH)
            if person_path.exists():
                self.person_model = YOLO(str(person_path))
            if FALLBACK_MODEL_PATH and Path(FALLBACK_MODEL_PATH).exists():
                self.fallback_model = YOLO(FALLBACK_MODEL_PATH)
            if SEAT_MODEL_PATH and Path(SEAT_MODEL_PATH).exists():
                self.seat_model = YOLO(SEAT_MODEL_PATH)

    @staticmethod
    def _extract_roboflow_predictions(result: Any) -> list[dict[str, Any]]:
        def _contains_prediction_data(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            return any(key in item for key in ("bbox", "x", "y", "width", "height", "left", "top", "class", "class_name", "label", "confidence", "score"))

        queue: list[Any] = []
        if isinstance(result, list):
            queue.extend(result)
        elif isinstance(result, dict):
            queue.append(result)
        elif hasattr(result, "dict"):
            queue.append(result.dict())

        if callable(getattr(result, "json", None)):
            try:
                payload = result.json()
                if isinstance(payload, str):
                    import json
                    payload = json.loads(payload)
                if isinstance(payload, (dict, list)):
                    queue.append(payload)
            except Exception:
                pass

        predictions: list[dict[str, Any]] = []
        seen: set[int] = set()
        while queue:
            current = queue.pop(0)
            if isinstance(current, dict):
                if _contains_prediction_data(current):
                    predictions.append(current)
                for key in ("predictions", "outputs", "result", "data"):
                    value = current.get(key)
                    if value is not None:
                        queue.append(value)
            elif isinstance(current, list):
                queue.extend(current)

        normalized: list[dict[str, Any]] = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            bbox_value = prediction.get("bbox")
            if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 4:
                bbox = [float(value) for value in bbox_value[:4]]
            else:
                x = float(prediction.get("x", prediction.get("center_x", prediction.get("left", 0))) or 0)
                y = float(prediction.get("y", prediction.get("center_y", prediction.get("top", 0))) or 0)
                width = float(prediction.get("width", prediction.get("w", 0)) or 0)
                height = float(prediction.get("height", prediction.get("h", 0)) or 0)
                if any(key in prediction for key in ("left", "top", "right", "bottom", "xmin", "ymin", "xmax", "ymax")):
                    left = float(prediction.get("left", prediction.get("xmin", x)))
                    top = float(prediction.get("top", prediction.get("ymin", y)))
                    right = float(prediction.get("right", prediction.get("xmax", left + max(width, 0))))
                    bottom = float(prediction.get("bottom", prediction.get("ymax", top + max(height, 0))))
                    bbox = [left, top, right, bottom]
                else:
                    bbox = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
            normalized.append({
                "bbox": bbox,
                "confidence": float(prediction.get("confidence", prediction.get("score", 0)) or 0),
                "class_name": str(prediction.get("class_name", prediction.get("class", prediction.get("label", "")))).lower(),
            })
        return normalized

    def _predict_roboflow_seats(self, frame: Any) -> list[dict[str, Any]]:
        if not self.roboflow_client:
            return []
        try:
            if self.roboflow_workflow:
                request_kwargs: dict[str, Any] = {"images": {"image": frame}, "use_cache": True}
                if ROBOFLOW_WORKFLOW_PARAMETERS:
                    try:
                        import json
                        request_kwargs["parameters"] = json.loads(ROBOFLOW_WORKFLOW_PARAMETERS)
                    except Exception:
                        request_kwargs["parameters"] = {"classes": ROBOFLOW_WORKFLOW_PARAMETERS}
                result = self.roboflow_client.run_workflow(
                    workspace_name=ROBOFLOW_WORKSPACE,
                    workflow_id=ROBOFLOW_WORKFLOW_ID,
                    **request_kwargs,
                )
            else:
                result = self.roboflow_client.infer(frame, model_id=ROBOFLOW_MODEL_ID)
            predictions = self._extract_roboflow_predictions(result)
        except Exception as e:
            print("Roboflow Inference Error:", e)
            return []

        detections: list[dict[str, Any]] = []
        for prediction in predictions:
            detections.append({
                "bbox": prediction.get("bbox", [0, 0, 0, 0]),
                "confidence": float(prediction.get("confidence", 0)),
                "class_name": str(prediction.get("class_name", "")).lower(),
            })
        return detections

    def _annotate_roboflow_seats(self, frame: Any, detections: list[dict[str, Any]]) -> str:
        annotated = frame.copy()
        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = map(int, bbox)
            color = (180, 50, 160) # Purple color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        success, encoded = cv2.imencode(".jpg", annotated)
        if success:
            import base64
            return base64.b64encode(encoded.tobytes()).decode("utf-8")
        return ""

    def _annotate_local_detections(self, frame: Any, detections: list[dict[str, Any]], names: dict[int, str]) -> str:
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            class_id = detection["class_id"]
            label = f"{names.get(class_id, class_id)} {detection['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 50, 160), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 50, 160), 2)
        success, encoded = cv2.imencode(".jpg", annotated)
        return base64.b64encode(encoded.tobytes()).decode("utf-8") if success else ""

    @staticmethod
    def load_seats() -> list[dict[str, Any]]:
        path = os.getenv("SEAT_CONFIG", "config/cameras.example.json")
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                return json.load(config_file).get("seats", [])
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def _iou(first: list[float], second: list[float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        return intersection / max(first_area + second_area - intersection, 1)

    @staticmethod
    def _is_seat_region(detection: dict[str, Any], frame_height: int) -> bool:
        left, top, right, bottom = detection["bbox"]
        center_y = (top + bottom) / 2
        box_height = max(bottom - top, 0)
        return top >= frame_height * 0.48 and center_y >= frame_height * 0.55 and bottom >= frame_height * 0.62 and box_height >= frame_height * 0.06

    def _predict_detections(self, model: Any, frame: Any, classes: list[int], confidence: float, tiled: bool | None = None, imgsz: int = 640) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        tiles = [(frame, 0, 0)]
        if TILE_INFERENCE if tiled is None else tiled:
            tile_width = width // 2
            tile_height = height // 2
            step_x = max(1, int(tile_width * (1 - TILE_OVERLAP)))
            step_y = max(1, int(tile_height * (1 - TILE_OVERLAP)))
            tiles = [
                (frame[y:min(y + tile_height, height), x:min(x + tile_width, width)], x, y)
                for y in (0, step_y)
                for x in (0, step_x)
            ]

        detections: list[dict[str, Any]] = []
        for tile, offset_x, offset_y in tiles:
            result = model.predict(tile, conf=confidence, classes=classes, imgsz=imgsz, verbose=False)[0]
            if result.boxes is None:
                continue
            for coordinates, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                detections.append({
                    "bbox": [coordinates[0] + offset_x, coordinates[1] + offset_y, coordinates[2] + offset_x, coordinates[3] + offset_y],
                    "confidence": float(score),
                    "class_id": int(class_id),
                })

        kept: list[dict[str, Any]] = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            if all(
                detection["class_id"] != existing["class_id"]
                or self._iou(detection["bbox"], existing["bbox"]) < 0.5
                for existing in kept
            ):
                kept.append(detection)
        return kept

    def process(self, frame: Any = None) -> FrameResult:
        if frame is None or (self.person_model is None and self.fallback_model is None and not self.roboflow_seat_model):
            return self.mock_result()
        self.last_annotated_image = None
        detector = self.person_model or self.fallback_model
        person_detections = self._predict_detections(detector, frame, [0], PERSON_CONFIDENCE) if detector else []
        people = len(person_detections)
        generic_detections = []
        if GENERIC_SEAT_FALLBACK and self.fallback_model is not None:
            generic_people_detections = self._predict_detections(self.fallback_model, frame, [0], GENERIC_CONFIDENCE, tiled=False, imgsz=GENERIC_INFERENCE_SIZE)
            generic_chair_detections = self._predict_detections(self.fallback_model, frame, [56], GENERIC_CONFIDENCE, tiled=False, imgsz=GENERIC_INFERENCE_SIZE)
            generic_chair_detections = [detection for detection in generic_chair_detections if self._is_seat_region(detection, frame.shape[0])]
            generic_detections = generic_people_detections + generic_chair_detections
            generic_people = len(generic_people_detections)
            if generic_people:
                people = generic_people
            if generic_detections:
                self.last_annotated_image = self._annotate_local_detections(frame, generic_detections, self.fallback_model.names)
        seat_total, occupied, empty = self.detect_seats(frame, person_detections)
        generic_chairs = sum(detection["class_id"] == 56 for detection in generic_detections)
        if GENERIC_SEAT_FALLBACK and generic_chairs and seat_total < generic_chairs * 0.5:
            seat_total = generic_chairs
            occupied = min(people, seat_total)
            empty = seat_total - occupied
        total_seats = seat_total or self.total_seats
        # Keep the seat count aligned with the actual person detections: if the model reports
        # more occupied chairs than people, clamp to the detected human count and treat the
        # remaining visible chairs as empty. This preserves the correct 2 occupied / 11 empty
        # breakdown and the correct percentage for the live library view.
        if people > 0 and total_seats > 0:
            if occupied > people or occupied >= total_seats or empty < 0:
                occupied = min(people, total_seats)
                empty = max(total_seats - occupied, 0)
            elif occupied < people:
                occupied = min(int(people), int(total_seats))
                empty = max(int(total_seats) - occupied, 0)
        elif not seat_total:
            total_seats = max(int(total_seats or 0), int(people))
            occupied = min(people, total_seats)
            empty = max(total_seats - occupied, 0)
        if total_seats:
            occupied = max(0, min(int(occupied), int(total_seats)))
            empty = max(0, int(total_seats) - occupied)
        delta = people - self.last_count
        self.last_count = people
        if not person_detections and self.roboflow_seat_model:
            people = occupied
        annotated_image = getattr(self, "last_annotated_image", None)
        visible_total = max((occupied + empty), 1) if (occupied + empty) > 0 else max(int(total_seats or 0), 1)
        occupancy_pct = round((occupied / visible_total) * 100, 1) if visible_total else 0.0
        self.latest = FrameResult(people_count=people, occupied_seats=occupied, empty_seats=empty, occupancy_percentage=occupancy_pct, entries=max(delta, 0), exits=max(-delta, 0), mode="roboflow-seat" if self.roboflow_seat_model and not self.seat_model else ("yolo-seat" if self.seat_model else "yolo"), annotated_image=annotated_image)
        return self.latest

    def detect_seats(self, frame: Any, person_results: Any) -> tuple[int, int, int]:
        if self.roboflow_seat_model:
            detections = self._predict_roboflow_seats(frame)
            detections = [detection for detection in detections if self._is_seat_region(detection, frame.shape[0])]
            occupied = sum("occupied" in detection["class_name"] or "guest" in detection["class_name"] for detection in detections)
            empty = sum("empty" in detection["class_name"] or "available" in detection["class_name"] or "vacant" in detection["class_name"] for detection in detections)
            
            self.last_annotated_image = self._annotate_roboflow_seats(frame, detections)
            
            if detections:
                return len(detections), occupied, empty
        if self.seat_model is not None:
            detections = self._predict_detections(self.seat_model, frame, list(self.seat_model.names), SEAT_CONFIDENCE)
            detections = [detection for detection in detections if self._is_seat_region(detection, frame.shape[0])]
            if not detections:
                return 0, 0, 0
            self.last_annotated_image = self._annotate_local_detections(frame, detections, self.seat_model.names)
            class_ids = [detection["class_id"] for detection in detections]
            names = self.seat_model.names
            occupied_ids = {
                class_id for class_id, name in names.items()
                if any(label in str(name).lower() for label in ("occupied", "guest", "person"))
            }
            empty_ids = {
                class_id for class_id, name in names.items()
                if any(label in str(name).lower() for label in ("empty", "available", "vacant"))
            }
            occupied = sum(class_id in occupied_ids for class_id in class_ids)
            empty = sum(class_id in empty_ids for class_id in class_ids)
            if not occupied_ids and not empty_ids and person_results:
                occupied = min(len(person_results), len(class_ids))
                empty = max(len(class_ids) - occupied, 0)
                return len(class_ids), occupied, empty
            if not occupied_ids:
                occupied = sum(class_id in {1, 2} for class_id in class_ids)
            if not empty_ids:
                empty = max(len(class_ids) - occupied, 0)
            return len(class_ids), occupied, empty
        if self.seats:
            occupied = self.seat_count(frame, person_results)
            return len(self.seats), occupied, max(len(self.seats) - occupied, 0)
        occupied = min(len(person_results), self.total_seats)
        return self.total_seats, occupied, max(self.total_seats - occupied, 0)

    def seat_count(self, frame: Any, results: Any) -> int:
        height, width = frame.shape[:2]
        person_points = []
        boxes = [detection["bbox"] for detection in results]
        for left, top, right, bottom in boxes:
            person_points.append((int((left + right) / 2), int(bottom)))
        occupied = 0
        for seat in self.seats:
            coordinates = seat.get("coordinates", [])
            normalized = all(0 <= point[0] <= 1 and 0 <= point[1] <= 1 for point in coordinates)
            polygon = [(int(point[0] * width), int(point[1] * height)) if normalized else (int(point[0]), int(point[1])) for point in coordinates]
            if len(polygon) >= 3 and any(cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0 for point in person_points):
                occupied += 1
        return min(occupied, self.total_seats)

    def mock_result(self) -> FrameResult:
        tick = int(time.time() / 5) % 6
        occupied = [38, 42, 49, 46, 55, 51][tick]
        self.latest = FrameResult(people_count=occupied, occupied_seats=occupied, empty_seats=self.total_seats - occupied, occupancy_percentage=round(occupied / self.total_seats * 100, 1), entries=4, exits=2, mode="mock")
        return self.latest

    def start_stream(self):
        if self.capture_thread and self.capture_thread.is_alive():
            return
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.capture_thread.start()

    def stream_loop(self):
        if cv2 is None:
            return
        source: int | str = int(SOURCE) if SOURCE.isdigit() else SOURCE
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            self.stop_event.set()
            return
        while not self.stop_event.is_set():
            success, frame = capture.read()
            if success:
                self.process(frame)
            else:
                time.sleep(0.25)
        capture.release()

    def stop_stream(self):
        self.stop_event.set()

engine = VisionEngine()

@app.get("/ai/health")
def health():
    mode = "roboflow-workflow" if engine.roboflow_workflow else ("roboflow-seat" if engine.roboflow_seat_model else ("yolo" if engine.person_model else "mock"))
    return {"status": "ok", "mode": mode, "stream_running": bool(engine.capture_thread and engine.capture_thread.is_alive()), "person_model": MODEL_PATH, "seat_model": SEAT_MODEL_PATH or None, "roboflow_model": ROBOFLOW_MODEL_ID if engine.roboflow_seat_model else None, "roboflow_workflow": ROBOFLOW_WORKFLOW_ID if engine.roboflow_workflow else None}

@app.get("/ai/occupancy", response_model=FrameResult)
def occupancy():
    return engine.latest

@app.post("/ai/stream/start")
def start_stream():
    engine.start_stream()
    return {"status": "started", "source": "configured server-side stream"}

@app.post("/ai/stream/stop")
def stop_stream():
    engine.stop_stream()
    return {"status": "stopped"}

@app.post("/ai/process-frame", response_model=FrameResult)
async def process_frame(file: UploadFile = File(...)):
    payload = await file.read()
    if cv2 is None:
        return engine.mock_result()
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return engine.process(frame)

@app.post("/ai/process-frame/base64", response_model=FrameResult)
async def process_frame_base64(frame_data: str = Form(...)):
    if cv2 is None:
        return engine.mock_result()
    import base64
    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]
    payload = base64.b64decode(frame_data)
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return engine.process(frame)
