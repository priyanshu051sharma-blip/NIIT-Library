from __future__ import annotations

import os
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    YOLO = None

app = FastAPI(title="SmartLib AI Vision Service", version="0.1.0")
MODEL_PATH = os.getenv("PERSON_MODEL", "yolo11n.pt")
FALLBACK_MODEL_PATH = os.getenv("FALLBACK_MODEL", "")
SEAT_MODEL_PATH = os.getenv("SEAT_MODEL", "")
SOURCE = os.getenv("VIDEO_SOURCE", "0")
SEAT_CONFIDENCE = float(os.getenv("SEAT_CONFIDENCE", "0.35"))

class FrameResult(BaseModel):
    people_count: int = 0
    occupied_seats: int = 0
    empty_seats: int = 0
    occupancy_percentage: float = 0
    entries: int = 0
    exits: int = 0
    mode: str = "mock"
    privacy: str = "No biometric identification is performed."

class VisionEngine:
    def __init__(self):
        self.person_model = None
        self.fallback_model = None
        self.seat_model = None
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
    def load_seats() -> list[dict[str, Any]]:
        path = os.getenv("SEAT_CONFIG", "config/cameras.example.json")
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                return json.load(config_file).get("seats", [])
        except (OSError, json.JSONDecodeError):
            return []

    def process(self, frame: Any = None) -> FrameResult:
        if frame is None or (self.person_model is None and self.fallback_model is None):
            return self.mock_result()
        detector = self.person_model or self.fallback_model
        results = detector.track(frame, persist=True, classes=[0], verbose=False)
        people = len(results[0].boxes) if results and results[0].boxes is not None else 0
        seat_total, occupied, empty = self.detect_seats(frame, results)
        total_seats = seat_total or self.total_seats
        if not seat_total:
            occupied = min(people, total_seats)
            empty = max(total_seats - occupied, 0)
        delta = people - self.last_count
        self.last_count = people
        self.latest = FrameResult(people_count=people, occupied_seats=occupied, empty_seats=empty, occupancy_percentage=round(occupied / total_seats * 100, 1), entries=max(delta, 0), exits=max(-delta, 0), mode="yolo-seat" if self.seat_model else "yolo")
        return self.latest

    def detect_seats(self, frame: Any, person_results: Any) -> tuple[int, int, int]:
        if self.seat_model is not None:
            results = self.seat_model.predict(frame, conf=SEAT_CONFIDENCE, verbose=False)
            boxes = results[0].boxes
            if boxes is None or boxes.cls is None:
                return 0, 0, 0
            class_ids = [int(class_id) for class_id in boxes.cls.cpu().tolist()]
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
            if not occupied_ids:
                occupied = sum(class_id in {1, 2} for class_id in class_ids)
            if not empty_ids:
                empty = max(len(class_ids) - occupied, 0)
            return len(class_ids), occupied, empty
        if self.seats:
            occupied = self.seat_count(frame, person_results)
            return len(self.seats), occupied, max(len(self.seats) - occupied, 0)
        occupied = min(len(person_results[0].boxes) if person_results and person_results[0].boxes is not None else 0, self.total_seats)
        return self.total_seats, occupied, max(self.total_seats - occupied, 0)

    def seat_count(self, frame: Any, results: Any) -> int:
        height, width = frame.shape[:2]
        person_points = []
        boxes = results[0].boxes.xyxy.cpu().tolist() if results and results[0].boxes is not None else []
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
    return {"status": "ok", "mode": "yolo" if engine.person_model else "mock", "stream_running": bool(engine.capture_thread and engine.capture_thread.is_alive()), "person_model": MODEL_PATH, "seat_model": SEAT_MODEL_PATH or None}

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
