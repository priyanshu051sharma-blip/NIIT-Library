from __future__ import annotations

import os
import json
import base64
import math
import threading
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any
from datetime import datetime, timezone
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, Request, build_opener

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import StreamingResponse

try:
    import cv2
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda *_: None
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    YOLO = None

app = FastAPI(title="SmartLib AI Vision Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3001").split(","),
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
MODEL_PATH = os.getenv("PERSON_MODEL", "yolo11n.pt")
FALLBACK_MODEL_PATH = os.getenv("FALLBACK_MODEL", "")
SEAT_MODEL_PATH = os.getenv("SEAT_MODEL", "")


def configured_video_source() -> int | str:
    video_source = os.getenv("VIDEO_SOURCE", "").strip()
    rtsp_url = os.getenv("RTSP_URL", "").strip()
    source = os.getenv("CAMERA_STREAM_URL", "").strip()
    if not source:
        camera_host = os.getenv("CAMERA_HOST", "").strip()
        if camera_host:
            protocol = os.getenv("CAMERA_PROTOCOL", "http").strip().lower()
            port = os.getenv("CAMERA_PORT", "80").strip()
            path = os.getenv("CAMERA_STREAM_PATH", "/cgi-bin/mjpg/video.cgi").strip()
            if path and not path.startswith(("/", "?")):
                path = f"/{path}"
            host = f"[{camera_host}]" if ":" in camera_host and not camera_host.startswith("[") else camera_host
            source = f"{protocol}://{host}:{port}{path}"

    if not source:
        source = video_source if video_source and video_source != "0" else rtsp_url or video_source or "0"

    parsed = urlsplit(source)
    username = os.getenv("CAMERA_USERNAME", "")
    password = os.getenv("CAMERA_PASSWORD", "")
    if parsed.scheme.lower() in {"http", "https"} and parsed.hostname:
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        source = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    elif parsed.scheme and parsed.hostname and (username or password) and parsed.username is None:
        credentials = f"{quote(username, safe='')}:{quote(password, safe='')}@"
        source = urlunsplit((parsed.scheme, credentials + parsed.netloc, parsed.path, parsed.query, parsed.fragment))

    return int(source) if source.isdigit() else source


SOURCE = configured_video_source()
CAMERA_NAMES = {
    1: "Library-1",
    2: "Library-2",
    3: "Library-3",
    4: "Library-4",
    5: "Library Main",
    6: "Cabin-1",
    7: "Cabin-2",
}
CAMERA_PURPOSES = {
    "cam-1": "library",
    "cam-2": "library",
    "cam-3": "entrance",
    "cam-4": "library",
    "cam-5": "entrance",
    "cam-6": "discussion",
    "cam-7": "discussion",
}
CAMERA_GROUPS = {"library", "discussion", "entrance"}


def parse_camera_channels(value: str) -> tuple[int, ...]:
    try:
        channels = tuple(dict.fromkeys(int(channel.strip()) for channel in value.split(",") if channel.strip()))
    except ValueError as error:
        raise ValueError("CAMERA_CHANNELS must be a comma-separated list of positive integers.") from error
    if not channels or any(channel < 1 for channel in channels):
        raise ValueError("CAMERA_CHANNELS must contain at least one positive channel number.")
    return channels


def configured_camera_sources(source: int | str, channels: tuple[int, ...]) -> list[dict[str, Any]]:
    if not isinstance(source, str) or source == "0":
        return []

    parsed = urlsplit(source)
    if parsed.scheme.lower() != "rtsp":
        return [{"id": "camera-1", "name": "Camera 1", "channel": None, "source": source}]

    sources = []
    for channel in channels:
        query = parse_qsl(parsed.query, keep_blank_values=True)
        channel_parameter = next((index for index, (key, _) in enumerate(query) if key == "channel"), None)
        if channel_parameter is None:
            query.append(("channel", str(channel)))
        else:
            query[channel_parameter] = ("channel", str(channel))
        camera_source = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
        sources.append({
            "id": f"cam-{channel}",
            "name": CAMERA_NAMES.get(channel, f"Camera {channel}"),
            "channel": channel,
            "source": camera_source,
        })
    return sources


CAMERA_CHANNELS = parse_camera_channels(os.getenv("CAMERA_CHANNELS", "1,2,3,4,5,6,7"))
SEAT_CONFIDENCE = float(os.getenv("SEAT_CONFIDENCE", "0.15"))
PERSON_CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.25"))
CAMERA_PERSON_CONFIDENCE = {"cam-2": 0.15, "cam-4": 0.15}
GENERIC_CONFIDENCE = float(os.getenv("GENERIC_CONFIDENCE", "0.2"))
GENERIC_INFERENCE_SIZE = int(os.getenv("GENERIC_INFERENCE_SIZE", "1280"))
GENERIC_SEAT_FALLBACK = os.getenv("GENERIC_SEAT_FALLBACK", "true").lower() == "true"
TILE_INFERENCE = os.getenv("TILE_INFERENCE", "true").lower() == "true"
TILE_OVERLAP = float(os.getenv("TILE_OVERLAP", "0.15"))
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
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


class CameraCalibration(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=3, max_length=64)

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if any(
            not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1
            for x, y in points
        ):
            raise ValueError("ROI coordinates must be finite normalized values between 0 and 1.")
        return points


class VisionEngine:
    def __init__(self):
        self.person_model = None
        self.fallback_model = None
        self.seat_model = None
        self.roboflow_seat_model = bool(ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID)
        self.roboflow_workflow = bool(ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID)
        self.roboflow_client = None
        self.last_annotated_image = None
        self._processing_lock = threading.Lock()
        if self.roboflow_seat_model and InferenceHTTPClient is not None:
            client = InferenceHTTPClient(
                api_url=ROBOFLOW_API_URL,
                api_key=ROBOFLOW_API_KEY
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
        self.frame_lock = threading.Lock()
        self.latest_frame: bytes | None = None
        self.last_frame_at: str | None = None
        self.camera_status = "not_configured" if SOURCE == "0" else "connecting"
        self.camera_error: str | None = None
        self.reconnect_attempts = 0
        self.latest = self.mock_result()
        if YOLO and os.getenv("AI_MODE", "mock") == "real":
            person_path = Path(MODEL_PATH)
            fallback_path = Path(FALLBACK_MODEL_PATH) if FALLBACK_MODEL_PATH else None
            if fallback_path and fallback_path.exists():
                self.fallback_model = YOLO(str(fallback_path))
            if person_path.exists():
                if fallback_path and person_path.resolve() == fallback_path.resolve():
                    self.person_model = self.fallback_model
                else:
                    self.person_model = YOLO(str(person_path))
            if SEAT_MODEL_PATH and Path(SEAT_MODEL_PATH).exists():
                self.seat_model = YOLO(SEAT_MODEL_PATH)
                self.roboflow_seat_model = False
                self.roboflow_workflow = False

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

    def _annotate_local_detections(
        self,
        frame: Any,
        detections: list[dict[str, Any]],
        names: dict[int, str],
        base_image: str | None = None,
    ) -> str:
        if base_image:
            annotated = cv2.imdecode(
                np.frombuffer(base64.b64decode(base_image), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if annotated is None:
                raise ValueError("Could not decode the existing detection overlay.")
        else:
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
    def _person_seated_in_chair(
        chair: dict[str, Any],
        person: dict[str, Any],
    ) -> bool:
        left, top, right, bottom = chair["bbox"]
        person_left, person_top, person_right, person_bottom = person["bbox"]
        anchor_x = (person_left + person_right) / 2
        anchor_y = person_top + (person_bottom - person_top) * 0.7
        return left <= anchor_x <= right and top <= anchor_y <= bottom

    @staticmethod
    def _person_anchor_in_seat_roi(
        person: dict[str, Any],
        frame_shape: tuple[int, ...],
        seat_roi: list[tuple[float, float]] | None,
    ) -> bool:
        if not seat_roi:
            return False
        height, width = frame_shape[:2]
        left, top, right, bottom = person["bbox"]
        anchor = (
            (left + right) / (2 * width),
            (top + (bottom - top) * 0.7) / height,
        )
        return cv2.pointPolygonTest(np.asarray(seat_roi, dtype=np.float32), anchor, False) >= 0

    @classmethod
    def _matched_occupied_chairs(
        cls,
        chairs: list[dict[str, Any]],
        people: list[dict[str, Any]],
    ) -> int:
        matches = []
        for chair_index, chair in enumerate(chairs):
            for person_index, person in enumerate(people):
                if cls._person_seated_in_chair(chair, person):
                    chair_left, chair_top, chair_right, chair_bottom = chair["bbox"]
                    person_left, person_top, person_right, person_bottom = person["bbox"]
                    anchor_x = (person_left + person_right) / 2
                    anchor_y = person_top + (person_bottom - person_top) * 0.7
                    center_x = (chair_left + chair_right) / 2
                    center_y = (chair_top + chair_bottom) / 2
                    distance = (anchor_x - center_x) ** 2 + (anchor_y - center_y) ** 2
                    matches.append((distance, chair_index, person_index))
        matches.sort()

        matched_chairs = set()
        matched_people = set()
        for _, chair_index, person_index in matches:
            if chair_index not in matched_chairs and person_index not in matched_people:
                matched_chairs.add(chair_index)
                matched_people.add(person_index)
        return len(matched_chairs)

    @classmethod
    def _merge_detections(cls, *batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        detections = sorted(
            (detection for batch in batches for detection in batch),
            key=lambda detection: detection["confidence"],
            reverse=True,
        )
        merged: list[dict[str, Any]] = []
        for detection in detections:
            if all(
                cls._iou(detection["bbox"], existing["bbox"]) < 0.5
                for existing in merged
            ):
                merged.append(detection)
        return merged

    @staticmethod
    def _is_seat_region(detection: dict[str, Any], frame_height: int) -> bool:
        left, top, right, bottom = detection["bbox"]
        center_y = (top + bottom) / 2
        box_height = max(bottom - top, 0)
        return top >= frame_height * 0.48 and center_y >= frame_height * 0.55 and bottom >= frame_height * 0.62 and box_height >= frame_height * 0.06

    @staticmethod
    def _detection_in_seat_roi(
        detection: dict[str, Any],
        frame_shape: tuple[int, ...],
        seat_roi: list[tuple[float, float]] | None,
    ) -> bool:
        if not seat_roi:
            return True
        height, width = frame_shape[:2]
        left, top, right, bottom = detection["bbox"]
        center = ((left + right) / (2 * width), (top + bottom) / (2 * height))
        polygon = np.array(seat_roi, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, center, False) >= 0

    @staticmethod
    def _seat_in_seat_roi(
        seat: dict[str, Any],
        frame_shape: tuple[int, ...],
        seat_roi: list[tuple[float, float]],
    ) -> bool:
        coordinates = seat.get("coordinates", [])
        if len(coordinates) < 3:
            return False
        height, width = frame_shape[:2]
        normalized = all(0 <= point[0] <= 1 and 0 <= point[1] <= 1 for point in coordinates)
        x_scale = width if normalized else 1
        y_scale = height if normalized else 1
        left = min(point[0] * x_scale for point in coordinates)
        top = min(point[1] * y_scale for point in coordinates)
        right = max(point[0] * x_scale for point in coordinates)
        bottom = max(point[1] * y_scale for point in coordinates)
        detection = {"bbox": [left, top, right, bottom]}
        return VisionEngine._detection_in_seat_roi(detection, frame_shape, seat_roi)

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

    def process(
        self,
        frame: Any = None,
        seat_roi: list[tuple[float, float]] | None = None,
        person_confidence: float | None = None,
    ) -> FrameResult:
        with self._processing_lock:
            return self._process_frame(frame, seat_roi, person_confidence)

    def _process_frame(
        self,
        frame: Any = None,
        seat_roi: list[tuple[float, float]] | None = None,
        person_confidence: float | None = None,
    ) -> FrameResult:
        if frame is None:
            return self.mock_result()
        person_confidence = PERSON_CONFIDENCE if person_confidence is None else person_confidence
        self.last_annotated_image = None
        if self.person_model is None and self.fallback_model is None and not self.roboflow_seat_model:
            self._cache_frame(frame)
            return self.mock_result()
        detector = self.person_model or self.fallback_model
        generic_people_detections: list[dict[str, Any]] = []
        generic_chair_detections: list[dict[str, Any]] = []
        if GENERIC_SEAT_FALLBACK and self.fallback_model is not None:
            generic_predictions = self._predict_detections(
                self.fallback_model,
                frame,
                [0, 56],
                min(person_confidence, GENERIC_CONFIDENCE),
                tiled=False,
                imgsz=GENERIC_INFERENCE_SIZE,
            )
            generic_people_detections = [
                detection for detection in generic_predictions
                if detection["class_id"] == 0
                and detection["confidence"] >= person_confidence
            ]
            generic_chair_detections = [
                detection for detection in generic_predictions
                if detection["class_id"] == 56
            ]
        if detector is self.fallback_model and GENERIC_SEAT_FALLBACK:
            person_detections = generic_people_detections
        else:
            person_detections = self._predict_detections(detector, frame, [0], person_confidence) if detector else []
        if GENERIC_SEAT_FALLBACK and self.fallback_model is not detector:
            person_detections = self._merge_detections(
                person_detections,
                generic_people_detections,
            )
        people = len(person_detections)
        if seat_roi is None:
            seat_total, occupied, empty = self.detect_seats(frame, person_detections)
        else:
            seat_total, occupied, empty = self.detect_seats(frame, person_detections, seat_roi)
        generic_detections = []
        if GENERIC_SEAT_FALLBACK and self.fallback_model is not None:
            generic_chair_detections = [
                detection for detection in generic_chair_detections
                if (
                    self._detection_in_seat_roi(detection, frame.shape[:2], seat_roi)
                    if seat_roi is not None
                    else self._is_seat_region(detection, frame.shape[0])
                )
            ]
            generic_detections = generic_people_detections + generic_chair_detections
            if generic_detections:
                self.last_annotated_image = self._annotate_local_detections(
                    frame,
                    person_detections + generic_chair_detections,
                    self.fallback_model.names,
                    base_image=self.last_annotated_image,
                )
        generic_chairs = sum(detection["class_id"] == 56 for detection in generic_detections)
        if self.seat_model is not None:
            total_seats = max(seat_total, generic_chairs)
            people_in_seat_roi = sum(
                self._person_anchor_in_seat_roi(person, frame.shape[:2], seat_roi)
                for person in person_detections
            )
            generic_occupied = self._matched_occupied_chairs(
                generic_chair_detections,
                person_detections,
            )
            occupied = max(0, min(max(int(occupied), generic_occupied, people_in_seat_roi), int(total_seats)))
            if GENERIC_SEAT_FALLBACK and self.fallback_model is not None:
                person_limit = people_in_seat_roi if seat_roi else people
                occupied = min(occupied, person_limit)
            empty = max(0, int(total_seats) - occupied)
        else:
            total_seats = seat_total if seat_roi is not None else seat_total or self.total_seats
            if people > 0 and total_seats > 0:
                if occupied > people and occupied > 0:
                    occupied = min(people, total_seats)
                elif occupied + empty > total_seats:
                    overspill = occupied + empty - total_seats
                    occupied = max(0, occupied - overspill)
                elif occupied == 0 and (empty == total_seats or empty == 0):
                    occupied = min(people, total_seats)
                empty = max(total_seats - occupied, 0)
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
        self._cache_frame(frame, annotated_image)
        return self.latest

    def _cache_frame(self, frame: Any, annotated_image: str | None = None) -> None:
        if cv2 is None:
            return
        if annotated_image:
            frame_bytes = base64.b64decode(annotated_image)
        else:
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                return
            frame_bytes = encoded.tobytes()
        with self.frame_lock:
            self.latest_frame = frame_bytes
            self.last_frame_at = datetime.now(timezone.utc).isoformat()

    def detect_seats(
        self,
        frame: Any,
        person_results: Any,
        seat_roi: list[tuple[float, float]] | None = None,
    ) -> tuple[int, int, int]:
        if self.seat_model is not None:
            detections = self._predict_detections(self.seat_model, frame, list(self.seat_model.names), SEAT_CONFIDENCE)
            detections = [
                detection for detection in detections
                if self._detection_in_seat_roi(detection, frame.shape[:2], seat_roi)
            ]
            if not detections:
                self.last_annotated_image = None
                return 0, 0, 0
            else:
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
        if self.roboflow_seat_model:
            detections = self._predict_roboflow_seats(frame)
            detections = [
                detection for detection in detections
                if self._is_seat_region(detection, frame.shape[0])
                and self._detection_in_seat_roi(detection, frame.shape[:2], seat_roi)
            ]
            occupied = sum("occupied" in detection["class_name"] or "guest" in detection["class_name"] for detection in detections)
            empty = sum("empty" in detection["class_name"] or "available" in detection["class_name"] or "vacant" in detection["class_name"] for detection in detections)
            
            self.last_annotated_image = self._annotate_roboflow_seats(frame, detections) if detections else None
            
            if detections:
                return len(detections), occupied, empty
        if self.seats:
            configured_seats = self.seats
            if seat_roi is not None:
                configured_seats = [
                    seat for seat in self.seats
                    if self._seat_in_seat_roi(seat, frame.shape[:2], seat_roi)
                ]
            if configured_seats:
                occupied = self.seat_count(frame, person_results, configured_seats)
                return len(configured_seats), occupied, max(len(configured_seats) - occupied, 0)
        if seat_roi is not None:
            return 0, 0, 0
        occupied = min(len(person_results), self.total_seats)
        return self.total_seats, occupied, max(self.total_seats - occupied, 0)

    def seat_count(
        self,
        frame: Any,
        results: Any,
        seats: list[dict[str, Any]] | None = None,
    ) -> int:
        height, width = frame.shape[:2]
        person_points = []
        boxes = [detection["bbox"] for detection in results]
        for left, top, right, bottom in boxes:
            person_points.append((int((left + right) / 2), int(bottom)))
        occupied = 0
        for seat in self.seats if seats is None else seats:
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
        self.camera_status = "connecting"
        self.camera_error = None
        self.capture_thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.capture_thread.start()

    def stream_loop(self):
        if cv2 is None:
            self.camera_status = "opencv_unavailable"
            self.camera_error = "OpenCV is not installed."
            return
        while not self.stop_event.is_set():
            try:
                if self.is_http_source():
                    self.capture_http_mjpeg()
                else:
                    self.capture_video_source()
            except HTTPError as error:
                if error.code == 401:
                    self.camera_status = "authentication_failed"
                    self.camera_error = "Camera authentication failed. Check the configured username and password."
                elif error.code == 404:
                    self.camera_status = "endpoint_not_found"
                    self.camera_error = "The configured camera stream path was not found."
                else:
                    self.camera_status = "reconnecting"
                    self.camera_error = f"Camera endpoint returned HTTP {error.code}."
            except URLError:
                self.camera_status = "reconnecting"
                self.camera_error = "Unable to reach the configured camera stream."
            except (OSError, TimeoutError):
                self.camera_status = "reconnecting"
                self.camera_error = "The camera stream stopped responding."
            if not self.stop_event.is_set():
                self.reconnect_attempts += 1
                time.sleep(5)
        self.camera_status = "stopped"

    @staticmethod
    def is_http_source() -> bool:
        return isinstance(SOURCE, str) and urlsplit(SOURCE).scheme.lower() in {"http", "https"}

    @staticmethod
    def iter_mjpeg_frames(stream: Any, stop_event: threading.Event):
        buffer = bytearray()
        read_chunk = getattr(stream, "read1", None) or stream.read
        while not stop_event.is_set():
            chunk = read_chunk(4096)
            if not chunk:
                return
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > 1:
                        del buffer[:-1]
                    break
                if start:
                    del buffer[:start]
                end = buffer.find(b"\xff\xd9", 2)
                if end < 0:
                    if len(buffer) > 16 * 1024 * 1024:
                        raise OSError("Camera returned an oversized MJPEG frame.")
                    break
                yield bytes(buffer[:end + 2])
                del buffer[:end + 2]

    def capture_http_mjpeg(self):
        credentials = HTTPPasswordMgrWithDefaultRealm()
        credentials.add_password(
            None,
            SOURCE,
            os.getenv("CAMERA_USERNAME", ""),
            os.getenv("CAMERA_PASSWORD", ""),
        )
        opener = build_opener(HTTPDigestAuthHandler(credentials))
        camera_request = Request(
            SOURCE,
            headers={"Accept": "multipart/x-mixed-replace", "User-Agent": "SmartLibCamera/1.0"},
        )
        with opener.open(camera_request, timeout=10) as response:
            if not response.headers.get_content_type().startswith("multipart/"):
                self.camera_status = "unsupported_stream_format"
                self.camera_error = "The camera endpoint did not return an MJPEG multipart stream."
                return
            self.camera_status = "connected"
            self.camera_error = None
            for jpeg_frame in self.iter_mjpeg_frames(response, self.stop_event):
                frame = cv2.imdecode(np.frombuffer(jpeg_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.process(frame)
            if not self.stop_event.is_set():
                self.camera_status = "reconnecting"
                self.camera_error = "The camera stream ended before the connection was stopped."

    def capture_video_source(self):
        capture = cv2.VideoCapture(SOURCE)
        try:
            if not capture.isOpened():
                raise OSError("The configured camera stream could not be opened.")
            self.camera_status = "connected"
            self.camera_error = None
            while not self.stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    return
                self.process(frame)
        finally:
            capture.release()

    def stop_stream(self):
        self.stop_event.set()
        self.camera_status = "stopped"

class CameraFeed:
    def __init__(self, camera: dict[str, Any]):
        self.id = camera["id"]
        self.name = camera["name"]
        self.channel = camera["channel"]
        self.purpose = CAMERA_PURPOSES.get(self.id, "library")
        self.person_confidence = CAMERA_PERSON_CONFIDENCE.get(self.id, PERSON_CONFIDENCE)
        self.source = camera["source"]
        self.lifecycle_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.capture_thread: threading.Thread | None = None
        self.latest_raw_frame: Any = None
        self.latest_raw_sequence = 0
        self.latest_source_frame: bytes | None = None
        self.source_sequence = 0
        self.latest_annotated_frame: bytes | None = None
        self.annotated_sequence = 0
        self.latest = FrameResult(mode="waiting")
        self.seat_roi: list[tuple[float, float]] | None = None
        self.status = "connecting"
        self.error: str | None = None
        self.last_frame_at: str | None = None
        self.last_detection_at: str | None = None
        self.reconnect_attempts = 0
        self.previous_people: int | None = None
        self.detection_history: deque[tuple[int, int, int]] = deque(maxlen=5)

    @staticmethod
    def _encode_preview(frame: Any, max_width: int, quality: int) -> bytes | None:
        height, width = frame.shape[:2]
        if width > max_width:
            resized_height = max(1, round(height * max_width / width))
            frame = cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded.tobytes() if success else None

    def start(self) -> None:
        with self.lifecycle_lock:
            if self.capture_thread and self.capture_thread.is_alive():
                if not self.stop_event.is_set():
                    return
                self.capture_thread.join(timeout=15)
                if self.capture_thread.is_alive():
                    raise RuntimeError(f"Previous capture for {self.name} has not stopped.")
            self.stop_event = threading.Event()
            self.status = "connecting"
            self.error = None
            with self.condition:
                self.detection_history.clear()
                self.previous_people = None
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(self.stop_event,),
                daemon=True,
                name=f"camera-{self.id}",
            )
            self.capture_thread.start()

    def _capture_loop(self, stop_event: threading.Event) -> None:
        if cv2 is None:
            self.status = "opencv_unavailable"
            self.error = "OpenCV is not installed."
            return

        while not stop_event.is_set():
            capture = None
            try:
                if isinstance(self.source, int):
                    capture = cv2.VideoCapture(self.source)
                else:
                    capture = cv2.VideoCapture(
                        self.source,
                        cv2.CAP_FFMPEG,
                        [
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                        ],
                    )
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    raise OSError("The configured camera stream could not be opened.")

                next_preview_at = 0.0
                while not stop_event.is_set():
                    success, frame = capture.read()
                    if not success:
                        raise OSError("The camera stream stopped responding.")
                    self.status = "connected"
                    self.error = None
                    now = time.monotonic()
                    with self.condition:
                        self.latest_raw_frame = frame
                        self.latest_raw_sequence += 1
                        self.last_frame_at = datetime.now(timezone.utc).isoformat()
                    if now >= next_preview_at:
                        preview = self._encode_preview(frame, max_width=720, quality=68)
                        if preview:
                            with self.condition:
                                self.latest_source_frame = preview
                                self.source_sequence += 1
                                self.condition.notify_all()
                        next_preview_at = now + 0.25
            except (OSError, TimeoutError, cv2.error) as error:
                if not stop_event.is_set():
                    self.status = "reconnecting"
                    self.error = f"Camera capture failed: {type(error).__name__}."
            finally:
                if capture is not None:
                    capture.release()
            if not stop_event.is_set():
                self.reconnect_attempts += 1
                stop_event.wait(3)
        self.status = "stopped"
        with self.condition:
            self.condition.notify_all()

    def raw_snapshot(self) -> tuple[int, Any]:
        with self.condition:
            return self.latest_raw_sequence, self.latest_raw_frame

    def publish_detection(self, sequence: int, frame: Any, result: FrameResult) -> None:
        annotated = cv2.imdecode(
            np.frombuffer(base64.b64decode(result.annotated_image), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        ) if result.annotated_image else frame
        encoded = self._encode_preview(annotated, max_width=960, quality=78)
        with self.condition:
            self.detection_history.append((
                result.people_count,
                result.occupied_seats,
                result.occupied_seats + result.empty_seats,
            ))
            history = tuple(self.detection_history)
            people = int(median(sample[0] for sample in history) + 0.5)
            occupied = int(median(sample[1] for sample in history) + 0.5)
            total = int(median(sample[2] for sample in history) + 0.5)
            occupied = min(occupied, people, total)
            previous_people = people if self.previous_people is None else self.previous_people
            self.previous_people = people
            result = result.model_copy(update={
                "people_count": people,
                "occupied_seats": occupied,
                "empty_seats": max(total - occupied, 0),
                "occupancy_percentage": round(occupied / total * 100, 1) if total else 0,
                "entries": max(people - previous_people, 0),
                "exits": max(previous_people - people, 0),
            })
            self.latest = result
            self.latest_annotated_frame = encoded
            self.annotated_sequence += 1
            self.last_detection_at = datetime.now(timezone.utc).isoformat()
            self.condition.notify_all()

    def record_inference_error(self, error: Exception) -> None:
        with self.condition:
            self.error = f"Frame inference failed: {type(error).__name__}."

    def public_state(self) -> dict[str, Any]:
        with self.condition:
            return {
                "id": self.id,
                "name": self.name,
                "channel": self.channel,
                "purpose": self.purpose,
                "status": self.status,
                "stream_running": bool(self.capture_thread and self.capture_thread.is_alive()),
                "last_frame_at": self.last_frame_at,
                "last_detection_at": self.last_detection_at,
                "reconnect_attempts": self.reconnect_attempts,
                "error": self.error,
                "seat_roi": self.seat_roi,
                "calibrated": self.seat_roi is not None,
                "occupancy": self.latest.model_dump(exclude={"annotated_image"}),
            }

    def stream(self, annotated: bool):
        stop_event = self.stop_event
        sequence = -1
        while not stop_event.is_set():
            with self.condition:
                image_sequence = self.annotated_sequence if annotated else self.source_sequence
                image = self.latest_annotated_frame if annotated else self.latest_source_frame
                if image is None or image_sequence == sequence:
                    self.condition.wait(timeout=1)
                    image_sequence = self.annotated_sequence if annotated else self.source_sequence
                    image = self.latest_annotated_frame if annotated else self.latest_source_frame
            if image and image_sequence != sequence:
                sequence = image_sequence
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(image)).encode()
                    + b"\r\n\r\n"
                    + image
                    + b"\r\n"
                )


class CameraMonitor:
    def __init__(
        self,
        vision_engine: VisionEngine,
        cameras: list[dict[str, Any]],
        calibration_path: str | Path | None = None,
    ):
        self.engine = vision_engine
        self.feeds = {camera["id"]: CameraFeed(camera) for camera in cameras}
        self.default_camera_id = "cam-2" if "cam-2" in self.feeds else next(iter(self.feeds), None)
        self.lifecycle_lock = threading.Lock()
        self.calibration_lock = threading.Lock()
        self.calibration_path = Path(
            calibration_path or os.getenv("CAMERA_CALIBRATION_PATH", "/data/seat-areas.json")
        )
        self.stop_event = threading.Event()
        self.inference_thread: threading.Thread | None = None
        self._load_calibrations()

    def _load_calibrations(self) -> None:
        if not self.calibration_path.exists():
            return
        try:
            calibrations = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Could not read persisted camera seat-area calibrations.") from error
        if not isinstance(calibrations, dict):
            raise RuntimeError("Persisted camera calibrations must be a JSON object.")
        for camera_id, points in calibrations.items():
            if camera_id in self.feeds:
                calibration = CameraCalibration(points=points)
                self.feeds[camera_id].seat_roi = calibration.points

    def set_calibration(
        self,
        feed: CameraFeed,
        points: list[tuple[float, float]] | None,
    ) -> None:
        with self.calibration_lock:
            calibrations = {
                camera_id: current_feed.seat_roi
                for camera_id, current_feed in self.feeds.items()
                if current_feed.seat_roi is not None and camera_id != feed.id
            }
            if points is not None:
                calibrations[feed.id] = points
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.calibration_path.with_name(
                f"{self.calibration_path.name}.{threading.get_ident()}.tmp"
            )
            temporary_path.write_text(json.dumps(calibrations, indent=2), encoding="utf-8")
            temporary_path.replace(self.calibration_path)
            with feed.condition:
                feed.seat_roi = points

    def start(self) -> None:
        if not self.feeds:
            return
        with self.lifecycle_lock:
            if self.inference_thread and self.inference_thread.is_alive():
                if not self.stop_event.is_set():
                    for feed in self.feeds.values():
                        feed.start()
                    return
                self.inference_thread.join(timeout=15)
                if self.inference_thread.is_alive():
                    raise RuntimeError("The previous camera inference worker has not stopped.")
            self.stop_event = threading.Event()
            for feed in self.feeds.values():
                feed.start()
            self.inference_thread = threading.Thread(
                target=self._inference_loop,
                args=(self.stop_event,),
                daemon=True,
                name="camera-inference",
            )
            self.inference_thread.start()

    def _inference_loop(self, stop_event: threading.Event) -> None:
        processed_sequences = {camera_id: 0 for camera_id in self.feeds}
        while not stop_event.is_set():
            processed_any = False
            for camera_id, feed in self.feeds.items():
                if stop_event.is_set():
                    return
                sequence, frame = feed.raw_snapshot()
                if frame is None or sequence <= processed_sequences[camera_id]:
                    continue
                try:
                    with feed.condition:
                        seat_roi = feed.seat_roi
                    result = self.engine.process(
                        frame,
                        seat_roi=seat_roi,
                        person_confidence=feed.person_confidence,
                    )
                    feed.publish_detection(sequence, frame, result)
                except Exception as error:
                    feed.record_inference_error(error)
                processed_sequences[camera_id] = sequence
                processed_any = True
            if not processed_any:
                stop_event.wait(0.05)

    def stop(self) -> None:
        self.stop_event.set()
        for feed in self.feeds.values():
            feed.stop_event.set()
            with feed.condition:
                feed.condition.notify_all()

    def get_feed(self, camera_id: str | None) -> CameraFeed:
        selected_id = camera_id or self.default_camera_id
        if selected_id is None or selected_id not in self.feeds:
            raise HTTPException(status_code=404, detail="Camera not found.")
        return self.feeds[selected_id]

    def public_state(self) -> list[dict[str, Any]]:
        return [feed.public_state() for feed in self.feeds.values()]

    def occupancy(self, camera_id: str | None, group: str | None = None) -> FrameResult:
        if camera_id is not None:
            return self.get_feed(camera_id).latest

        selected_group = group or "library"
        if selected_group not in CAMERA_GROUPS | {"all"}:
            raise HTTPException(status_code=422, detail="Unknown camera group.")
        feeds = [
            feed
            for feed in self.feeds.values()
            if selected_group == "all" or feed.purpose == selected_group
        ]
        results = [
            (feed.purpose, feed.latest)
            for feed in feeds
            if feed.status == "connected" and feed.latest.mode != "waiting"
        ]
        if not results:
            return FrameResult(mode="waiting")

        people = sum(result.people_count for _, result in results)
        seating_results = [
            result
            for purpose, result in results
            if purpose != "entrance"
        ]
        occupied = sum(result.occupied_seats for result in seating_results)
        empty = sum(result.empty_seats for result in seating_results)
        total = occupied + empty
        modes = {result.mode for _, result in results}
        return FrameResult(
            people_count=people,
            occupied_seats=occupied,
            empty_seats=empty,
            occupancy_percentage=round(occupied / total * 100, 1) if total else 0,
            entries=sum(result.entries for _, result in results),
            exits=sum(result.exits for _, result in results),
            mode=modes.pop() if len(modes) == 1 else "multi-camera",
        )


engine = VisionEngine()
camera_monitor = CameraMonitor(engine, configured_camera_sources(SOURCE, CAMERA_CHANNELS))


@app.on_event("startup")
def start_configured_stream():
    camera_monitor.start()


@app.get("/ai/health")
def health():
    mode = "roboflow-workflow" if engine.roboflow_workflow else ("roboflow-seat" if engine.roboflow_seat_model else ("yolo" if engine.person_model else "mock"))
    active_streams = sum(camera["stream_running"] for camera in camera_monitor.public_state())
    return {
        "status": "ok",
        "mode": mode,
        "stream_running": active_streams > 0,
        "camera_count": len(camera_monitor.feeds),
        "active_cameras": active_streams,
        "person_model": MODEL_PATH,
        "seat_model": SEAT_MODEL_PATH or None,
        "roboflow_model": ROBOFLOW_MODEL_ID if engine.roboflow_seat_model else None,
        "roboflow_workflow": ROBOFLOW_WORKFLOW_ID if engine.roboflow_workflow else None,
    }


@app.get("/ai/cameras")
def cameras():
    return camera_monitor.public_state()


@app.put("/ai/cameras/{camera_id}/roi")
def save_camera_roi(camera_id: str, calibration: CameraCalibration):
    feed = camera_monitor.get_feed(camera_id)
    try:
        camera_monitor.set_calibration(feed, calibration.points)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not persist the camera seat-area calibration.") from error
    return {"camera_id": feed.id, "seat_roi": feed.seat_roi, "calibrated": True}


@app.delete("/ai/cameras/{camera_id}/roi")
def clear_camera_roi(camera_id: str):
    feed = camera_monitor.get_feed(camera_id)
    try:
        camera_monitor.set_calibration(feed, None)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not persist the camera seat-area calibration.") from error
    return {"camera_id": feed.id, "seat_roi": None, "calibrated": False}


@app.get("/ai/occupancy", response_model=FrameResult)
def occupancy(
    camera_id: str | None = Query(default=None),
    group: str | None = Query(default=None, pattern="^(library|discussion|entrance|all)$"),
):
    if camera_monitor.feeds:
        return camera_monitor.occupancy(camera_id, group)
    return engine.latest


@app.get("/ai/camera/status")
def camera_status(camera_id: str | None = Query(default=None)):
    if camera_monitor.feeds:
        return camera_monitor.get_feed(camera_id).public_state()
    return {
        "status": engine.camera_status,
        "stream_running": bool(engine.capture_thread and engine.capture_thread.is_alive()),
        "last_frame_at": engine.last_frame_at,
        "reconnect_attempts": engine.reconnect_attempts,
        "error": engine.camera_error,
    }


@app.get("/ai/stream.mjpeg")
def camera_stream(
    camera_id: str | None = Query(default=None),
    annotated: bool = Query(default=False),
):
    if camera_monitor.feeds:
        feed = camera_monitor.get_feed(camera_id)
        frame_generator = feed.stream(annotated)
    else:
        def frame_generator():
            while not engine.stop_event.is_set():
                with engine.frame_lock:
                    frame = engine.latest_frame
                if frame:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.1)

    return StreamingResponse(
        frame_generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


@app.get("/ai/camera/frame.jpg")
def camera_frame(
    camera_id: str | None = Query(default=None),
    annotated: bool = Query(default=False),
):
    feed = camera_monitor.get_feed(camera_id)
    with feed.condition:
        image = feed.latest_annotated_frame if annotated else feed.latest_source_frame
    if image is None:
        raise HTTPException(status_code=503, detail="The selected camera has not produced a frame yet.")
    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


@app.post("/ai/stream/start")
def start_stream():
    camera_monitor.start()
    return {"status": "started", "camera_count": len(camera_monitor.feeds)}


@app.post("/ai/stream/stop")
def stop_stream():
    camera_monitor.stop()
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
