from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run SmartLib person tracking on a video, webcam, or RTSP source.")
    parser.add_argument("--source", required=True, help="Video path, webcam index, or RTSP URL")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--seat-model", default="", help="Optional YOLO chair model; classes 1 and 2 are treated as occupied")
    parser.add_argument("--output", type=Path, default=Path("ai-service/runs/live-test.mp4"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(args.model)
    seat_model = YOLO(args.seat_model) if args.seat_model else None
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.track(frame, persist=True, classes=[0], verbose=False)[0]
        people = len(result.boxes) if result.boxes is not None else 0
        annotated = result.plot()
        label = f"SmartLib people: {people}"
        if seat_model:
            seats = seat_model.predict(frame, verbose=False)[0]
            class_ids = [int(class_id) for class_id in seats.boxes.cls.cpu().tolist()] if seats.boxes is not None and seats.boxes.cls is not None else []
            label += f" | chairs: {len(class_ids)} | occupied: {sum(class_id in {1, 2} for class_id in class_ids)}"
            annotated = seats.plot()
        cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 210, 160), 2)
        writer.write(annotated)
        frames += 1
        if args.show:
            cv2.imshow("SmartLib live test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frames} frames -> {args.output}")


if __name__ == "__main__":
    main()
