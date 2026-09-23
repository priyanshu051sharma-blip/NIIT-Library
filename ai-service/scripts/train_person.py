from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a person detector for SmartLib camera viewpoints.")
    parser.add_argument("--data", type=Path, default=Path("ai-service/data/person/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="smartlib-person")
    args = parser.parse_args()
    model = YOLO(args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, device=args.device, project="ai-service/runs", name=args.name, patience=5, workers=2, cache=False)
    print("Best weights:", Path("ai-service/runs") / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
