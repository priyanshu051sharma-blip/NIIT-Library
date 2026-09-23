from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("archive (1)/valid/valid"))
    parser.add_argument("--output", type=Path, default=Path("ai-service/data/person-test.mp4"))
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    images = sorted(args.images.glob("*.jpg"))[: args.limit]
    if not images:
        raise SystemExit("No JPG images found")
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is not None:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    writer.release()
    print(f"Created {len(images)}-frame test video at {args.output}")


if __name__ == "__main__":
    main()
