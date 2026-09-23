from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def convert(split: str, archive_root: Path, output_root: Path) -> int:
    source_dir = archive_root / split / split
    csv_path = source_dir / "_annotations.csv"
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[str]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as annotation_file:
        for row in csv.DictReader(annotation_file):
            width = float(row["width"])
            height = float(row["height"])
            center_x = ((float(row["xmin"]) + float(row["xmax"])) / 2) / width
            center_y = ((float(row["ymin"]) + float(row["ymax"])) / 2) / height
            box_width = (float(row["xmax"]) - float(row["xmin"])) / width
            box_height = (float(row["ymax"]) - float(row["ymin"])) / height
            rows[row["filename"]].append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    for filename, labels in rows.items():
        source_image = source_dir / filename
        if not source_image.exists():
            continue
        shutil.copy2(source_image, image_dir / filename)
        (label_dir / f"{Path(filename).stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("archive (1)"))
    parser.add_argument("--output", type=Path, default=Path("ai-service/data/person"))
    args = parser.parse_args()
    counts = {split: convert(split, args.archive, args.output) for split in ("train", "valid", "test")}
    (args.output / "data.yaml").write_text(f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: person\n", encoding="utf-8")
    print(f"Prepared person dataset: {counts}")


if __name__ == "__main__":
    main()
