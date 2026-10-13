"""Deterministic conversion between YOLO labels and Qwen grounding JSON."""
from __future__ import annotations
import json
from typing import Iterable, Sequence

CLASSES = ("pothole", "crack", "manhole")


def yolo_to_box(label: str, yolo: Sequence[float]) -> dict:
    if label not in CLASSES:
        raise ValueError(f"unsupported label: {label!r}")
    if len(yolo) != 4:
        raise ValueError("YOLO box must contain xc, yc, width, height")
    xc, yc, width, height = map(float, yolo)
    if not all(0.0 <= value <= 1.0 for value in (xc, yc, width, height)):
        raise ValueError("YOLO values must be in [0, 1]")
    if width <= 0 or height <= 0:
        raise ValueError("box width and height must be positive")
    raw = ((xc-width/2)*1000, (yc-height/2)*1000,
           (xc+width/2)*1000, (yc+height/2)*1000)
    x1, y1, x2, y2 = (max(0, min(1000, int(round(v)))) for v in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box collapses after rounding and clamping")
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects: Iterable[dict]) -> str:
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda item: (item["bbox_2d"][1], item["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def box_to_pixels(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("expected a 4-coordinate box and positive image dimensions")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError("normalized box must be ordered and within [0, 1000]")
    return (round(x1 * width / 1000), round(y1 * height / 1000),
            round(x2 * width / 1000), round(y2 * height / 1000))


def yolo_to_pixels(yolo: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    xc, yc, bw, bh = map(float, yolo)
    return ((xc-bw/2)*width, (yc-bh/2)*height,
            (xc+bw/2)*width, (yc+bh/2)*height)
