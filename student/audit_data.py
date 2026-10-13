"""Audit authentic instructor-prepared manifests and write aggregate-only evidence."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
from student.data import CLASSES


def audit_dataset(data_root="data", out_path="report/evidence/dataset_audit.json"):
    root = Path(data_root)
    meta = json.loads((root / "splits/meta.json").read_text(encoding="utf-8"))
    seen_ids, sessions, result = set(), defaultdict(set), {}
    for split in ("train", "val", "test"):
        rows = [json.loads(s) for s in (root / f"splits/{split}.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        split_ids, counts, outbounds, degenerate = set(), {c: 0 for c in CLASSES}, 0, 0
        for row in rows:
            image_id = row["image_id"]
            if image_id in split_ids or image_id in seen_ids:
                raise ValueError(f"duplicate image_id {image_id}")
            split_ids.add(image_id); seen_ids.add(image_id)
            sessions[split].add(row["session"])
            image_path = root / row["image_path"]
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != (row["width"], row["height"]):
                    raise ValueError(f"dimension mismatch for {image_id}")
            if row["width"] != 640 or row["height"] != 360:
                raise ValueError(f"unexpected resolution for {image_id}")
            for obj in row["objects"]:
                if obj["label"] not in CLASSES:
                    raise ValueError(f"unknown class for {image_id}: {obj['label']}")
                xc, yc, w, h = obj["yolo"]
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    raise ValueError(f"invalid normalized YOLO values for {image_id}")
                x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(1, x2), min(1, y2)
                if cx2 <= cx1 or cy2 <= cy1 or w == 0 or h == 0:
                    degenerate += 1
                outbounds += int(x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1)
                counts[obj["label"]] += 1
        result[split] = {"images": len(rows), "boxes": counts,
                         "boxes_clipped_at_image_edge": outbounds,
                         "degenerate_or_invisible_boxes": degenerate}
    overlaps = {"train_val": len(sessions["train"] & sessions["val"]),
                "train_test": len(sessions["train"] & sessions["test"]),
                "val_test": len(sessions["val"] & sessions["test"])}
    if any(overlaps.values()):
        raise ValueError(f"session leakage: {overlaps}")
    if len(seen_ids) != sum(v["images"] for v in result.values()):
        raise ValueError("image count mismatch")
    evidence = {"dataset_version": meta["dataset_version"], "splits": result,
                "session_intersections": overlaps, "total_unique_images": len(seen_ids)}
    dest = Path(out_path); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    print(json.dumps(audit_dataset(), ensure_ascii=False, indent=2))
