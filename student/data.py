"""Dataset checks and Qwen multimodal example encoding/collation."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
from student.labels import CLASSES, objects_to_answer

PROMPT = ("Detect road defects in this image. Return one plain JSON list only, without prose or Markdown fences. "
          "Each item must have exactly the keys label and bbox_2d. Allowed labels: pothole, crack, manhole. "
          "bbox_2d is [x1,y1,x2,y2] with integer coordinates on the normalized 0-1000 scale. "
          "If no objects are detected, return [].")


def validate_splits(root="data"):
    root = Path(root)
    meta = json.loads((root / "splits/meta.json").read_text())
    all_ids, sessions, summary = set(), defaultdict(set), {}
    for split in ("train", "val", "test"):
        rows = [json.loads(x) for x in (root / f"splits/{split}.jsonl").read_text().splitlines() if x.strip()]
        ids, boxes = set(), {c: 0 for c in CLASSES}
        for row in rows:
            assert row["image_id"] not in ids and row["image_id"] not in all_ids
            ids.add(row["image_id"]); all_ids.add(row["image_id"])
            sessions[split].add(row["session"])
            image_path = root / row["image_path"]
            with Image.open(image_path) as im:
                im.verify()
                assert im.size == (row["width"], row["height"])
            for obj in row["objects"]:
                assert obj["label"] in CLASSES
                xc, yc, w, h = obj["yolo"]
                assert all(0 <= v <= 1 for v in (xc, yc, w, h)) and w > 0 and h > 0
                assert xc-w/2 >= -1e-8 and yc-h/2 >= -1e-8
                assert xc+w/2 <= 1+1e-8 and yc+h/2 <= 1+1e-8
                boxes[obj["label"]] += 1
        summary[split] = {"images": len(rows), "boxes": boxes}
    assert not (sessions["train"] & sessions["val"] or sessions["train"] & sessions["test"] or sessions["val"] & sessions["test"])
    assert meta["dataset_version"]
    return {"dataset_version": meta["dataset_version"], "counts": summary}


def make_user_message(image, prompt=PROMPT):
    return {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}


def encode_example(processor, image, objects):
    user = make_user_message(image)
    prompt = processor.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
    answer = objects_to_answer(objects)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": answer}], tokenize=True,
        add_generation_prompt=False, return_dict=True, return_tensors="pt")
    n = prompt["input_ids"].shape[1]
    if full["input_ids"].shape[1] < n or not full["input_ids"][0, :n].equal(prompt["input_ids"][0]):
        raise ValueError("prompt tokens are not an exact prefix of full conversation")
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100
    return full


def collate(examples, pad_token_id):
    import torch
    text_keys = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")
    max_len = max(int(ex["input_ids"].shape[-1]) for ex in examples)
    batch = {}
    for key in text_keys:
        if not all(key in ex for ex in examples):
            continue
        pad = -100 if key == "labels" else (pad_token_id if key == "input_ids" else 0)
        tensors = []
        for ex in examples:
            t = ex[key]
            amount = max_len - t.shape[-1]
            tensors.append(torch.nn.functional.pad(t, (0, amount), value=pad))
        batch[key] = torch.cat(tensors, dim=0)
    for key in set.intersection(*(set(ex.keys()) for ex in examples)) - set(text_keys):
        values = [ex[key] for ex in examples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.cat(values, dim=0)
    return batch
