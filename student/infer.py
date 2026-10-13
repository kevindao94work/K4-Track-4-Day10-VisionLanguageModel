"""Deterministic batch inference and prediction JSONL utilities."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
from student.data import PROMPT, make_user_message


def predict_one(model, processor, image, max_new_tokens=256):
    messages = [make_user_message(image)]
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    prompt_len = batch["input_ids"].shape[1]
    output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0]


def generate_split(model, processor, rows, data_root="data", out_path="work/pred.jsonl"):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            with Image.open(Path(data_root) / row["image_path"]) as im:
                raw = predict_one(model, processor, im.convert("RGB"))
            f.write(json.dumps({"image_id": row["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
    return out


def read_split(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
