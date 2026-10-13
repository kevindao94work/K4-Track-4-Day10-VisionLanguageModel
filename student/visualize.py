"""Ground truth and prediction overlays for actual dataset images."""
from pathlib import Path
from PIL import Image, ImageDraw
from student.labels import box_to_pixels, yolo_to_pixels
from scripts.score import parse_output

COLORS = {"pothole": "lime", "crack": "cyan", "manhole": "yellow", "other": "magenta"}


def annotate(image_path, row, raw, output_path, title="prediction"):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for obj in row["objects"]:
        box = yolo_to_pixels(obj["yolo"], row["width"], row["height"])
        draw.rectangle(box, outline=COLORS[obj["label"]], width=3)
        draw.text((box[0], max(0, box[1]-12)), f"GT {obj['label']}", fill=COLORS[obj["label"]])
    parsed = parse_output(raw)
    if parsed:
        for obj in parsed:
            box = box_to_pixels(obj["bbox"], row["width"], row["height"])
            col = COLORS.get(obj["label"], COLORS["other"])
            draw.rectangle(box, outline=col, width=1)
            draw.text((box[0], box[1]+2), f"{title} {obj['label']}", fill=col)
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)
