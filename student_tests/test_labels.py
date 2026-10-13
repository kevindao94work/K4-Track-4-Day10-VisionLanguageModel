import pytest
from student.labels import box_to_pixels, objects_to_answer, yolo_to_box


def test_center_box():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4]) == {
        "label": "pothole", "bbox_2d": [400, 300, 600, 700]}


def test_corner_clamps():
    assert yolo_to_box("crack", [0.05, 0.1, 0.1, 0.2])["bbox_2d"] == [0, 0, 100, 200]


def test_order_and_compact_json():
    got = objects_to_answer([
        {"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
        {"label": "manhole", "yolo": [0.2, 0.2, 0.1, 0.1]},
    ])
    assert got == '[{"label":"manhole","bbox_2d":[150,150,250,250]},{"label":"crack","bbox_2d":[750,750,850,850]}]'


def test_empty():
    assert objects_to_answer([]) == "[]"


@pytest.mark.parametrize("label,box", [("bad", [0.5]*4), ("crack", [0.5,0.5,0,0.2]), ("crack", [1.1,0.5,0.1,0.1])])
def test_reject_invalid(label, box):
    with pytest.raises(ValueError):
        yolo_to_box(label, box)


def test_round_trip_pixel_tolerance():
    box = yolo_to_box("manhole", [0.51, 0.52, 0.12, 0.2])["bbox_2d"]
    px = box_to_pixels(box, 640, 360)
    assert px == (288, 151, 365, 223)
