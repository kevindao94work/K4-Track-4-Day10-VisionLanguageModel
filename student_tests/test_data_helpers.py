import torch
from student.data import collate


def test_collate_pads_labels_and_concats_vision():
    examples = [
        {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
         "labels": torch.tensor([[-100, 2, 3]]), "pixel_values": torch.ones(4, 5), "image_grid_thw": torch.ones(1, 3)},
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
         "labels": torch.tensor([[-100, 2]]), "pixel_values": torch.ones(4, 5)*2, "image_grid_thw": torch.ones(1, 3)*2},
    ]
    batch = collate(examples, pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1,2,3],[1,2,0]]
    assert batch["labels"].tolist() == [[-100,2,3],[-100,2,-100]]
    assert batch["pixel_values"].shape == (8,5)
    assert batch["image_grid_thw"].shape == (2,3)
