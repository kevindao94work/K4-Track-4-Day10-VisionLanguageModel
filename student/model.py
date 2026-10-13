"""Qwen3-VL loading, adapter selection, and module audit helpers."""
from __future__ import annotations
import re
import torch

LLM_PATTERN = r"language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
VIT_PATTERN = r"visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
PROJECTOR_MARKERS = ("visual.merger", "visual.deepstack_merger_list")


def load_model(model_id="Qwen/Qwen3-VL-4B-Instruct", device_map="auto"):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def apply_setup(model, setup):
    from peft import LoraConfig, get_peft_model
    if setup not in {"e1", "e2", "e3"}:
        raise ValueError("setup must be e1, e2, or e3")
    for p in model.parameters():
        p.requires_grad_(False)
    names = [name for name, mod in model.named_modules() if hasattr(mod, "weight") and re.fullmatch(LLM_PATTERN, name)]
    if len(names) != 252:
        raise RuntimeError(f"expected 252 LLM targets, found {len(names)}")
    if setup == "e3":
        vit = [name for name, mod in model.named_modules() if hasattr(mod, "weight") and re.fullmatch(VIT_PATTERN, name)]
        if len(vit) != 96:
            raise RuntimeError(f"expected 96 ViT targets, found {len(vit)}")
        targets = names + vit
    else:
        targets = names
    peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                  target_modules=targets, bias="none"))
    if setup in {"e2", "e3"}:
        updated = 0
        for name, p in peft_model.named_parameters():
            if any(marker in name for marker in PROJECTOR_MARKERS):
                p.data = p.data.float()
                p.requires_grad_(True)
                updated += p.numel()
        if updated == 0:
            raise RuntimeError("projector parameters were not found")
    peft_model.enable_input_require_grads()
    return peft_model


def audit_model(model):
    report = {"quantized_modules": [], "projector": [], "lm_head": []}
    for name, module in model.named_modules():
        if "4bit" in type(module).__name__.lower() or "4bit" in name.lower():
            report["quantized_modules"].append({"name": name, "type": type(module).__name__})
        if any(marker in name for marker in PROJECTOR_MARKERS):
            report["projector"].append({"name": name, "type": type(module).__name__})
        if name == "lm_head":
            report["lm_head"].append({"name": name, "type": type(module).__name__})
    return report
