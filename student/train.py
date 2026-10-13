"""Compact fp16 QLoRA trainer with separately saved projector parameters."""
from __future__ import annotations
import json, math, random, time
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from student.model import PROJECTOR_MARKERS


def _extra_trainables(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and any(marker in n for marker in PROJECTOR_MARKERS)}


def save_checkpoint(model, directory, step, optimizer=None, scheduler=None, scaler=None):
    directory = Path(directory) / f"step-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    torch.save(_extra_trainables(model), directory / "extra_trainable.pt")
    state = {"step": step, "rng": torch.random.get_rng_state(), "python_rng": random.getstate()}
    if torch.cuda.is_available(): state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if optimizer is not None: state["optimizer"] = optimizer.state_dict()
    if scheduler is not None: state["scheduler"] = scheduler.state_dict()
    if scaler is not None: state["scaler"] = scaler.state_dict()
    torch.save(state, directory / "training_state.pt")
    return directory


def restore_extra_trainables(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    missing = set(state) - set(params)
    if missing: raise KeyError(f"projector weights not found in model: {sorted(missing)[:3]}")
    with torch.no_grad():
        for name, value in state.items(): params[name].copy_(value.to(params[name].device, dtype=params[name].dtype))


def restore_training_state(path, optimizer=None, scheduler=None, scaler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state: scaler.load_state_dict(state["scaler"])
    if "rng" in state: torch.random.set_rng_state(state["rng"])
    if "python_rng" in state: random.setstate(state["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state: torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state.get("step", 0))


def train(model, batches, output_dir, max_steps=300, lr=2e-4, grad_accum=4, seed=42):
    random.seed(seed); torch.manual_seed(seed)
    model.train(); model.config.use_cache = False
    try: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError: model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warmup = max(1, int(max_steps * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/warmup if s < warmup else .5*(1+math.cos(math.pi*(s-warmup)/max(1,max_steps-warmup))))
    scaler = torch.amp.GradScaler("cuda")
    it = iter(batches); steps = 0; started = time.perf_counter(); skipped_steps = 0
    torch.cuda.reset_peak_memory_stats()
    while steps < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try: batch = next(it)
            except StopIteration:
                it = iter(batches); batch = next(it)
            batch = {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16): loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {steps+1}")
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer); clip_grad_norm_(params, 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        # A reduced scale means GradScaler skipped this optimizer update.
        if scaler.get_scale() < old_scale:
            skipped_steps += 1
            continue
        scheduler.step(); steps += 1
        if steps % 100 == 0 or steps == max_steps:
            save_checkpoint(model, output_dir, steps, optimizer, scheduler, scaler)
    minutes = (time.perf_counter()-started)/60
    peak = torch.cuda.max_memory_allocated()/1024**3
    return {"trainable_params": sum(p.numel() for p in params), "peak_vram_gb": peak,
            "train_minutes": minutes, "steps": steps, "skipped_steps": skipped_steps}
