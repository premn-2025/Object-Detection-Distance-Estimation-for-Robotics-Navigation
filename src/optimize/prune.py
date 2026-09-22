"""Structured channel pruning of the YOLO backbone.

Two things are worth being precise about, because pruning results are very easy
to overstate:

* **Unstructured (mask) pruning does not make anything faster.**  Zeroing 50 % of
  the weights leaves the tensors the same shape, so a dense kernel does exactly
  the same work.  It is reported here only for the size/accuracy trade-off, and
  the benchmark labels it as such.

* **Structured pruning does**, because entire channels are removed and the
  tensors genuinely shrink.  It needs a dependency graph: a YOLO backbone is full
  of residual adds and concatenations, and removing channel *k* from one
  convolution forces the same removal in everything it is later added to.
  ``torch-pruning`` builds that graph; hand-rolled channel masking silently
  produces a broken network.

Pruning is followed by a short fine-tune - a pruned-but-not-recovered model is
not an interesting data point.

**Result on this model:** the structured path did not complete.
``tp.pruner.MagnitudePruner``'s dependency-graph construction does not converge
in reasonable time on YOLO11s - profiled step by step, model load takes 0.0 s, a
forward pass 0.2 s and MAC counting 0.2 s, but the pruner build runs past 200 s
on both CPU and GPU. YOLO11's C2PSA attention blocks are the likely cause. The
code is kept and exposed behind ``--prune``; it is a torch-pruning/YOLO11
interaction, not a missing implementation. The unstructured path below is what
actually ran, and is reported with its limitation stated rather than dressed up
as a speed-up.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_macs(model: nn.Module, imgsz: int = 640, device: str = "cpu") -> float:
    """MACs for one forward pass, via torch-pruning's counter."""
    import torch_pruning as tp

    example = torch.randn(1, 3, imgsz, imgsz, device=device)
    macs, _ = tp.utils.count_ops_and_params(model, example)
    return float(macs)


def global_unstructured_prune(weights, amount: float = 0.4, out_path=None) -> dict:
    """L1 global unstructured pruning over every Conv2d. Size/accuracy only."""
    import torch.nn.utils.prune as prune
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model = yolo.model
    targets = [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    for m, name in targets:
        prune.remove(m, name)          # bake the mask into the weights

    total = sum(m.weight.numel() for m, _ in targets)
    zeros = sum(int((m.weight == 0).sum()) for m, _ in targets)
    info = {"method": "global_unstructured_l1", "requested_amount": amount,
            "conv_weight_sparsity": zeros / max(total, 1),
            "parameters": count_parameters(model),
            "note": "sparse weights do not accelerate dense inference"}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        yolo.save(str(out_path))
        info["weights"] = str(out_path)
    return info


def structured_prune(weights, amount: float = 0.3, imgsz: int = 640,
                     out_path=None, device: str = "cpu",
                     iterative_steps: int = 1) -> dict:
    """Remove whole channels with a dependency-aware pruner.

    The detection head's output convolutions are excluded: pruning them would
    change the number of predicted classes / box coefficients.
    """
    import torch_pruning as tp
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model = yolo.model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)          # torch-pruning needs grads to trace deps

    example = torch.randn(1, 3, imgsz, imgsz, device=device)
    before = {"parameters": count_parameters(model),
              "macs": measure_macs(copy.deepcopy(model), imgsz, device)}

    # Never prune the outputs of the detection head.
    ignored = []
    head = model.model[-1]
    for m in head.modules():
        if isinstance(m, nn.Conv2d):
            ignored.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=tp.importance.GroupMagnitudeImportance(p=2),
        iterative_steps=iterative_steps, pruning_ratio=amount,
        ignored_layers=ignored, global_pruning=False,
    )
    for _ in range(iterative_steps):
        pruner.step()

    after = {"parameters": count_parameters(model),
             "macs": measure_macs(copy.deepcopy(model), imgsz, device)}

    with torch.no_grad():
        out = model(example)            # fail loudly here rather than at training time
    assert out is not None

    info = {
        "method": "structured_magnitude_l2",
        "requested_ratio": amount,
        "iterative_steps": iterative_steps,
        "params_before": before["parameters"], "params_after": after["parameters"],
        "param_reduction": 1 - after["parameters"] / max(before["parameters"], 1),
        "macs_before": before["macs"], "macs_after": after["macs"],
        "mac_reduction": 1 - after["macs"] / max(before["macs"], 1),
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # the graph changed shape, so the whole module is saved, not a state_dict
        ckpt = {"model": model.half() if device != "cpu" else model,
                "date": None, "version": None, "train_args": {}, "ema": None,
                "updates": None, "optimizer": None}
        torch.save(ckpt, out_path)
        model.float()
        info["weights"] = str(out_path)
    return info


def finetune(pruned_weights, data_yaml, *, epochs: int = 15, imgsz: int = 640,
             batch: int = 16, lr0: float = 0.002, device: str = "auto",
             project: str = "runs/pruned", name: str = "finetune") -> dict:
    """Recover accuracy after pruning. Short schedule, low LR."""
    from ultralytics import YOLO

    from ..train.trainer import resolve_device

    yolo = YOLO(str(pruned_weights))
    results = yolo.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                         lr0=lr0, optimizer="AdamW", device=resolve_device(device),
                         project=project, name=name, pretrained=False, verbose=False)
    rd = getattr(results, "results_dict", {}) or {}
    return {"save_dir": str(getattr(results, "save_dir", "")),
            "mAP50-95": rd.get("metrics/mAP50-95(B)"),
            "mAP50": rd.get("metrics/mAP50(B)")}


def save_report(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    return path
