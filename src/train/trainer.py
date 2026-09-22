"""Thin, explicit wrapper around ultralytics training.

Keeping this separate from the CLI means the two stages, the ablation and the
post-pruning fine-tune all go through exactly one code path, so the numbers they
produce are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path="configs/train.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train(stage_cfg: dict, common: dict, *, model: str = None, device: str = "auto",
          name: str = None, overrides: dict = None):
    """Run one training stage. Returns the ultralytics results object."""
    from ultralytics import YOLO

    cfg = dict(common)
    cfg.update({k: v for k, v in stage_cfg.items() if v is not None})
    if overrides:
        cfg.update(overrides)

    # `hard_augment` is ours, not an ultralytics argument: pop it and install the
    # albumentations stack before the trainer builds its dataloaders.
    preset = cfg.pop("hard_augment", "off")
    if preset and preset != "off":
        from .augment import install_hard_augmentation

        install_hard_augmentation(preset)

    weights = model or cfg.pop("model", None)
    if weights is None:
        raise ValueError("no starting weights given for this stage")
    cfg.pop("model", None)
    if name:
        cfg["name"] = name
    cfg["device"] = resolve_device(device)

    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"dataset descriptor {data_yaml} is missing - run scripts/prepare_data.py first")

    print(f"[train] start={weights} data={cfg['data']} epochs={cfg.get('epochs')} "
          f"imgsz={cfg.get('imgsz')} device={cfg['device']}")
    yolo = YOLO(weights)
    results = yolo.train(**cfg)
    return yolo, results


def summarise(results, out_path=None) -> dict:
    """Pull the headline metrics out of an ultralytics result object."""
    rd = getattr(results, "results_dict", {}) or {}
    summary = {
        "mAP50-95": rd.get("metrics/mAP50-95(B)"),
        "mAP50": rd.get("metrics/mAP50(B)"),
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
        "fitness": rd.get("fitness"),
        "save_dir": str(getattr(results, "save_dir", "")),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2))
    return summary


def evaluate(weights, data_yaml, *, device: str = "auto", imgsz: int = 640,
             split: str = "val", out_path=None) -> dict:
    """Validate a checkpoint and return per-class plus overall metrics."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), imgsz=imgsz, split=split,
                  device=resolve_device(device), verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    per_class = {}
    try:
        for i, ci in enumerate(m.ap_class_index):
            per_class[names[int(ci)]] = {
                "mAP50-95": float(m.box.maps[int(ci)]),
                "precision": float(m.box.p[i]),
                "recall": float(m.box.r[i]),
                "mAP50": float(m.box.ap50[i]),
            }
    except (AttributeError, IndexError, TypeError):
        pass
    out = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "mAP50-95": float(m.box.map),
        "mAP50": float(m.box.map50),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": per_class,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
    return out
