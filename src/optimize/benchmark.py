"""Speed + accuracy benchmark across model variants.

Rules this follows, so the table means something:

* every variant is timed through the *same* pre/post-processing path
  (``src.models.detector``), on the *same* frames, in the same process;
* latency is the median over many runs after a warm-up, and is measured in two
  passes keeping the second - the first variant in a sweep is otherwise
  penalised by CPU frequency ramp and cache warming (see `time_variant`);
* GPU timings synchronise before stopping the clock;
* accuracy is re-measured for every variant on the same validation split, so a
  quantisation that quietly destroys localisation cannot hide behind its FPS.

Latency is reported end-to-end (letterbox + forward + NMS) because that is what
bounds the robot's control loop; the raw forward pass alone would flatter the
quantised models.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Variant:
    """One thing to benchmark."""

    label: str
    weights: str
    device: str = "cpu"
    half: bool = False
    class_names: list = None
    imgsz: int = 640
    notes: str = ""
    extra: dict = field(default_factory=dict)


def host_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info


def load_frames(image_dir, n: int = 32, size=None) -> list:
    """A fixed set of real frames - synthetic noise is not a fair NMS workload."""
    import cv2

    paths = sorted(Path(image_dir).glob("*.jpg"))[:n]
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if size is not None:
            img = cv2.resize(img, size)
        frames.append(img)
    if not frames:
        raise FileNotFoundError(f"no usable jpg frames under {image_dir}")
    return frames


def _sync(device: str) -> None:
    if device != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


def time_variant(variant: Variant, frames, runs: int = 60, warmup: int = 10,
                 passes: int = 2) -> dict:
    """Median end-to-end latency and FPS for one variant.

    Timed in two passes, keeping the second. The first variant measured in a
    suite is otherwise penalised by effects that have nothing to do with the
    model - CPU frequency ramp, page-cache warming, allocator growth. That is not
    hypothetical: measured first in the sweep, the FP32 baseline read 11.6 FPS,
    while the identical checkpoint measured back-to-back with others read 26.7
    FPS. Taking the first number would have inflated every CPU speed-up ratio in
    the table by more than 2x.
    """
    import time

    from ..models.detector import build_detector

    det = build_detector(variant.weights, device=variant.device, imgsz=variant.imgsz,
                         class_names=variant.class_names, half=variant.half)
    for i in range(warmup):
        det.infer(frames[i % len(frames)])
    _sync(variant.device)

    times = []
    for _ in range(max(1, passes)):
        times = []
        for i in range(runs):
            f = frames[i % len(frames)]
            t0 = time.perf_counter()
            det.infer(f)
            _sync(variant.device)
            times.append(time.perf_counter() - t0)
    t = np.asarray(times)
    return {
        "latency_ms_median": float(np.median(t) * 1e3),
        "latency_ms_p90": float(np.percentile(t, 90) * 1e3),
        "fps": float(1.0 / np.median(t)),
        "runs": runs,
    }


def accuracy_variant(variant: Variant, data_yaml, split: str = "val",
                     subset: int = 500, gt_cache: dict = None) -> dict:
    """mAP for one variant, through one evaluator for every backend.

    Deliberately not ultralytics' ``val``: that path differs between a .pt and an
    ONNX model, and on CPU an INT8 model needs ~25 minutes for the full split, so
    a six-variant sweep would cost over an hour. Routing every backend through
    :mod:`src.optimize.cocoeval` on a fixed subset keeps the comparison
    apples-to-apples (identical pre/post-processing, identical images, identical
    metric) and adds AP-by-object-size for free.
    """
    from pathlib import Path as _P

    import yaml as _yaml

    from ..models.detector import build_detector
    from .cocoeval import evaluate, run_detector, yolo_labels_to_coco

    try:
        cfg = _yaml.safe_load(_P(data_yaml).read_text())
        root = _P(cfg["path"])
        names = [cfg["names"][i] for i in sorted(cfg["names"])]
        images_dir, labels_dir = root / "images" / split, root / "labels" / split

        if gt_cache is not None and "gt" in gt_cache:
            gt = gt_cache["gt"]
        else:
            paths = sorted(images_dir.glob("*.jpg"))
            if subset and len(paths) > subset:
                paths = paths[::max(1, len(paths) // subset)][:subset]
            gt = yolo_labels_to_coco(images_dir, labels_dir, names, image_paths=paths)
            if gt_cache is not None:
                gt_cache["gt"] = gt

        det = build_detector(variant.weights, device=variant.device, imgsz=variant.imgsz,
                             class_names=names, half=variant.half, conf=0.001)
        res = evaluate(gt, run_detector(det, gt, images_dir), names)
        return {"mAP50-95": res["mAP50-95"], "mAP50": res["mAP50"],
                "AP_small": res["AP_small"], "AP_large": res["AP_large"],
                "eval_images": len(gt["images"])}
    except Exception as exc:                      # noqa: BLE001 - report, do not abort the suite
        return {"mAP50-95": None, "mAP50": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


def run_suite(variants, frames, data_yaml=None, *, runs: int = 60,
              measure_accuracy: bool = True, split: str = "val",
              subset: int = 500) -> list:
    rows = []
    gt_cache: dict = {}          # build the COCO ground truth once, reuse per variant
    for v in variants:
        if not Path(v.weights).exists():
            print(f"  [bench] skip {v.label}: {v.weights} missing")
            continue
        print(f"  [bench] {v.label} ...", flush=True)
        row = {"variant": v.label, "weights": v.weights, "device": v.device,
               "imgsz": v.imgsz, "notes": v.notes,
               "size_mb": round(Path(v.weights).stat().st_size / 1e6, 2)}
        row.update(v.extra)
        try:
            row.update(time_variant(v, frames, runs=runs))
        except Exception as exc:                  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        if measure_accuracy and data_yaml is not None and "error" not in row:
            row.update(accuracy_variant(v, data_yaml, split, subset, gt_cache))
        rows.append(row)
        fps = row.get("fps")
        print(f"      fps={fps:.1f}" if fps else "      failed",
              f"mAP50-95={row.get('mAP50-95')}" if measure_accuracy else "")
    return rows


def to_markdown(rows, baseline_label: str = None) -> str:
    """Render the benchmark as a table, with speed-ups relative to a baseline."""
    base = next((r for r in rows if r["variant"] == baseline_label), None)
    head = ("| variant | device | size (MB) | latency p50 (ms) | FPS | speed-up | "
            "mAP50-95 | mAP50 | notes |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        if "error" in r and "fps" not in r:
            lines.append(f"| {r['variant']} | {r['device']} | {r.get('size_mb','-')} | "
                         f"- | - | - | - | - | {r.get('error','')} |")
            continue
        speed = "-"
        if base and base.get("fps") and r.get("fps") and r["device"] == base["device"]:
            speed = f"{r['fps'] / base['fps']:.2f}x"
        m95 = r.get("mAP50-95")
        m50 = r.get("mAP50")
        lines.append(
            f"| {r['variant']} | {r['device']} | {r.get('size_mb','-')} | "
            f"{r.get('latency_ms_median', float('nan')):.1f} | {r.get('fps', float('nan')):.1f} | "
            f"{speed} | {m95:.4f} | {m50:.4f} | {r.get('notes','')} |"
            if m95 is not None and m50 is not None else
            f"| {r['variant']} | {r['device']} | {r.get('size_mb','-')} | "
            f"{r.get('latency_ms_median', float('nan')):.1f} | {r.get('fps', float('nan')):.1f} | "
            f"{speed} | - | - | {r.get('notes','')} |")
    return "\n".join(lines)


def save(rows, host: dict, out_dir="outputs/benchmark") -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "benchmark.json"
    js.write_text(json.dumps({"host": host, "results": rows}, indent=2, default=float))
    md = out_dir / "benchmark.md"
    return js, md
