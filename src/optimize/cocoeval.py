"""COCO-style evaluation with size stratification, for any detector backend.

Ultralytics' own ``val`` reports one overall mAP. Two things this project needs
that it does not give directly:

* **AP by object size.** A cone 40 m away is twelve pixels tall. Overall mAP is
  dominated by the large, near, easy objects, so it hides exactly the regime a
  navigation system cares about. COCO's small / medium / large split makes the
  hard regime visible.
* **The same metric for a quantised ONNX model and for a corrupted copy of the
  validation set.** Running everything through one evaluator that takes a
  ``BaseDetector`` keeps INT8, pruned and FP32 numbers comparable, and lets the
  robustness benchmark reuse the identical code path.

COCO size bands (in pixels of box area): small < 32², medium < 96², large above.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np


def yolo_labels_to_coco(images_dir, labels_dir, class_names, *, image_paths=None) -> dict:
    """Build a COCO ground-truth dict from a YOLO split."""
    import cv2

    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    paths = list(image_paths) if image_paths is not None else sorted(images_dir.glob("*.jpg"))

    images, annotations = [], []
    ann_id = 1
    for img_id, p in enumerate(paths, start=1):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lf = labels_dir / f"{p.stem}.txt"
        if not lf.exists():
            continue
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            ci, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            bw_px, bh_px = bw * w, bh * h
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": ci + 1,
                "bbox": [x, y, bw_px, bh_px], "area": bw_px * bh_px, "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }


def run_detector(detector, gt: dict, images_dir, *, degrade=None, conf: float = 0.001,
                 progress: bool = False) -> list:
    """Run a detector over the GT images, returning COCO-format detections.

    ``degrade`` is an optional ``callable(bgr_image) -> bgr_image`` applied before
    inference, which is how the robustness benchmark corrupts the val set without
    ever writing the corrupted frames to disk.
    """
    import cv2

    images_dir = Path(images_dir)
    results = []
    iterator = gt["images"]
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="eval", unit="img", leave=False)

    for rec in iterator:
        img = cv2.imread(str(images_dir / rec["file_name"]))
        if img is None:
            continue
        if degrade is not None:
            img = degrade(img)
        det = detector.infer(img)
        for box, score, ci in zip(det.boxes, det.scores, det.class_ids):
            x1, y1, x2, y2 = (float(v) for v in box)
            results.append({
                "image_id": rec["id"], "category_id": int(ci) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
            })
    return results


def evaluate(gt: dict, detections: list, class_names, *, quiet: bool = True) -> dict:
    """COCOeval, returning overall, per-size and per-class AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"mAP50-95": 0.0, "mAP50": 0.0, "AP_small": 0.0, "AP_medium": 0.0,
                "AP_large": 0.0, "per_class": {}, "n_detections": 0,
                "note": "detector produced no boxes above threshold"}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(detections))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    s = ev.stats
    out = {
        "mAP50-95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
        "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
        "AR_100": float(s[8]),
        "n_detections": len(detections),
        "n_gt_boxes": len(gt["annotations"]),
    }

    # per class, by re-running the summary restricted to one category at a time
    per_class = {}
    for i, name in enumerate(class_names):
        with contextlib.redirect_stdout(io.StringIO()):
            ev_c = COCOeval(coco_gt, coco_dt, "bbox")
            ev_c.params.catIds = [i + 1]
            ev_c.evaluate()
            ev_c.accumulate()
            ev_c.summarize()
        n = sum(1 for a in gt["annotations"] if a["category_id"] == i + 1)
        per_class[name] = {"mAP50-95": float(ev_c.stats[0]), "mAP50": float(ev_c.stats[1]),
                           "AP_small": float(ev_c.stats[3]), "n_gt": n}
    out["per_class"] = per_class
    return out


def gt_size_histogram(gt: dict, class_names) -> dict:
    """How many GT boxes fall in each COCO size band - context for AP_small."""
    bands = {"small (<32^2 px)": 0, "medium (32-96^2 px)": 0, "large (>96^2 px)": 0}
    per_class = {n: dict(bands) for n in class_names}
    for a in gt["annotations"]:
        area = a["area"]
        key = ("small (<32^2 px)" if area < 32 ** 2
               else "medium (32-96^2 px)" if area < 96 ** 2 else "large (>96^2 px)")
        bands[key] += 1
        name = class_names[a["category_id"] - 1]
        per_class[name][key] += 1
    return {"overall": bands, "per_class": per_class}


def save(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    return path
