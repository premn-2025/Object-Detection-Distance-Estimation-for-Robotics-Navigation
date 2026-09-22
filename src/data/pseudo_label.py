"""Cross-checked pseudo-labelling.

Two problems this solves, both caused by combining partially-annotated datasets:

1. **Stop signs are out of domain.**  COCO's stop-sign images are web photos.
   BDD100K has 34 k dash-cam ``traffic sign`` boxes but never says which are stop
   signs.  A COCO-pretrained detector knows what a stop sign looks like but fires
   on billboards and bus liveries.  Requiring *both* - a stop-sign prediction
   **and** a BDD100K ``traffic sign`` box at the same location - turns two noisy
   signals into clean, in-domain labels.  Precision comes from the agreement, not
   from trusting either source.

2. **ROADWork has no stop-sign labels at all.**  Every ROADWork frame containing
   an unlabelled stop sign is a false negative that actively teaches the model to
   suppress stop signs in exactly the viewpoint we care about.  Here there is no
   second opinion available, so the detector runs alone at a much higher
   confidence threshold and the result is treated as what it is: a weaker label.

Everything written by this module is tagged in ``pseudo_labels.json`` so the
provenance of any box can be traced back later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import Annotation, YoloDatasetWriter

COCO_STOP_SIGN_ID = 11          # 'stop sign' in the 80-class COCO ordering


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4)/(M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)).astype(np.float32)


def _coco_detector(weights: str, device: str, conf: float):
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names

    def run(frame):
        r = model.predict(frame, conf=conf, verbose=False, device=device)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        cls = b.cls.cpu().numpy().astype(int)
        keep = cls == COCO_STOP_SIGN_ID
        return (b.xyxy.cpu().numpy()[keep].astype(np.float32),
                b.conf.cpu().numpy()[keep].astype(np.float32))

    assert names.get(COCO_STOP_SIGN_ID) == "stop sign", (
        f"class {COCO_STOP_SIGN_ID} is {names.get(COCO_STOP_SIGN_ID)!r}, not 'stop sign'")
    return run


def _hash_fraction(key: str, seed: int = 0) -> float:
    """Deterministic value in [0, 1) from a name.

    The split must be a pure function of the image name, not of its position in
    an iteration. A sequential RNG looks equivalent but is not: re-running the
    miner with a different confidence threshold changes how many frames are kept,
    which shifts every later draw, which moves images across the split boundary
    and leaves stale copies behind in the old one. That is exactly the leak
    ``scripts/check_leakage.py`` caught.
    """
    import hashlib

    h = hashlib.blake2s(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def mine_bdd_stop_signs(bdd_zip: Path, out_root: Path, class_names, *,
                        weights: str = "yolo11m.pt", device: str = "cpu",
                        conf: float = 0.35, iou_thresh: float = 0.35,
                        max_side: int = 1280, val_fraction: float = 0.15,
                        limit: int = 0, writer: YoloDatasetWriter = None,
                        seed: int = 0) -> tuple:
    """Mine stop signs from BDD100K by detector/annotation agreement."""
    from .bdd100k import iter_sign_frames

    stop_idx = list(class_names).index("stop_sign")
    writer = writer or YoloDatasetWriter(out_root, class_names, max_side=max_side)
    detect = _coco_detector(weights, device, conf)

    kept, examined, rejected = 0, 0, 0
    manifest = []
    for stem, frame, sign_boxes in iter_sign_frames(bdd_zip, limit=limit):
        examined += 1
        det_boxes, det_conf = detect(frame)
        if len(det_boxes) == 0:
            continue
        gt = np.asarray(sign_boxes, np.float32).reshape(-1, 4)
        ious = iou_matrix(det_boxes, gt)
        agree = ious.max(axis=1) >= iou_thresh if gt.size else np.zeros(len(det_boxes), bool)
        rejected += int((~agree).sum())
        if not agree.any():
            continue
        # snap to the human-drawn box: BDD100K's corners are more consistent than
        # the detector's, and the distance estimate is only as good as the box
        best = ious.argmax(axis=1)
        anns = [Annotation(stop_idx, *gt[best[i]]) for i in np.nonzero(agree)[0]]
        split = "val" if _hash_fraction(stem, seed) < val_fraction else "train"
        if writer.add(split, f"bddstop_{stem}", frame, anns):
            kept += len(anns)
            manifest.append({"stem": f"bddstop_{stem}", "split": split, "source": "bdd100k",
                             "n_boxes": len(anns),
                             "conf": [round(float(c), 3) for c in det_conf[agree]]})
    stats = {"examined": examined, "boxes_kept": kept,
             "detections_rejected_no_gt_agreement": rejected}
    print(f"  [mine:bdd] {stats}")
    return writer, manifest, stats


def complete_roadwork_stop_signs(image_dir: Path, label_dir: Path, class_names, *,
                                 weights: str = "yolo11m.pt", device: str = "cpu",
                                 conf: float = 0.55, limit: int = 0,
                                 pattern: str = "rw_*.jpg") -> tuple:
    """Append stop-sign boxes to already-written ROADWork label files.

    No second opinion is available here, so ``conf`` defaults much higher than in
    :func:`mine_bdd_stop_signs`.  This trades recall for precision on purpose: a
    wrong extra box is worse than a missing one when the base labels are good.

    ``pattern`` restricts the scan to ROADWork frames.  It must: the same
    directory also holds COCO frames, which already carry human stop-sign labels,
    and re-detecting those would append a second, near-duplicate box to every one
    of them.
    """
    import cv2

    stop_idx = list(class_names).index("stop_sign")
    detect = _coco_detector(weights, device, conf)
    images = sorted(Path(image_dir).glob(pattern))
    if limit:
        images = images[:limit]

    added, touched = 0, 0
    manifest = []
    for img_path in tqdm(images, desc="pseudo:roadwork", unit="img"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        boxes, confs = detect(frame)
        if len(boxes) == 0:
            continue
        h, w = frame.shape[:2]
        lbl = Path(label_dir) / f"{img_path.stem}.txt"
        lines = lbl.read_text().splitlines() if lbl.exists() else []
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            ann = Annotation(stop_idx, float(x1), float(y1), float(x2), float(y2)).clipped(w, h)
            if ann.w < 6 or ann.h < 6:
                continue
            lines.append(ann.to_yolo(w, h))
            added += 1
            manifest.append({"stem": img_path.stem, "conf": round(float(c), 3)})
        lbl.write_text("\n".join(lines))
        touched += 1
    stats = {"images_scanned": len(images), "images_modified": touched, "boxes_added": added}
    print(f"  [pseudo:roadwork] {stats}")
    return manifest, stats


def save_manifest(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
