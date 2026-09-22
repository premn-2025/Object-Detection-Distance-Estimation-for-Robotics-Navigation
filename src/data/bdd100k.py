"""BDD100K -> (a) domain-adaptation pre-training set, (b) dash-cam stop signs.

BDD100K does not contain cones or barriers, and its ``traffic sign`` class lumps
every sign together, so it cannot supply the three target classes directly.  It
is still the most valuable dataset here, for two reasons:

(a) **Domain adaptation.**  The detector starts from COCO weights - web photos.
    The robot sees a forward-facing camera on a road, at dusk, in rain.  One
    fine-tuning pass over BDD100K's 10 classes moves the backbone into that
    domain before it ever sees a cone.  See ``src/train/stage1_bdd.py``.

(b) **In-domain stop signs.**  ``src/data/pseudo_label.py`` runs the
    COCO-pretrained detector over these frames and keeps a stop-sign prediction
    only where BDD100K independently annotated a ``traffic sign`` box at the
    same place.  Two weak, independent signals agreeing gives labels far cleaner
    than either alone, and they are dash-cam frames rather than web photos.

The 10 k-image official validation split is used (the only redistributable
packaging of BDD100K I could obtain) and re-split internally for stage 1.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .common import Annotation, YoloDatasetWriter, imdecode_bytes

CSV_MEMBER = "bdd100k.csv"

# BDD100K detection classes, in a fixed order so the stage-1 head is reproducible.
BDD_CLASSES = [
    "car", "traffic sign", "traffic light", "pedestrian", "truck", "bus",
    "bicycle", "rider", "motorcycle", "train",
]
# Rare/ambiguous labels folded into their obvious parent rather than dropped.
BDD_ALIASES = {"other vehicle": "car", "other person": "pedestrian", "trailer": "truck"}


def read_labels(zip_path: Path) -> dict:
    """image member path -> list of (label, x1, y1, x2, y2)."""
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(CSV_MEMBER).decode("utf8")
    out = defaultdict(list)
    for row in csv.DictReader(io.StringIO(raw)):
        try:
            box = (float(row["xmin"]), float(row["ymin"]),
                   float(row["xmax"]), float(row["ymax"]))
        except (TypeError, ValueError):
            continue
        out[row["image_path"].lstrip("/")].append((row["label"], *box))
    return dict(out)


def _member_lookup(zf: zipfile.ZipFile) -> dict:
    return {n.lstrip("/"): n for n in zf.namelist() if n.lower().endswith(".jpg")}


def convert_stage1(zip_path: Path, out_root: Path, *, max_side: int = 1280,
                   val_fraction: float = 0.15, min_box_px: float = 4.0,
                   limit: int = 0, seed: int = 0) -> YoloDatasetWriter:
    """Build the 10-class BDD100K YOLO set used for domain-adaptive pre-training."""
    import random

    labels = read_labels(zip_path)
    idx = {n: i for i, n in enumerate(BDD_CLASSES)}
    writer = YoloDatasetWriter(out_root, BDD_CLASSES, max_side=max_side)

    keys = sorted(labels)
    random.Random(seed).shuffle(keys)
    if limit:
        keys = keys[:limit]
    n_val = int(len(keys) * val_fraction)
    split_of = {k: ("val" if i < n_val else "train") for i, k in enumerate(keys)}

    with zipfile.ZipFile(zip_path) as zf:
        members = _member_lookup(zf)
        missing = 0
        for key in tqdm(keys, desc="bdd100k:stage1", unit="img"):
            member = members.get(key)
            if member is None:
                missing += 1
                continue
            anns = []
            for label, x1, y1, x2, y2 in labels[key]:
                label = BDD_ALIASES.get(label, label)
                if label not in idx or (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
                    continue
                anns.append(Annotation(idx[label], x1, y1, x2, y2))
            frame = imdecode_bytes(zf.read(member))
            if frame is None:
                missing += 1
                continue
            writer.add(split_of[key], f"bdd_{Path(key).stem}", frame, anns)
        if missing:
            print(f"  [bdd100k] skipped {missing} unreadable images")
    return writer


def iter_sign_frames(zip_path: Path, limit: int = 0):
    """Yield ``(stem, bgr_image, traffic_sign_boxes)`` for stop-sign mining."""
    labels = read_labels(zip_path)
    keys = [k for k, v in labels.items() if any(l == "traffic sign" for l, *_ in v)]
    keys.sort()
    if limit:
        keys = keys[:limit]
    with zipfile.ZipFile(zip_path) as zf:
        members = _member_lookup(zf)
        for key in tqdm(keys, desc="bdd100k:sign-frames", unit="img"):
            member = members.get(key)
            if member is None:
                continue
            frame = imdecode_bytes(zf.read(member))
            if frame is None:
                continue
            boxes = [(x1, y1, x2, y2) for l, x1, y1, x2, y2 in labels[key]
                     if l == "traffic sign"]
            yield Path(key).stem, frame, boxes


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", default="data/raw/bdd100k_val.zip")
    p.add_argument("--out", default="data/processed/bdd_stage1")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    w = convert_stage1(Path(args.zip), Path(args.out), max_side=args.max_side,
                       limit=args.limit)
    print(w.summary())
    w.write_yaml()
    w.write_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
