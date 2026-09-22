"""Shared helpers for turning heterogeneous datasets into one YOLO-format set.

Every converter in this package emits the same on-disk layout::

    <root>/images/<split>/<stem>.jpg
    <root>/labels/<split>/<stem>.txt      # cls cx cy w h, all normalised
    <root>/data.yaml                      # ultralytics dataset descriptor

Images are re-encoded at a bounded resolution on the way in: the sources ship
1920x1080 JPEGs, training happens at 640, and keeping 10 GB of full-resolution
frames around only slows the dataloader down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml


@dataclass
class Annotation:
    """One axis-aligned box in absolute pixel coordinates."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def scaled(self, s: float) -> "Annotation":
        return Annotation(self.cls, self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s)

    def clipped(self, w: int, h: int) -> "Annotation":
        return Annotation(self.cls,
                          float(np.clip(self.x1, 0, w - 1)), float(np.clip(self.y1, 0, h - 1)),
                          float(np.clip(self.x2, 0, w - 1)), float(np.clip(self.y2, 0, h - 1)))

    def to_yolo(self, w: int, h: int) -> str:
        cx, cy = (self.x1 + self.x2) / 2 / w, (self.y1 + self.y2) / 2 / h
        bw, bh = self.w / w, self.h / h
        return f"{self.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


@dataclass
class Sample:
    """An image plus its boxes, before it is written out."""

    stem: str
    annotations: list = field(default_factory=list)


class YoloDatasetWriter:
    """Accumulates samples into a YOLO directory tree."""

    def __init__(self, root: Path, class_names: Sequence[str], max_side: int = 1280,
                 jpeg_quality: int = 90) -> None:
        self.root = Path(root)
        self.class_names = list(class_names)
        self.max_side = int(max_side)
        self.jpeg_quality = int(jpeg_quality)
        self.counts: dict = {}

    def _dirs(self, split: str):
        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return img_dir, lbl_dir

    def add(self, split: str, stem: str, image: np.ndarray, annotations: Iterable) -> bool:
        """Write one image + label file. Returns False if the image was unusable."""
        if image is None or image.size == 0:
            return False
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        img_dir, lbl_dir = self._dirs(split)
        ok = cv2.imwrite(str(img_dir / f"{stem}.jpg"), image,
                         [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return False

        lines = []
        for ann in annotations:
            a = ann.scaled(scale).clipped(w, h)
            if a.w < 2 or a.h < 2:          # sub-2px boxes survive nothing downstream
                continue
            lines.append(a.to_yolo(w, h))
            key = (split, self.class_names[a.cls])
            self.counts[key] = self.counts.get(key, 0) + 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        self.counts[(split, "__images__")] = self.counts.get((split, "__images__"), 0) + 1
        if not lines:
            self.counts[(split, "__negatives__")] = self.counts.get((split, "__negatives__"), 0) + 1
        return True

    def write_yaml(self, extra: dict = None) -> Path:
        splits = sorted({s for s, _ in self.counts})
        doc = {
            "path": str(self.root.resolve()).replace("\\", "/"),
            "names": {i: n for i, n in enumerate(self.class_names)},
        }
        for s in splits:
            if (self.root / "images" / s).exists():
                doc[s] = f"images/{s}"
        if extra:
            doc.update(extra)
        p = self.root / "data.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        return p

    def write_stats(self) -> Path:
        stats: dict = {}
        for (split, name), n in sorted(self.counts.items()):
            stats.setdefault(split, {})[name] = n
        p = self.root / "stats.json"
        p.write_text(json.dumps(stats, indent=2))
        return p

    def summary(self) -> str:
        lines = []
        for split in sorted({s for s, _ in self.counts}):
            imgs = self.counts.get((split, "__images__"), 0)
            neg = self.counts.get((split, "__negatives__"), 0)
            per = {n: c for (s, n), c in self.counts.items()
                   if s == split and not n.startswith("__")}
            body = "  ".join(f"{k}={v}" for k, v in sorted(per.items()))
            lines.append(f"  {split:6s} images={imgs:6d} (neg {neg:5d})  {body}")
        return "\n".join(lines)


def imdecode_bytes(raw: bytes):
    """JPEG bytes -> BGR array, or None when the file is corrupt."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
