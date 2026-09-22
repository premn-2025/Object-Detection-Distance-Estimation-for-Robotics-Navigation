"""ROADWork (CMU) -> unified navigation classes.

ROADWork ships COCO-style annotations over 1920x1080 dash-cam frames of real
road work zones.  It is the only public driving dataset I found that annotates
cones and temporary barriers densely, which is exactly what a ground robot has
to avoid.

Its 50-way taxonomy is collapsed to ``cone`` / ``barrier`` via
``configs/classes.yaml``; everything else is dropped, but images whose only
content was dropped still go in as negatives, because "work vehicle but no cone"
is precisely the kind of frame that produces false positives otherwise.

Images are read straight out of ``images.zip`` - unpacking 10.6 GB to disk only
to downscale it again would be a waste.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from .common import Annotation, YoloDatasetWriter, imdecode_bytes

SPLIT_FILES = {
    "train": "annotations/instances_train_gps_split.json",
    "val": "annotations/instances_val_gps_split.json",
}


def load_class_config(path="configs/classes.yaml") -> tuple:
    cfg = yaml.safe_load(Path(path).read_text())
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    return names, cfg["roadwork_map"]


def _index_zip(zf: zipfile.ZipFile) -> dict:
    """basename -> member name, so annotation file names can be looked up."""
    return {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}


def convert(images_zip: Path, ann_dir: Path, out_root: Path,
            class_names, roadwork_map: dict, *, max_side: int = 1280,
            min_box_px: float = 6.0, keep_negatives: bool = True,
            limit: int = 0) -> YoloDatasetWriter:
    """Stream ROADWork into ``out_root`` in YOLO format."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    writer = YoloDatasetWriter(out_root, class_names, max_side=max_side)

    with zipfile.ZipFile(images_zip) as zf:
        members = _index_zip(zf)
        for split, rel in SPLIT_FILES.items():
            ann_path = Path(ann_dir) / Path(rel).name
            if not ann_path.exists():
                raise FileNotFoundError(f"missing ROADWork annotation file: {ann_path}")
            coco = json.loads(ann_path.read_text())

            cat_name = {c["id"]: c["name"] for c in coco["categories"]}
            by_image = defaultdict(list)
            for a in coco["annotations"]:
                if a.get("iscrowd"):
                    continue
                unified = roadwork_map.get(cat_name.get(a["category_id"], ""))
                if unified is None:
                    continue
                x, y, w, h = a["bbox"]
                if w < min_box_px or h < min_box_px:
                    continue
                by_image[a["image_id"]].append(
                    Annotation(name_to_idx[unified], x, y, x + w, y + h))

            images = coco["images"]
            if limit:
                images = images[:limit]
            missing = 0
            for img in tqdm(images, desc=f"roadwork:{split}", unit="img"):
                anns = by_image.get(img["id"], [])
                if not anns and not keep_negatives:
                    continue
                member = members.get(img["file_name"])
                if member is None:
                    missing += 1
                    continue
                frame = imdecode_bytes(zf.read(member))
                if frame is None:
                    missing += 1
                    continue
                writer.add(split, f"rw_{Path(img['file_name']).stem}", frame, anns)
            if missing:
                print(f"  [roadwork:{split}] skipped {missing} unreadable/absent images")
    return writer


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-zip", default="data/raw/images.zip")
    p.add_argument("--ann-dir", default="data/raw/annotations")
    p.add_argument("--out", default="data/processed/nav")
    p.add_argument("--classes", default="configs/classes.yaml")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--limit", type=int, default=0, help="debug: cap images per split")
    args = p.parse_args(argv)

    names, mapping = load_class_config(args.classes)
    w = convert(Path(args.images_zip), Path(args.ann_dir), Path(args.out), names, mapping,
                max_side=args.max_side, limit=args.limit)
    print(w.summary())
    w.write_yaml()
    w.write_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
