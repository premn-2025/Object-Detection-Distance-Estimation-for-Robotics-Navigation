"""COCO 2017 -> the ``stop_sign`` class.

ROADWork has no stop-sign category (its only sign classes are temporary work-zone
signage), so the third class comes from COCO, which annotates ``stop sign`` in
~1.7 k train and ~75 val images.

Only the images that actually contain a stop sign are fetched - roughly 250 MB
instead of the 19 GB full ``train2017`` archive.

Known domain gap: COCO stop-sign photos are web imagery, not dash-cam frames.
That is partly why ``pseudo_label.py`` exists - it transfers the stop-sign
knowledge of the COCO-pretrained detector onto in-domain ROADWork frames.
"""

from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from .common import Annotation, YoloDatasetWriter, imdecode_bytes

IMAGE_URL = "http://images.cocodataset.org/{split}/{file_name}"
ANN_MEMBERS = {
    "train": "annotations/instances_train2017.json",
    "val": "annotations/instances_val2017.json",
}
# COCO's own split -> the split we put it in.  COCO val2017 is only ~75 stop-sign
# images, which is too few to validate on alone, so it becomes our val set and
# COCO train2017 feeds our train set.
SPLIT_DIR = {"train": "train2017", "val": "val2017"}


def _fetch(session: requests.Session, split: str, file_name: str, cache: Path, retries: int = 4):
    dst = cache / file_name
    if dst.exists() and dst.stat().st_size > 0:
        return dst.read_bytes()
    url = IMAGE_URL.format(split=SPLIT_DIR[split], file_name=file_name)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                dst.write_bytes(r.content)
                return r.content
        except requests.RequestException:
            pass
    return None


def convert(ann_zip: Path, out_root: Path, class_names, coco_map: dict, *,
            max_side: int = 1280, min_box_px: float = 6.0, workers: int = 16,
            cache_dir: Path = Path("data/interim/coco_images"), limit: int = 0,
            writer: YoloDatasetWriter = None) -> YoloDatasetWriter:
    """Add COCO stop-sign images to (optionally an existing) YOLO dataset."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    writer = writer or YoloDatasetWriter(out_root, class_names, max_side=max_side)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ann_zip) as zf, requests.Session() as session:
        for split, member in ANN_MEMBERS.items():
            coco = json.loads(zf.read(member))
            wanted = {c["id"]: coco_map[c["name"]] for c in coco["categories"]
                      if c["name"] in coco_map}
            if not wanted:
                raise RuntimeError(f"none of {list(coco_map)} present in COCO categories")

            by_image: dict = {}
            for a in coco["annotations"]:
                if a.get("iscrowd") or a["category_id"] not in wanted:
                    continue
                x, y, w, h = a["bbox"]
                if w < min_box_px or h < min_box_px:
                    continue
                by_image.setdefault(a["image_id"], []).append(
                    Annotation(name_to_idx[wanted[a["category_id"]]], x, y, x + w, y + h))

            imgs = [i for i in coco["images"] if i["id"] in by_image]
            if limit:
                imgs = imgs[:limit]
            print(f"  [coco:{split}] {len(imgs)} images carrying "
                  f"{sum(len(v) for v in by_image.values())} boxes")

            def work(img):
                return img, _fetch(session, split, img["file_name"], cache_dir)

            failed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for img, raw in tqdm(pool.map(work, imgs), total=len(imgs),
                                     desc=f"coco:{split}", unit="img"):
                    if raw is None:
                        failed += 1
                        continue
                    frame = imdecode_bytes(raw)
                    if frame is None:
                        failed += 1
                        continue
                    writer.add(split, f"coco_{Path(img['file_name']).stem}",
                               frame, by_image[img["id"]])
            if failed:
                print(f"  [coco:{split}] {failed} images could not be downloaded")
    return writer


def main(argv=None) -> int:
    import argparse

    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann-zip", default="data/raw/coco_ann.zip")
    p.add_argument("--out", default="data/processed/nav")
    p.add_argument("--classes", default="configs/classes.yaml")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.classes).read_text())
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    w = convert(Path(args.ann_zip), Path(args.out), names, cfg["coco_map"],
                max_side=args.max_side, limit=args.limit)
    print(w.summary())
    w.write_yaml()
    w.write_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
