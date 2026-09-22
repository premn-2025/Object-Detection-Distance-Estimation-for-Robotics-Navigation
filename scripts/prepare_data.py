"""Build the unified navigation dataset.

    python scripts/prepare_data.py --all

Sources, and why each one is here:

  ROADWork   cones + every kind of temporary barrier, in dash-cam view.       (positives)
  COCO       the only stop-sign annotations available at scale.               (positives)
  BDD100K    stop signs mined in-domain by detector/annotation agreement,     (positives)
             plus the 10-class set used for stage-1 domain adaptation.

The three sources annotate different things, so naively concatenating them
teaches the model that a stop sign in a ROADWork frame is background.
``--pseudo-roadwork`` closes that gap; see ``src/data/pseudo_label.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import bdd100k, coco_stop_signs, pseudo_label, roadwork  # noqa: E402
from src.data.common import YoloDatasetWriter  # noqa: E402


def load_classes(path: str):
    cfg = yaml.safe_load(Path(path).read_text())
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    return names, cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/processed/nav")
    p.add_argument("--classes", default="configs/classes.yaml")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--limit", type=int, default=0, help="debug: cap images per source")

    p.add_argument("--all", action="store_true", help="every step below")
    p.add_argument("--roadwork", action="store_true")
    p.add_argument("--coco", action="store_true")
    p.add_argument("--mine-bdd", action="store_true")
    p.add_argument("--pseudo-roadwork", action="store_true")
    p.add_argument("--bdd-stage1", action="store_true")

    p.add_argument("--roadwork-zip", default="data/raw/images.zip")
    p.add_argument("--roadwork-ann", default="data/raw/annotations")
    p.add_argument("--coco-ann", default="data/raw/coco_ann.zip")
    p.add_argument("--bdd-zip", default="data/raw/bdd100k_val.zip")
    p.add_argument("--stage1-out", default="data/processed/bdd_stage1")

    p.add_argument("--pseudo-weights", default="yolo11m.pt",
                   help="COCO-pretrained detector used as the stop-sign teacher")
    p.add_argument("--pseudo-device", default="0")
    p.add_argument("--mine-conf", type=float, default=0.35)
    p.add_argument("--pseudo-conf", type=float, default=0.55)
    args = p.parse_args(argv)

    if args.all:
        args.roadwork = args.coco = args.mine_bdd = args.pseudo_roadwork = True

    names, _ = load_classes(args.classes)
    out_root = Path(args.out)
    writer = YoloDatasetWriter(out_root, names, max_side=args.max_side)
    manifest = {"classes": names, "sources": {}}

    if args.roadwork:
        print("[1] ROADWork -> cone / barrier")
        cfg = yaml.safe_load(Path(args.classes).read_text())
        rw_writer = roadwork.convert(Path(args.roadwork_zip), Path(args.roadwork_ann),
                                     out_root, names, cfg["roadwork_map"],
                                     max_side=args.max_side, limit=args.limit)
        print(rw_writer.summary())
        manifest["sources"]["roadwork"] = {f"{s}/{n}": v
                                           for (s, n), v in rw_writer.counts.items()}

    if args.coco:
        print("[2] COCO -> stop_sign")
        cfg = yaml.safe_load(Path(args.classes).read_text())
        coco_stop_signs.convert(Path(args.coco_ann), out_root, names, cfg["coco_map"],
                                max_side=args.max_side, limit=args.limit, writer=writer)
        manifest["sources"]["coco"] = {f"{s}/{n}": v for (s, n), v in writer.counts.items()}

    if args.mine_bdd:
        print("[3] BDD100K -> in-domain stop signs (detector AND annotation must agree)")
        _, mined, stats = pseudo_label.mine_bdd_stop_signs(
            Path(args.bdd_zip), out_root, names, weights=args.pseudo_weights,
            device=args.pseudo_device, conf=args.mine_conf, max_side=args.max_side,
            limit=args.limit, writer=writer)
        manifest["sources"]["bdd_mined_stop_signs"] = stats
        pseudo_label.save_manifest(out_root / "pseudo_labels_bdd.json", mined)

    if args.pseudo_roadwork:
        print("[4] completing ROADWork frames with stop-sign pseudo-labels")
        total = {}
        for split in ("train", "val"):
            img_dir = out_root / "images" / split
            lbl_dir = out_root / "labels" / split
            if not img_dir.exists():
                continue
            rw_images = sorted(img_dir.glob("rw_*.jpg"))
            if not rw_images:
                continue
            man, stats = pseudo_label.complete_roadwork_stop_signs(
                img_dir, lbl_dir, names, weights=args.pseudo_weights,
                device=args.pseudo_device, conf=args.pseudo_conf, limit=args.limit)
            # restrict to ROADWork frames only
            total[split] = stats
            pseudo_label.save_manifest(out_root / f"pseudo_labels_roadwork_{split}.json", man)
        manifest["sources"]["roadwork_pseudo_stop_signs"] = total

    if args.bdd_stage1:
        print("[5] BDD100K -> 10-class domain-adaptation set")
        w = bdd100k.convert_stage1(Path(args.bdd_zip), Path(args.stage1_out),
                                   max_side=args.max_side, limit=args.limit)
        print(w.summary())
        w.write_yaml()
        w.write_stats()

    # final descriptor + tallies, recomputed from what is actually on disk
    summary = recount(out_root, names)
    (out_root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))
    (out_root / "stats.json").write_text(json.dumps(summary, indent=2))
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\n=== dataset summary ===")
    print(json.dumps(summary, indent=2))
    return 0


def recount(root: Path, names) -> dict:
    """Recount boxes from the label files - the writers' tallies drift when
    several sources append to the same tree."""
    out = {}
    for split in ("train", "val"):
        lbl_dir = root / "labels" / split
        if not lbl_dir.exists():
            continue
        per_class = {n: 0 for n in names}
        images = negatives = 0
        by_prefix: dict = {}
        for f in lbl_dir.glob("*.txt"):
            images += 1
            prefix = f.stem.split("_")[0]
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            if not lines:
                negatives += 1
            for line in lines:
                try:
                    ci = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                if 0 <= ci < len(names):
                    per_class[names[ci]] += 1
        out[split] = {"images": images, "negative_images": negatives,
                      "boxes": per_class, "images_by_source": by_prefix}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
