"""Measure how the detector holds up in hard conditions, and on small objects.

    python scripts/eval_robustness.py --weights runs/stage2/nav_yolo11s/weights/best.pt

Produces three things:

1. **Size-stratified AP** (COCO small / medium / large). Overall mAP is dominated
   by big near objects; AP_small is the number that says whether a cone 40 m
   ahead is found while there is still time to react.
2. **A corruption sweep** — low light, glare, fog, rain, motion blur, defocus,
   sensor noise, JPEG, downscale — at chosen severities, reported as retained
   accuracy relative to clean. The corruptions come from ``src/data/degrade.py``,
   implemented independently of the training augmentation so the benchmark is not
   just measuring memorised noise.
3. **A single robustness score**: mean retained mAP50-95 across all corruptions.

Corruptions are applied in memory, so nothing is written to disk and every
variant sees byte-identical ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.degrade import CORRUPTIONS, DESCRIPTIONS, apply  # noqa: E402
from src.models.detector import build_detector  # noqa: E402
from src.optimize.cocoeval import (evaluate, gt_size_histogram,  # noqa: E402
                                   run_detector, save, yolo_labels_to_coco)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="data/processed/nav")
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001,
                   help="low, as COCO AP needs the full precision/recall curve")
    p.add_argument("--limit", type=int, default=600,
                   help="images to evaluate per condition (0 = all)")
    p.add_argument("--severities", type=int, nargs="+", default=[2, 4])
    p.add_argument("--corruptions", nargs="+", default=sorted(CORRUPTIONS))
    p.add_argument("--out", default="outputs/metrics/robustness.json")
    p.add_argument("--markdown", default="docs/robustness.md")
    p.add_argument("--classes", default=None, help="comma-separated, for ONNX weights")
    p.add_argument("--preview", action="store_true",
                   help="also write a corruption preview grid")
    args = p.parse_args(argv)

    from src.train.trainer import resolve_device

    root = Path(args.dataset)
    data_yaml = yaml.safe_load((root / "data.yaml").read_text())
    class_names = ([data_yaml["names"][i] for i in sorted(data_yaml["names"])]
                   if args.classes is None else args.classes.split(","))

    images_dir = root / "images" / args.split
    labels_dir = root / "labels" / args.split
    paths = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        step = max(1, len(paths) // args.limit)
        paths = paths[::step][:args.limit]
    print(f"[robust] {len(paths)} {args.split} images, classes={class_names}")

    print("[robust] building COCO ground truth ...")
    gt = yolo_labels_to_coco(images_dir, labels_dir, class_names, image_paths=paths)
    hist = gt_size_histogram(gt, class_names)
    print(f"[robust] GT boxes by size: {hist['overall']}")

    det = build_detector(args.weights, device=resolve_device(args.device),
                         imgsz=args.imgsz, conf=args.conf,
                         class_names=class_names if args.weights.endswith(".onnx") else None)

    if args.preview:
        import cv2

        from src.data.degrade import preview_grid

        img = cv2.imread(str(paths[0]))
        print("[robust] preview ->", preview_grid(img, 3))

    print("[robust] clean baseline ...")
    clean = evaluate(gt, run_detector(det, gt, images_dir, progress=True),
                     class_names)
    print(f"  clean mAP50-95={clean['mAP50-95']:.4f}  mAP50={clean['mAP50']:.4f}  "
          f"AP_small={clean['AP_small']:.4f}  AP_med={clean['AP_medium']:.4f}  "
          f"AP_large={clean['AP_large']:.4f}")

    rows = []
    for name in args.corruptions:
        for sev in args.severities:
            def degrade(img, _n=name, _s=sev):
                return apply(img, _n, _s)

            r = evaluate(gt, run_detector(det, gt, images_dir, degrade=degrade,
                                          progress=True), class_names)
            retained = (r["mAP50-95"] / clean["mAP50-95"]) if clean["mAP50-95"] > 0 else 0.0
            rows.append({"corruption": name, "severity": sev,
                         "description": DESCRIPTIONS.get(name, ""),
                         "mAP50-95": r["mAP50-95"], "mAP50": r["mAP50"],
                         "AP_small": r["AP_small"], "AP_large": r["AP_large"],
                         "retained": retained})
            print(f"  {name:<13} sev{sev}  mAP50-95={r['mAP50-95']:.4f}  "
                  f"retained={100*retained:5.1f}%")

    score = float(sum(r["retained"] for r in rows) / len(rows)) if rows else 0.0
    report = {"weights": args.weights, "split": args.split, "n_images": len(paths),
              "imgsz": args.imgsz, "gt_size_histogram": hist,
              "clean": clean, "corruptions": rows, "robustness_score": score}
    save(report, args.out)
    print(f"\n[robust] robustness score (mean retained mAP50-95) = {100*score:.1f}%")

    if args.markdown:
        write_markdown(report, class_names, Path(args.markdown))
        print(f"[robust] markdown -> {args.markdown}")
    print(f"[robust] json -> {args.out}")
    return 0


def write_markdown(report, class_names, path: Path) -> None:
    c = report["clean"]
    lines = [
        "# Robustness and small-object performance", "",
        f"Model: `{report['weights']}`  ·  {report['n_images']} validation images  "
        f"·  inference at {report['imgsz']}px", "",
        "Corruptions are implemented in `src/data/degrade.py` with OpenCV, "
        "**independently of the albumentations stack used for training "
        "augmentation** (`src/train/augment.py`). If test-time corruption reused "
        "the training code, this table would measure memorisation of one noise "
        "generator rather than robustness.", "",
        "## 1. Clean baseline, stratified by object size", "",
        "| metric | value |", "|---|---|",
        f"| mAP50-95 | {c['mAP50-95']:.4f} |",
        f"| mAP50 | {c['mAP50']:.4f} |",
        f"| **AP small** (<32² px) | **{c['AP_small']:.4f}** |",
        f"| AP medium (32–96² px) | {c['AP_medium']:.4f} |",
        f"| AP large (>96² px) | {c['AP_large']:.4f} |", "",
        "Ground-truth boxes by size band:", "",
        "| band | count |", "|---|---|",
    ]
    for k, v in report["gt_size_histogram"]["overall"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "AP_small is the number that matters for navigation: it is the "
              "regime of cones far enough ahead to still be avoidable. It is always "
              "much lower than the headline mAP, and any report that quotes only the "
              "headline is hiding this.", ""]

    lines += ["## 2. Per-class (clean)", "",
              "| class | mAP50-95 | mAP50 | AP small | GT boxes |", "|---|---|---|---|---|"]
    for name, v in c.get("per_class", {}).items():
        lines.append(f"| {name} | {v['mAP50-95']:.4f} | {v['mAP50']:.4f} | "
                     f"{v['AP_small']:.4f} | {v['n_gt']} |")
    lines.append("")

    lines += ["## 3. Corruption sweep", "",
              "`retained` = corrupted mAP50-95 ÷ clean mAP50-95.", "",
              "| corruption | simulates | severity | mAP50-95 | AP small | retained |",
              "|---|---|---|---|---|---|"]
    for r in sorted(report["corruptions"], key=lambda x: -x["retained"]):
        lines.append(f"| {r['corruption']} | {r['description']} | {r['severity']} | "
                     f"{r['mAP50-95']:.4f} | {r['AP_small']:.4f} | "
                     f"{100*r['retained']:.1f}% |")
    lines += ["", f"**Robustness score (mean retained mAP50-95): "
              f"{100*report['robustness_score']:.1f}%**", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
