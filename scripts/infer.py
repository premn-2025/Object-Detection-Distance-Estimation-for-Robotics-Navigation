"""Run detection + distance estimation on images, a folder, or a video.

    python scripts/infer.py --weights runs/stage2/nav_yolo11s/weights/best.pt \
                            --source data/processed/nav/images/val --limit 12
    python scripts/infer.py --weights ... --source clip.mp4 --out outputs/clip.mp4 --bev

Every annotation is ``<Class>, <distance>m`` as specified; ``--sigma`` appends the
propagated uncertainty and ``--method`` shows which cue produced the number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import NavigationPipeline  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def collect_images(source: Path, limit: int = 0, pattern: str = "*"):
    if source.is_dir():
        files = sorted(p for p in source.glob(pattern) if p.suffix.lower() in IMAGE_EXT)
    else:
        files = [source]
    return files[:limit] if limit else files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="outputs/predictions")
    p.add_argument("--camera-config", default="configs/camera.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pattern", default="*",
                   help="glob within the source dir, e.g. rw_* for ROADWork dash-cam frames (the only ones with valid camera geometry)")
    p.add_argument("--stride", type=int, default=1, help="video: process every Nth frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--sigma", action="store_true", help="show +/- uncertainty")
    p.add_argument("--method", action="store_true", help="show which cue was used")
    p.add_argument("--score", action="store_true", help="show detector confidence")
    p.add_argument("--bev", action="store_true", help="attach a bird's-eye panel")
    p.add_argument("--rulers", action="store_true", help="overlay ground-plane range rulers")
    p.add_argument("--classes", default=None, help="comma-separated names for ONNX weights")
    p.add_argument("--json", action="store_true", help="also write per-frame json")
    args = p.parse_args(argv)

    from src.train.trainer import resolve_device

    device = resolve_device(args.device)
    class_names = args.classes.split(",") if args.classes else None

    nav = NavigationPipeline.from_config(
        args.weights, camera_config=args.camera_config, device=device,
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, class_names=class_names,
        show_sigma=args.sigma, show_method=args.method, show_score=args.score,
        with_bev=args.bev)

    source = Path(args.source)
    out = Path(args.out)

    if source.suffix.lower() in VIDEO_EXT:
        out.parent.mkdir(parents=True, exist_ok=True)
        dst = out if out.suffix else out / "annotated.mp4"
        results = nav.process_video(source, dst, max_frames=args.max_frames,
                                    stride=args.stride)
        n = sum(len(r.obstacles) for r in results)
        print(f"[infer] {len(results)} frames -> {dst}  ({n} obstacles)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    files = collect_images(source, args.limit, args.pattern)
    if not files:
        print(f"[infer] nothing to do: {source}")
        return 1

    records = []
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        if args.rulers:
            from src.viz import draw_range_rulers

            frame = draw_range_rulers(frame, nav.camera)
        res = nav.process(frame)
        cv2.imwrite(str(out / f.name), res.annotated)
        rec = {"image": f.name, "detect_ms": round(res.detect_ms, 2),
               "range_ms": round(res.range_ms, 3),
               "obstacles": [{"annotation": o.annotation, "class": o.class_name,
                              "score": round(o.score, 3),
                              "distance_m": None if o.distance_m != o.distance_m
                              else round(o.distance_m, 2),
                              "sigma_m": None if o.sigma_m != o.sigma_m
                              else round(o.sigma_m, 2),
                              "forward_m": None if o.forward_m != o.forward_m
                              else round(o.forward_m, 2),
                              "lateral_m": None if o.lateral_m != o.lateral_m
                              else round(o.lateral_m, 2),
                              "method": o.range_method, "flags": o.flags}
                             for o in res.obstacles]}
        records.append(rec)
        labels = ", ".join(o.annotation for o in res.obstacles[:5]) or "(nothing)"
        print(f"  {f.name}: {labels}")

    if args.json:
        (out / "predictions.json").write_text(json.dumps(records, indent=2))
    print(f"[infer] {len(records)} images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
