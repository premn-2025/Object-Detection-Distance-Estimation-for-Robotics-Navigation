"""Bird's-eye view demo: perspective frame + metric IPM raster side by side.

    python scripts/demo_bev.py --weights runs/stage2/nav_yolo11s/weights/best.pt \
                               --source data/processed/nav/images/val --limit 8

The homography comes from the camera model, not from clicked points, so the
bird's-eye canvas is in metres and agrees with the reported distances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.bev import BEVTransform  # noqa: E402
from src.pipeline import NavigationPipeline  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default="data/processed/nav/images/val")
    p.add_argument("--out", default="outputs/bev")
    p.add_argument("--camera-config", default="configs/camera.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--x-range", type=float, nargs=2, default=(-10.0, 10.0))
    p.add_argument("--z-range", type=float, nargs=2, default=(1.0, 40.0))
    p.add_argument("--px-per-m", type=float, default=12.0)
    args = p.parse_args(argv)

    from src.train.trainer import resolve_device

    nav = NavigationPipeline.from_config(args.weights, camera_config=args.camera_config,
                                         device=resolve_device(args.device),
                                         conf=args.conf)
    bev = BEVTransform(nav.camera, x_range=tuple(args.x_range),
                       z_range=tuple(args.z_range), px_per_m=args.px_per_m)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)
    files = sorted(src.glob(args.pattern))[:args.limit] if src.is_dir() else [src]

    print(f"[bev] H_image_from_ground =\n{np.round(bev.H_image_from_ground(), 3)}")
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        res = nav.process(frame)
        warped = bev.warp(frame)
        warped = bev.draw_grid(warped)
        warped = bev.draw_objects(warped, res.obstacles)
        warped = bev.draw_robot(warped)

        h = res.annotated.shape[0]
        scale = h / warped.shape[0]
        panel = cv2.resize(warped, (int(warped.shape[1] * scale), h))
        combo = np.hstack([res.annotated, panel])
        cv2.imwrite(str(out / f"bev_{f.name}"), combo)
        names = ", ".join(o.annotation for o in res.obstacles[:6]) or "(nothing)"
        print(f"  {f.name}: {names}")

    print(f"[bev] wrote {len(files)} composites -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
