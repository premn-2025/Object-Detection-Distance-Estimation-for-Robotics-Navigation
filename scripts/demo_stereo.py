"""Epipolar geometry demo: motion stereo on a monocular dash-cam clip.

    python scripts/demo_stereo.py --weights runs/stage2/nav_yolo11s/weights/best.pt

Three things happen here:

1. the disparity-depth trade-off table is printed for a few candidate stereo
   rigs, which is the design calculation you do *before* buying a camera;
2. two consecutive frames of a real clip are treated as a stereo pair: features
   are matched, the essential matrix recovered, and points triangulated;
3. the scale-free triangulated depths are tied to metres using one detected cone
   whose size prior gives an absolute range - then the *other* detections are
   compared against the triangulated depth, which is an independent check on the
   monocular estimator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sequences import find_sequences, frame_interval_s  # noqa: E402
from src.geometry.epipolar import (epipolar_lines, estimate_relative_pose,  # noqa: E402
                                   motion_stereo_depth, stereo_range_table,
                                   symmetric_epipolar_error, triangulate)
from src.pipeline import NavigationPipeline  # noqa: E402


def draw_epipolar(img1, img2, pose, n: int = 14):
    """Side-by-side view with matched points and their epipolar lines."""
    out1, out2 = img1.copy(), img2.copy()
    idx = np.linspace(0, len(pose.pts1) - 1, min(n, len(pose.pts1))).astype(int)
    lines2 = epipolar_lines(pose.F, pose.pts1[idx], which=1)
    rng = np.random.default_rng(0)
    w = out2.shape[1]
    for (x, y), (a, b, c) in zip(pose.pts1[idx], lines2):
        col = tuple(int(v) for v in rng.integers(60, 255, 3))
        cv2.circle(out1, (int(x), int(y)), 5, col, -1)
        if abs(b) > 1e-6:
            p0 = (0, int(-c / b))
            p1 = (w, int(-(c + a * w) / b))
            cv2.line(out2, p0, p1, col, 1, cv2.LINE_AA)
    for (x, y) in pose.pts2[idx]:
        cv2.circle(out2, (int(x), int(y)), 4, (255, 255, 255), 1)
    return np.hstack([out1, out2])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None,
                   help="optional: cross-check detections against triangulation")
    p.add_argument("--images", default="data/processed/nav/images/val")
    p.add_argument("--out", default="outputs/stereo")
    p.add_argument("--camera-config", default="configs/camera.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--sequence", type=int, default=0)
    p.add_argument("--gap", type=int, default=1, help="frames between the stereo pair")
    p.add_argument("--speed-mps", type=float, default=None,
                   help="known ego speed -> metric baseline, instead of cue-based scale")
    args = p.parse_args(argv)

    from src.distance.camera import CameraModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------------------------------------------------------------- 1. table
    print("=== stereo design trade-off:  Z = f*B/d,  sigma_Z = Z^2*sigma_d/(f*B) ===")
    rigs = [("narrow 6 cm", 6.0e-2), ("typical 12 cm", 12.0e-2), ("wide 30 cm", 30.0e-2)]
    f_px = 700.0
    table = {}
    for label, B in rigs:
        rows = stereo_range_table(f_px, B)
        table[label] = rows
        print(f"\n  {label} baseline, f={f_px:.0f}px")
        print("    range(m)  disparity(px)  sigma(m)  rel.err")
        for r in rows:
            print(f"    {r['range_m']:7.0f}  {r['disparity_px']:12.2f}  "
                  f"{r['sigma_m']:8.3f}  {r['relative_error']*100:6.1f}%")
    report["design_table"] = table

    # -------------------------------------------------------- 2. motion stereo
    sequences = find_sequences(args.images, min_length=args.gap + 2)
    if not sequences:
        print(f"\n[stereo] no multi-frame clips under {args.images}; stopping after table")
        (out / "stereo_report.json").write_text(json.dumps(report, indent=2, default=float))
        return 0

    seq = sequences[min(args.sequence, len(sequences) - 1)]
    dt = frame_interval_s(seq["frame_ids"]) * args.gap
    f1 = cv2.imread(str(seq["frames"][0]))
    f2 = cv2.imread(str(seq["frames"][args.gap]))
    print(f"\n=== motion stereo on clip {seq['key']} (gap {args.gap}, dt {dt*1000:.0f} ms) ===")

    cam = CameraModel.from_yaml(args.camera_config).rescaled(f1.shape[1], f1.shape[0])
    K = cam.K

    try:
        pose = estimate_relative_pose(f1, f2, K)
    except RuntimeError as exc:
        print(f"[stereo] pose estimation failed: {exc}")
        (out / "stereo_report.json").write_text(json.dumps(report, indent=2, default=float))
        return 0

    err = symmetric_epipolar_error(pose.F, pose.pts1, pose.pts2)
    print(f"  inliers={pose.n_inliers}  median epipolar error={np.median(err):.2f} px")
    print(f"  unit translation t = {np.round(pose.t.ravel(), 3)}  "
          f"(forward-dominant means the camera drove forward)")
    cv2.imwrite(str(out / "epipolar_pair.jpg"), draw_epipolar(f1, f2, pose))

    baseline = args.speed_mps * dt if args.speed_mps else None
    ms = motion_stereo_depth(f1, f2, K, baseline_m=baseline)
    report["motion_stereo"] = {
        "clip": seq["key"], "gap_frames": args.gap, "dt_s": dt,
        "inliers": int(ms["n_inliers"]),
        "median_epipolar_error_px": ms["median_epipolar_error_px"],
        "metric": bool(baseline), "baseline_m": baseline,
        "translation_unit": [float(v) for v in pose.t.ravel()],
    }

    # --------------------------------------------- 3. cross-check the monocular
    if args.weights:
        from src.train.trainer import resolve_device

        nav = NavigationPipeline.from_config(args.weights, camera_config=args.camera_config,
                                             device=resolve_device(args.device), conf=0.3)
        res = nav.process(f1)
        pts = ms["points_2d"]
        depths = ms["depth"]
        rows = []
        for obs in res.obstacles:
            x1, y1, x2, y2 = obs.box
            inside = ((pts[:, 0] >= x1) & (pts[:, 0] <= x2) &
                      (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
            if inside.sum() < 3:
                continue
            rows.append({"class": obs.class_name,
                         "monocular_m": round(float(obs.distance_m), 2),
                         "triangulated_rel": round(float(np.median(depths[inside])), 4),
                         "n_points": int(inside.sum())})
        if len(rows) >= 2:
            # tie the scale-free reconstruction to metres with the nearest object,
            # then see whether the *others* land where the monocular model says
            anchor = min(rows, key=lambda r: r["monocular_m"])
            scale = anchor["monocular_m"] / anchor["triangulated_rel"]
            for r in rows:
                r["triangulated_m"] = round(r["triangulated_rel"] * scale, 2)
                r["ratio"] = round(r["triangulated_m"] / r["monocular_m"], 3)
            print("\n=== monocular vs motion-stereo (scale anchored on the nearest object) ===")
            print("  class        monocular   triangulated   ratio")
            for r in rows:
                print(f"  {r['class']:<11} {r['monocular_m']:>8.1f} m "
                      f"{r['triangulated_m']:>12.1f} m {r['ratio']:>8.2f}")
            print("  (the anchor is 1.00 by construction; the others are the test)")
            report["cross_check"] = {"anchor_class": anchor["class"],
                                     "scale_factor": scale, "objects": rows}
        else:
            print("\n[stereo] too few detections with enough matched features to cross-check")
            report["cross_check"] = {"objects": rows,
                                     "note": "insufficient textured detections"}
        cv2.imwrite(str(out / "detections_frame1.jpg"), res.annotated)

    (out / "stereo_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[stereo] outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
