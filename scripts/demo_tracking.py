"""Optical-flow tracking of cones across a ROADWork clip.

    python scripts/demo_tracking.py --weights runs/stage2/nav_yolo11s/weights/best.pt

Reassembles a real dash-cam sequence from the ROADWork file names, detects every
frame, associates boxes over time with IoU + Lucas-Kanade, and reports range rate
and time-to-collision per track.
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
from src.geometry.optical_flow import (FlowTracker, dense_flow,  # noqa: E402
                                       draw_tracks, flow_to_bgr)
from src.pipeline import NavigationPipeline  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--images", default="data/processed/nav/images/val")
    p.add_argument("--out", default="outputs/tracking")
    p.add_argument("--camera-config", default="configs/camera.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--sequence", type=int, default=0, help="index into the found clips")
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--min-length", type=int, default=6)
    p.add_argument("--dense-flow", action="store_true",
                   help="also render the Farneback flow field")
    p.add_argument("--video", action="store_true", help="write an mp4 as well as frames")
    args = p.parse_args(argv)

    from src.train.trainer import resolve_device

    sequences = find_sequences(args.images, min_length=args.min_length)
    if not sequences:
        print(f"[track] no multi-frame ROADWork clips under {args.images}")
        return 1
    print(f"[track] {len(sequences)} clips available; "
          f"longest = {sequences[0]['length']} frames")
    seq = sequences[min(args.sequence, len(sequences) - 1)]
    dt = frame_interval_s(seq["frame_ids"])
    print(f"[track] clip {seq['key']}  frames={seq['length']}  dt={dt*1000:.0f} ms")

    nav = NavigationPipeline.from_config(args.weights, camera_config=args.camera_config,
                                         device=resolve_device(args.device),
                                         conf=args.conf)
    tracker = FlowTracker()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    prev_gray = None
    summary = []

    frames = seq["frames"][:args.max_frames]
    for i, path in enumerate(frames):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        res = nav.process(frame)
        tracks = tracker.update(frame, res.obstacles, timestamp=i * dt)
        canvas = draw_tracks(res.annotated, tracks)

        if args.dense_flow:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = dense_flow(prev_gray, gray)
                vis = flow_to_bgr(flow)
                vis = cv2.resize(vis, (canvas.shape[1] // 3, canvas.shape[0] // 3))
                canvas[-vis.shape[0]:, -vis.shape[1]:] = vis
            prev_gray = gray

        cv2.imwrite(str(out / f"track_{i:04d}.jpg"), canvas)
        if args.video:
            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out / "tracking.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, 1.0 / dt), (w, h))
            writer.write(canvas)

    if writer is not None:
        writer.release()

    for trk in tracker.tracks:
        if trk.hits < 3:
            continue
        rate = trk.range_rate()
        ttc = trk.time_to_collision()
        dists = [d for d in trk.distances if np.isfinite(d)]
        summary.append({
            "track_id": trk.track_id, "class": trk.class_name, "hits": trk.hits,
            "first_distance_m": round(dists[0], 2) if dists else None,
            "last_distance_m": round(dists[-1], 2) if dists else None,
            "range_rate_mps": round(rate, 3) if rate is not None else None,
            "ttc_s": round(ttc, 2) if ttc is not None else None,
        })
    summary.sort(key=lambda r: (r["ttc_s"] is None, r["ttc_s"]))
    (out / "tracks.json").write_text(json.dumps(
        {"clip": seq["key"], "dt_s": dt, "tracks": summary}, indent=2))

    print(f"\n[track] {len(summary)} confirmed tracks")
    for r in summary[:12]:
        print(f"  #{r['track_id']:<3} {r['class']:<10} "
              f"{r['first_distance_m']}m -> {r['last_distance_m']}m  "
              f"rate={r['range_rate_mps']} m/s  ttc={r['ttc_s']} s")
    print(f"[track] frames + tracks.json -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
