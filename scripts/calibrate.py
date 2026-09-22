"""Fit camera height and pitch from labelled cones, then validate honestly.

    python scripts/calibrate.py --dataset data/processed/nav --write

Fits on the *train* cones, reports the residual on *val* cones, and separately
reports agreement on *barriers*, which never enter the fit. See
docs/distance_estimation.md section 6 for why focal length cannot be recovered
this way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distance.calibrate import (agreement_report, fit_extrinsics,  # noqa: E402
                                    height_sensitivity, load_samples_from_yolo,
                                    save_report)
from src.distance.camera import CameraModel  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/processed/nav")
    p.add_argument("--camera-config", default="configs/camera.yaml")
    p.add_argument("--fit-class", default="cone")
    p.add_argument("--check-class", default="barrier")
    p.add_argument("--min-box-px", type=float, default=100.0,
                   help="small boxes bias the fit upward; see height_sensitivity")
    p.add_argument("--min-base-row-frac", type=float, default=0.78,
                   help="require the object base well below the horizon")
    p.add_argument("--aspect", type=float, default=16 / 9,
                   help="restrict the fit to one camera, by image aspect ratio")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--camera-height", type=float, default=None,
                   help="known mounting height in metres; overrides the estimate")
    p.add_argument("--min-sensitivity-n", type=int, default=300,
                   help="smallest sample count trusted in the sensitivity table")
    p.add_argument("--out", default="outputs/calibration/report.json")
    p.add_argument("--write", action="store_true",
                   help="write the fitted extrinsics back into the camera config")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.camera_config).read_text())
    cam = CameraModel.from_dict(cfg)
    names = list(cfg["priors"])
    data_yaml = yaml.safe_load((Path(args.dataset) / "data.yaml").read_text())
    class_names = [data_yaml["names"][i] for i in sorted(data_yaml["names"])]

    fit_idx = class_names.index(args.fit_class)
    fit_h = cfg["priors"][args.fit_class]["height_m"]
    check_idx = class_names.index(args.check_class) if args.check_class in class_names else None

    root = Path(args.dataset)

    def samples(split, idx, height):
        return load_samples_from_yolo(root / "labels" / split, root / "images" / split,
                                      idx, height, min_h_px=args.min_box_px,
                                      limit=args.limit, aspect=args.aspect,
                                      stem_prefix="rw_",
                                      min_base_row_frac=args.min_base_row_frac)

    print(f"[calib] loading {args.fit_class} boxes (aspect {args.aspect:.3f} only) ...")
    train, size_tr = samples("train", fit_idx, fit_h)
    val, size_va = samples("val", fit_idx, fit_h)
    print(f"[calib] fit samples: train={len(train)} val={len(val)}  "
          f"image size {size_tr}")

    # The boxes are in the *stored* resolution, not the calibration resolution,
    # so the intrinsics must be rescaled before they can be compared.
    if size_tr:
        cam = cam.rescaled(*size_tr)
        print(f"[calib] intrinsics rescaled to {size_tr}: "
              f"fx={cam.fx:.1f} cy={cam.cy:.1f}")

    sens = height_sensitivity(root / "labels" / "train", root / "images" / "train",
                              fit_idx, fit_h, CameraModel.from_dict(cfg),
                              aspect=args.aspect)
    # Height and pitch are degenerate against each other, so height is taken from
    # the best-conditioned subset that still has enough samples to be stable, and
    # only pitch is fitted. See fit_extrinsics for the argument.
    usable = [s for s in sens if s["n"] >= args.min_sensitivity_n]
    fixed_h = args.camera_height or (usable[-1]["implied_height_m_median"] if usable else None)
    if fixed_h:
        src = "--camera-height" if args.camera_height else (
            f"height_sensitivity row box>={usable[-1]['min_box_px']}px "
            f"(n={usable[-1]['n']})")
        print(f"[calib] fixing camera height at {fixed_h:.2f} m from {src}; "
              f"fitting pitch only")

    report = fit_extrinsics(train, cam, val_samples=val,
                            height_bounds=(0.6, 3.0), fixed_height=fixed_h)
    report["height_sensitivity"] = sens
    report["conditioning_gate"] = {"min_box_px": args.min_box_px,
                                   "min_base_row_frac": args.min_base_row_frac}
    fitted = report["fitted"]
    print(json.dumps(report, indent=2))

    cam_fitted = CameraModel.from_dict({
        "intrinsics": {**cfg["intrinsics"],
                       "image_width": cam.image_width, "image_height": cam.image_height,
                       "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy},
        "extrinsics": {**cfg["extrinsics"],
                       "camera_height_m": fitted["camera_height_m"],
                       "pitch_deg": fitted["pitch_deg"]},
    })

    if check_idx is not None:
        ch = cfg["priors"][args.check_class]["height_m"]
        held_out, _ = samples("val", check_idx, ch)
        report["held_out_class"] = {
            "class": args.check_class,
            "before": agreement_report(held_out, cam),
            "after": agreement_report(held_out, cam_fitted),
            "note": ("this class never enters the fit, so its agreement ratio is "
                     "the only non-circular check available"),
        }
        print(f"\n[calib] held-out {args.check_class}: "
              f"{json.dumps(report['held_out_class'], indent=2)}")

    save_report(report, args.out)
    print(f"\n[calib] report -> {args.out}")

    if args.write:
        cfg["extrinsics"]["camera_height_m"] = round(fitted["camera_height_m"], 4)
        cfg["extrinsics"]["pitch_deg"] = round(fitted["pitch_deg"], 4)
        cfg["intrinsics"]["source"] = "auto_calibrated"
        Path(args.camera_config).write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"[calib] wrote extrinsics into {args.camera_config}")
        print("        (fx is unchanged and still rests on the FOV assumption)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
