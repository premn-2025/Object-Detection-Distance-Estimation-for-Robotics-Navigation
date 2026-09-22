"""Recovering camera pose from labelled boxes, with no calibration rig.

The datasets used here ship no intrinsics or extrinsics, and there is no ground
truth distance anywhere in them, so nothing can be fitted by regressing against
true range.  What *is* available is a consistency constraint:

    a cone of known physical height, standing on the road, must give the same
    range through the apparent-size model and through the ground-plane model.

Write that as a residual over the labelled cones and minimise it.

**What this can and cannot identify.**  At small pitch the size model gives
``Z = fy*H/h`` and the plane model gives ``Z = h_cam*fy/(v - cy)``.  Both are
proportional to ``fy``, so ``fy`` cancels in the ratio and is *not* recoverable
this way - it still has to come from the lens FOV, EXIF, or a checkerboard.  The
ratio does pin down the camera height, and the way the residual varies with
image row pins down the pitch.  So this fits ``(camera_height, pitch)`` only, and
says so.

**The obvious objection.**  After fitting, the two cues agree by construction, so
their agreement is no longer independent evidence that either is right.  That is
why :func:`fit_extrinsics` fits on one split and reports the residual on a
held-out split, and why ``scripts/calibrate.py`` also reports agreement on the
*barrier* class, which was never used in the fit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CameraModel


@dataclass
class CalibrationSample:
    """One labelled, ground-contacting object."""

    v_bottom: float      # image row of the base, pixels
    h_px: float          # box height, pixels
    height_m: float      # physical height prior for its class


def load_samples_from_yolo(labels_dir, images_dir, class_index: int, height_m: float,
                           *, min_h_px: float = 100.0, margin_px: float = 6.0,
                           limit: int = 0, aspect: float = None,
                           aspect_tol: float = 0.03, stem_prefix: str = None,
                           min_base_row_frac: float = 0.78) -> tuple:
    """Read YOLO labels and return ``(samples, modal_image_size)``.

    Boxes touching the frame edge are dropped: a truncated box has neither a
    trustworthy height nor a visible base.

    ``aspect`` filters to one camera.  This matters more than it looks: the
    assembled dataset contains at least three different cameras - ROADWork's
    16:9 dash-cam video, ROADWork's 4:3 hand-held stills, and COCO web photos
    with no consistent optics at all.  A single pinhole model cannot describe all
    of them, so fitting over the mixture produces a meaningless average.  Only
    the dash-cam geometry is the one a robot would have.

    ``min_h_px`` and ``min_base_row_frac`` are not cosmetic filters; they remove a
    real bias.  The implied height is ``H*(v - cy)/h_px``, so any systematic
    *under*-measurement of ``h_px`` inflates it, and that error is multiplicative
    in ``1/h_px`` - it blows up as boxes shrink.  Tight annotation, blur and the
    downscale to 1280px all shave a pixel or two off every box.  Fitted over all
    11.9 k cones (median box: 43 px) the estimate comes out at 2.2 m; restricted
    to well-conditioned cones (>=120 px, near the bottom of the frame) it settles
    at ~1.2-1.4 m, which is an actual dash-cam height.  Classic errors-in-
    variables attenuation, and :func:`height_sensitivity` reports the whole curve
    rather than hiding the choice.
    """
    from collections import Counter

    from PIL import Image

    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    samples = []
    sizes = Counter()
    files = sorted(labels_dir.glob(f"{stem_prefix}*.txt" if stem_prefix else "*.txt"))
    if limit:
        files = files[:limit]
    for lf in files:
        txt = lf.read_text().strip()
        if not txt:
            continue
        img = images_dir / f"{lf.stem}.jpg"
        try:
            # header only: decoding 12 k full frames just to read their size
            # exhausts memory while a training run holds the rest of it
            with Image.open(img) as im:
                W, H = im.size
        except (OSError, ValueError):
            continue
        if aspect is not None and abs(W / H - aspect) > aspect_tol:
            continue
        sizes[(W, H)] += 1
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != class_index:
                continue
            _, cx, cy, bw, bh = (float(p) for p in parts)
            x1, x2 = (cx - bw / 2) * W, (cx + bw / 2) * W
            y1, y2 = (cy - bh / 2) * H, (cy + bh / 2) * H
            h_px = y2 - y1
            if h_px < min_h_px:
                continue
            if y2 < min_base_row_frac * H:
                continue          # base too close to the horizon -> ill-conditioned
            if y1 <= margin_px or y2 >= H - margin_px or x1 <= margin_px or x2 >= W - margin_px:
                continue
            samples.append(CalibrationSample(v_bottom=y2, h_px=h_px, height_m=height_m))
    modal = sizes.most_common(1)[0][0] if sizes else None
    return samples, modal


def height_sensitivity(labels_dir, images_dir, class_index: int, height_m: float,
                       cam: CameraModel, *, aspect: float = 16 / 9,
                       stem_prefix: str = "rw_") -> list:
    """Implied camera height as the sample set is restricted to better cues.

    Published alongside the fit so a reader can see that the number depends on
    which cones are used, and why the well-conditioned end is the trustworthy
    one. One pass over the labels, thresholds applied in memory.
    """
    samples, size = load_samples_from_yolo(labels_dir, images_dir, class_index,
                                           height_m, min_h_px=1.0, aspect=aspect,
                                           stem_prefix=stem_prefix,
                                           min_base_row_frac=0.0)
    if not samples or size is None:
        return []
    c = cam.rescaled(*size)
    h_px = np.array([s.h_px for s in samples])
    v_b = np.array([s.v_bottom for s in samples])
    implied = height_m * (v_b - c.cy) / h_px

    rows = []
    for min_h, min_frac in ((0, 0.0), (40, 0.0), (60, 0.62), (80, 0.76),
                            (120, 0.83), (160, 0.86)):
        m = (h_px >= min_h) & (v_b >= min_frac * size[1])
        if m.sum() < 25:
            continue
        q = np.percentile(implied[m], [25, 50, 75])
        rows.append({"min_box_px": min_h, "min_base_row_frac": min_frac,
                     "n": int(m.sum()), "implied_height_m_median": float(q[1]),
                     "iqr": [float(q[0]), float(q[2])]})
    return rows


def _log_ratio(samples, cam: CameraModel, camera_height: float, pitch_deg: float) -> np.ndarray:
    """log(Z_size / Z_plane) for every sample, in optical-axis depth."""
    t = math.radians(pitch_deg)
    c, s = math.cos(t), math.sin(t)
    v = np.array([sm.v_bottom for sm in samples])
    h_px = np.array([sm.h_px for sm in samples])
    H = np.array([sm.height_m for sm in samples])

    z_size = cam.fy * H / h_px
    vn = (v - cam.cy) / cam.fy
    num, den = camera_height * (c - vn * s), (vn * c + s)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = num / den
        z_plane = camera_height * s + fwd * c
    bad = ~np.isfinite(z_plane) | (z_plane <= 1e-6)
    z_plane = np.where(bad, np.nan, z_plane)
    return np.log(z_size) - np.log(z_plane)


def _huber(r: np.ndarray, delta: float = 0.25) -> float:
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1e9
    a = np.abs(r)
    quad = np.minimum(a, delta)
    return float(np.sum(0.5 * quad ** 2 + delta * (a - quad)) / r.size)


def fit_extrinsics(train_samples, cam: CameraModel, *, val_samples=None,
                   height_bounds=(0.4, 3.0), pitch_bounds=(-12.0, 12.0),
                   huber_delta: float = 0.25, fixed_height: float = None) -> dict:
    """Fit camera pose by making the two range cues agree.

    ``fixed_height`` fits pitch alone. Use it: height and pitch are strongly
    degenerate here. Pitching the camera further down raises the horizon, which
    shrinks the implied range for a given image row, which is then compensated by
    a larger camera height - so the pair slides along a valley in the objective
    and the joint fit runs to whatever bound it is given (2.5 m and 10 deg, on
    this data). Neither number is separately identified.

    :func:`height_sensitivity` breaks the tie from outside the fit: restricted to
    well-conditioned cones the implied height converges on ~1.2-1.4 m, which is a
    real dash-cam mounting height. Fixing height there and solving for pitch
    alone gives one well-posed parameter instead of two badly-posed ones.
    """
    from scipy.optimize import minimize

    if len(train_samples) < 50:
        raise ValueError(f"only {len(train_samples)} calibration samples; need >= 50")

    def objective(p):
        h = float(fixed_height) if fixed_height else float(p[0])
        pitch = float(p[1])
        if not (height_bounds[0] <= h <= height_bounds[1]):
            return 1e9
        if not (pitch_bounds[0] <= pitch <= pitch_bounds[1]):
            return 1e9
        return _huber(_log_ratio(train_samples, cam, h, pitch), huber_delta)

    # closed-form start for the height (exact when pitch == 0), then refine both
    v = np.array([s.v_bottom for s in train_samples])
    h_px = np.array([s.h_px for s in train_samples])
    Hm = np.array([s.height_m for s in train_samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        h0 = np.nanmedian(Hm * (v - cam.cy) / h_px)
    h0 = float(fixed_height) if fixed_height else float(np.clip(h0, *height_bounds))

    best = None
    for pitch0 in (-4.0, 0.0, 4.0):
        res = minimize(objective, x0=np.array([h0, pitch0]), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res

    h_fit = float(fixed_height) if fixed_height else float(best.x[0])
    pitch_fit = float(best.x[1])

    def report(samples, h, p):
        if not samples:
            return None
        r = _log_ratio(samples, cam, h, p)
        r = r[np.isfinite(r)]
        return {
            "n": int(r.size),
            "median_abs_log_ratio": float(np.median(np.abs(r))),
            "median_ratio": float(np.exp(np.median(r))),
            "p90_abs_log_ratio": float(np.percentile(np.abs(r), 90)),
        }

    return {
        "fitted": {"camera_height_m": h_fit, "pitch_deg": pitch_fit},
        "free_parameters": ["pitch_deg"] if fixed_height else ["camera_height_m", "pitch_deg"],
        "height_source": ("fixed from height_sensitivity (degenerate with pitch)"
                          if fixed_height else "fitted jointly"),
        "seed": {"camera_height_m": h0, "pitch_deg": 0.0},
        "objective": float(best.fun),
        "train_before": report(train_samples, cam.camera_height_m, cam.pitch_deg),
        "train_after": report(train_samples, h_fit, pitch_fit),
        "val_before": report(val_samples, cam.camera_height_m, cam.pitch_deg),
        "val_after": report(val_samples, h_fit, pitch_fit),
        "note": ("fy is not identifiable from cue agreement (it cancels in the "
                 "ratio); it stays at its FOV-derived value"),
    }


def agreement_report(samples, cam: CameraModel) -> dict:
    """Cue agreement for a class that was *not* used in the fit."""
    r = _log_ratio(samples, cam, cam.camera_height_m, cam.pitch_deg)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0}
    return {
        "n": int(r.size),
        "median_ratio": float(np.exp(np.median(r))),
        "median_abs_log_ratio": float(np.median(np.abs(r))),
        "p90_abs_log_ratio": float(np.percentile(np.abs(r), 90)),
    }


def pitch_from_horizon(cam: CameraModel, horizon_row: float) -> float:
    """Pitch implied by a hand-marked horizon row, in degrees."""
    return math.degrees(math.atan((cam.cy - horizon_row) / cam.fy))


def calibrate_checkerboard(image_paths, pattern=(9, 6), square_size_m: float = 0.025) -> dict:
    """Standard intrinsic calibration - the procedure to use on the real robot.

    Nothing in this repo can run it (the public datasets have no calibration
    images), but a deployed robot should: the FOV assumption is the largest
    single source of bias in every distance reported here.
    """
    import cv2

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size_m

    obj_points, img_points, shape = [], [], None
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < 5:
        raise RuntimeError(f"only {len(obj_points)} usable views; need >= 5")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, shape, None, None)
    return {
        "rms_reprojection_px": float(rms),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "dist_coeffs": [float(v) for v in dist.ravel()],
        "n_views": len(obj_points),
        "image_size": list(shape),
    }


def save_report(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
