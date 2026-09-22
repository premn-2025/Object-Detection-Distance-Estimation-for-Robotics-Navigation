"""Monocular distance estimation for navigation obstacles.

Two independent range cues are computed per detection and fused:

``size``   Z_c = fy * H_real / h_px
           Needs a physical-height prior. Degrades as the box shrinks and is
           useless when the object is cut off by the image border.

``plane``  the bottom edge of the box is assumed to lie on a horizontal plane a
           known height below the camera (the road for a cone, road + mounting
           height for a stop sign). Needs no size prior, but its error grows
           quadratically with range and it is very sensitive to camera pitch.

Both are expressed as depth along the optical axis, fused by inverse variance in
log-space (the noise of both is multiplicative), and finally back-projected to a
3-D point in the level frame so the caller also gets a lateral offset.

Derivations and the error model: docs/distance_estimation.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

from .camera import CameraModel

Box = Sequence[float]  # (x1, y1, x2, y2) in pixels


@dataclass(frozen=True)
class ObjectPrior:
    """Physical size / mounting prior for one class."""

    name: str
    height_m: Optional[float] = None
    height_sigma_m: float = 0.0
    width_m: Optional[float] = None
    width_sigma_m: Optional[float] = None
    ground_contact: bool = True
    mount_height_m: float = 0.0
    mount_height_sigma_m: float = 0.0

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ObjectPrior":
        return cls(
            name=name,
            height_m=d.get("height_m"),
            height_sigma_m=float(d.get("height_sigma_m") or 0.0),
            width_m=d.get("width_m"),
            width_sigma_m=d.get("width_sigma_m"),
            ground_contact=bool(d.get("ground_contact", True)),
            mount_height_m=float(d.get("mount_height_m") or 0.0),
            mount_height_sigma_m=float(d.get("mount_height_sigma_m") or 0.0),
        )


@dataclass
class DistanceEstimate:
    """Result for a single detection."""

    valid: bool
    distance_m: float = math.nan      # euclidean range to the base point of the object
    sigma_m: float = math.nan         # 1-sigma, from propagated input uncertainty
    forward_m: float = math.nan       # along the direction of travel
    lateral_m: float = math.nan       # +ve = right of the camera axis
    method: str = "none"              # size | plane | fused | none
    size_m: float = math.nan          # component estimates, kept for diagnostics
    size_sigma_m: float = math.nan
    plane_m: float = math.nan
    plane_sigma_m: float = math.nan
    flags: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return "--" if not self.valid else f"{self.distance_m:.1f}m"

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k, v in list(d.items()):
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


class DistanceEstimator:
    """Turns boxes into ranges. Stateless and cheap - safe to call per frame."""

    def __init__(self, camera: CameraModel, priors: dict, *,
                 edge_sigma_px: float = 2.5, min_box_height_px: float = 8.0,
                 max_range_m: float = 80.0, min_range_m: float = 0.3,
                 truncation_margin_px: float = 4.0,
                 fusion: str = "inverse_variance") -> None:
        self.camera = camera
        self.priors = priors
        self.edge_sigma_px = float(edge_sigma_px)
        self.min_box_height_px = float(min_box_height_px)
        self.max_range_m = float(max_range_m)
        self.min_range_m = float(min_range_m)
        self.truncation_margin_px = float(truncation_margin_px)
        if fusion not in {"inverse_variance", "size_only", "ground_only"}:
            raise ValueError(f"unknown fusion mode {fusion!r}")
        self.fusion = fusion

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_yaml(cls, path, camera: CameraModel = None) -> "DistanceEstimator":
        cfg = yaml.safe_load(Path(path).read_text())
        cam = camera or CameraModel.from_dict(cfg)
        priors = {k: ObjectPrior.from_dict(k, v) for k, v in cfg["priors"].items()}
        est = cfg.get("estimator", {})
        return cls(cam, priors,
                   edge_sigma_px=est.get("bbox_edge_sigma_px", 2.5),
                   min_box_height_px=est.get("min_box_height_px", 8),
                   max_range_m=est.get("max_range_m", 80.0),
                   min_range_m=est.get("min_range_m", 0.3),
                   truncation_margin_px=est.get("truncation_margin_px", 4.0),
                   fusion=est.get("fusion", "inverse_variance"))

    def for_image_size(self, width: int, height: int) -> "DistanceEstimator":
        """Clone with intrinsics rescaled to a different image resolution."""
        if (width, height) == (self.camera.image_width, self.camera.image_height):
            return self
        return DistanceEstimator(self.camera.rescaled(width, height), self.priors,
                                 edge_sigma_px=self.edge_sigma_px,
                                 min_box_height_px=self.min_box_height_px,
                                 max_range_m=self.max_range_m,
                                 min_range_m=self.min_range_m,
                                 truncation_margin_px=self.truncation_margin_px,
                                 fusion=self.fusion)

    # ------------------------------------------------------------- cue: size
    def _size_cue(self, box: Box, prior: ObjectPrior, img_h: int):
        """Return (depth_zc, relative_sigma, flags) from the apparent-height model."""
        cam = self.camera
        _, y1, _, y2 = box
        h_px = y2 - y1
        if prior.height_m is None or prior.height_m <= 0:
            return math.nan, math.nan, ["size:no_prior"]
        if h_px < self.min_box_height_px:
            return math.nan, math.nan, ["size:box_too_small"]
        m = self.truncation_margin_px
        if y1 <= m or y2 >= img_h - m:
            return math.nan, math.nan, ["size:truncated"]

        depth = cam.fy * prior.height_m / h_px
        # relative error: two independent box edges + the spread of the prior
        sig_h_px = math.sqrt(2.0) * self.edge_sigma_px
        rel = math.hypot(sig_h_px / h_px, prior.height_sigma_m / prior.height_m)
        return depth, rel, []

    # ------------------------------------------------------------ cue: plane
    def _plane_cue(self, box: Box, prior: ObjectPrior, img_h: int):
        """Return (depth_zc, relative_sigma, flags) from the ground/mounting plane."""
        cam = self.camera
        y2 = box[3]
        if not prior.ground_contact and prior.mount_height_m <= 0:
            return math.nan, math.nan, ["plane:not_applicable"]
        if y2 >= img_h - self.truncation_margin_px:
            return math.nan, math.nan, ["plane:base_cut_off"]

        # Height of the plane the box bottom sits on, measured downward from the
        # camera. Negative for objects mounted above the camera (e.g. signs).
        plane_h = cam.camera_height_m - prior.mount_height_m
        if abs(plane_h) < 0.05:
            return math.nan, math.nan, ["plane:degenerate_height"]

        fwd = cam.ground_plane_range(y2, plane_h)
        if not math.isfinite(fwd) or fwd <= 0:
            return math.nan, math.nan, ["plane:above_horizon"]

        t = cam.pitch_rad
        c, s = math.cos(t), math.sin(t)
        vn = (y2 - cam.cy) / cam.fy
        n, d = (c - vn * s), (vn * c + s)
        if abs(n) < 1e-9 or abs(d) < 1e-9:
            return math.nan, math.nan, ["plane:degenerate"]

        # analytic d(log Z)/d(param)
        d_vn = -s / n - c / d
        d_h = 1.0 / plane_h
        d_t = (-s - vn * c) / n - (c - vn * s) / d

        sig_vn = self.edge_sigma_px / cam.fy
        sig_h = math.hypot(cam.camera_height_sigma_m, prior.mount_height_sigma_m)
        sig_t = math.radians(cam.pitch_sigma_deg)
        rel = math.sqrt((d_vn * sig_vn) ** 2 + (d_h * sig_h) ** 2 + (d_t * sig_t) ** 2)

        depth = plane_h * s + fwd * c      # horizontal range -> optical-axis depth
        if depth <= 0:
            return math.nan, math.nan, ["plane:behind_camera"]
        return depth, rel, []

    # ------------------------------------------------------------------ main
    def estimate(self, box: Box, class_name: str, image_shape=None) -> DistanceEstimate:
        """Estimate the range to one detection.

        Args:
            box: ``(x1, y1, x2, y2)`` in pixels of the image the detector ran on.
            class_name: key into the prior table.
            image_shape: ``(height, width)``; if it differs from the calibration
                resolution the intrinsics are rescaled automatically.
        """
        est = self
        if image_shape is not None:
            est = self.for_image_size(int(image_shape[1]), int(image_shape[0]))
            img_h = int(image_shape[0])
        else:
            img_h = est.camera.image_height

        prior = est.priors.get(class_name)
        if prior is None:
            return DistanceEstimate(valid=False, flags=[f"no_prior:{class_name}"])

        z_size, r_size, f_size = est._size_cue(box, prior, img_h)
        z_plane, r_plane, f_plane = est._plane_cue(box, prior, img_h)
        flags = list(f_size) + list(f_plane)

        if est.fusion == "size_only":
            z_plane = math.nan
        elif est.fusion == "ground_only":
            z_size = math.nan

        use_size = math.isfinite(z_size) and z_size > 0 and math.isfinite(r_size) and r_size > 0
        use_plane = math.isfinite(z_plane) and z_plane > 0 and math.isfinite(r_plane) and r_plane > 0

        size_range = est._to_range(z_size, box, img_h)
        plane_range = est._to_range(z_plane, box, img_h)

        if not use_size and not use_plane:
            return DistanceEstimate(valid=False, method="none",
                                    flags=flags or ["no_valid_cue"],
                                    size_m=size_range, plane_m=plane_range)

        if use_size and use_plane:
            # inverse-variance fusion of log-depth
            w_s, w_p = 1.0 / (r_size ** 2), 1.0 / (r_plane ** 2)
            depth = math.exp((w_s * math.log(z_size) + w_p * math.log(z_plane)) / (w_s + w_p))
            rel = math.sqrt(1.0 / (w_s + w_p))
            method = "fused"
            # disagreement between two supposedly independent cues is a useful
            # health signal - surface it rather than hiding it in the average
            ratio = max(z_size, z_plane) / min(z_size, z_plane)
            if ratio > 1.6:
                flags.append(f"cue_disagreement:{ratio:.2f}x")
        elif use_size:
            depth, rel, method = z_size, r_size, "size"
        else:
            depth, rel, method = z_plane, r_plane, "plane"

        cam = est.camera
        u = 0.5 * (box[0] + box[2])
        v = min(box[3], img_h - 1)          # base point: bottom-centre of the box
        p = cam.backproject(u, v, depth)
        dist = float(np.linalg.norm(p))

        out = DistanceEstimate(
            valid=True,
            distance_m=min(max(dist, est.min_range_m), est.max_range_m),
            sigma_m=dist * rel,
            forward_m=float(p[2]),
            lateral_m=float(p[0]),
            method=method,
            size_m=size_range,
            size_sigma_m=size_range * r_size if use_size else math.nan,
            plane_m=plane_range,
            plane_sigma_m=plane_range * r_plane if use_plane else math.nan,
            flags=flags,
        )
        if dist > est.max_range_m:
            out.flags.append("range_saturated")
        return out

    def _to_range(self, depth_zc: float, box: Box, img_h: int) -> float:
        """Convert an optical-axis depth into a euclidean range (for reporting)."""
        if not math.isfinite(depth_zc) or depth_zc <= 0:
            return math.nan
        u = 0.5 * (box[0] + box[2])
        v = min(box[3], img_h - 1)
        return float(np.linalg.norm(self.camera.backproject(u, v, depth_zc)))

    def estimate_many(self, boxes: Iterable[Box], class_names: Iterable[str],
                      image_shape=None) -> list:
        return [self.estimate(b, c, image_shape) for b, c in zip(boxes, class_names)]
