"""Pinhole camera model with a ground-plane pose.

Conventions (OpenCV):
    camera frame C :  X right, Y down, Z along the optical axis
    level  frame L :  X right, Y down, Z forward and horizontal
The two differ by the camera pitch ``theta`` (positive = nose down).  The road
is the plane ``Y_L = camera_height``.

Projection of a camera-frame point::

    u = fx * X_C / Z_C + cx
    v = fy * Y_C / Z_C + cy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics + pose of the camera relative to the ground plane."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    camera_height_m: float = 1.30
    pitch_deg: float = 0.0
    camera_height_sigma_m: float = 0.15
    pitch_sigma_deg: float = 1.5
    source: str = "fov_assumption"

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraModel":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> "CameraModel":
        intr, extr = cfg["intrinsics"], cfg.get("extrinsics", {})
        w, h = int(intr["image_width"]), int(intr["image_height"])
        fx = intr.get("fx")
        if fx is None:                      # derive from the FOV assumption
            fx = (w / 2.0) / math.tan(math.radians(intr["hfov_deg"]) / 2.0)
        return cls(
            fx=float(fx),
            fy=float(intr.get("fy", fx)),
            cx=float(intr.get("cx", w / 2.0)),
            cy=float(intr.get("cy", h / 2.0)),
            image_width=w,
            image_height=h,
            camera_height_m=float(extr.get("camera_height_m", 1.30)),
            pitch_deg=float(extr.get("pitch_deg", 0.0)),
            camera_height_sigma_m=float(extr.get("camera_height_sigma_m", 0.15)),
            pitch_sigma_deg=float(extr.get("pitch_sigma_deg", 1.5)),
            source=str(intr.get("source", "fov_assumption")),
        )

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, **kw) -> "CameraModel":
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   image_width=width, image_height=height, **kw)

    # ------------------------------------------------------------- properties
    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.image_width / 2.0) / self.fx))

    @property
    def horizon_row(self) -> float:
        """Image row of the horizon: ``v = cy - fy * tan(pitch)``."""
        return self.cy - self.fy * math.tan(self.pitch_rad)

    def R_level_from_cam(self) -> np.ndarray:
        """Rotation taking camera-frame vectors into the level frame."""
        t = self.pitch_rad
        c, s = math.cos(t), math.sin(t)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, c, s],
                         [0.0, -s, c]], dtype=np.float64)

    # --------------------------------------------------------------- geometry
    def rescaled(self, width: int, height: int) -> "CameraModel":
        """Return the same physical camera expressed for a resized image."""
        sx, sy = width / self.image_width, height / self.image_height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy,
                       cx=self.cx * sx, cy=self.cy * sy,
                       image_width=width, image_height=height)

    def backproject(self, u: float, v: float, depth_zc: float) -> np.ndarray:
        """Pixel + depth-along-optical-axis -> 3-D point in the level frame."""
        p_c = np.array([(u - self.cx) * depth_zc / self.fx,
                        (v - self.cy) * depth_zc / self.fy,
                        depth_zc], dtype=np.float64)
        return self.R_level_from_cam() @ p_c

    def depth_to_forward(self, depth_zc: float) -> float:
        """Optical-axis depth -> horizontal forward range (level frame Z)."""
        t = self.pitch_rad
        return depth_zc * math.cos(t)   # exact for a point on the optical axis

    def ground_plane_range(self, v: float, plane_height_m: float) -> float:
        """Horizontal range to a point lying on the plane ``Y_L = plane_height_m``.

        Derivation (docs/distance_estimation.md)::

            Z = h * (cos(t) - vn * sin(t)) / (vn * cos(t) + sin(t)),
            vn = (v - cy) / fy

        Returns ``inf`` when the ray is parallel to (or diverges from) the plane.
        """
        t = self.pitch_rad
        c, s = math.cos(t), math.sin(t)
        vn = (v - self.cy) / self.fy
        num = plane_height_m * (c - vn * s)
        den = vn * c + s
        if abs(den) < 1e-9 or num / den <= 0.0:
            return math.inf
        return num / den

    def forward_to_row(self, forward_z: float, plane_height_m: float) -> float:
        """Inverse of :meth:`ground_plane_range` - useful for drawing range rulers."""
        t = self.pitch_rad
        c, s = math.cos(t), math.sin(t)
        y_c = plane_height_m * c - forward_z * s
        z_c = plane_height_m * s + forward_z * c
        if abs(z_c) < 1e-9:
            return math.nan
        return self.cy + self.fy * y_c / z_c

    def as_dict(self) -> dict:
        return {
            "intrinsics": {
                "image_width": self.image_width, "image_height": self.image_height,
                "hfov_deg": round(self.hfov_deg, 4),
                "fx": round(self.fx, 4), "fy": round(self.fy, 4),
                "cx": round(self.cx, 4), "cy": round(self.cy, 4),
                "source": self.source,
            },
            "extrinsics": {
                "camera_height_m": round(self.camera_height_m, 4),
                "pitch_deg": round(self.pitch_deg, 4),
                "camera_height_sigma_m": self.camera_height_sigma_m,
                "pitch_sigma_deg": self.pitch_sigma_deg,
            },
        }
