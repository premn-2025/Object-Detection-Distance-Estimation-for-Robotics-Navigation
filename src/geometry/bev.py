"""Inverse perspective mapping - the road as a metric bird's-eye raster.

The homography here is not fitted from four clicked points; it is written down
directly from the camera model, so the output is in metres and consistent with
whatever :mod:`src.distance` reports.

A ground point ``(X, Z)`` in metres projects to::

    p_C = R_x(theta) @ (X, h, Z)^T
    (u, v, 1)^T ~ K @ p_C

Both sides are linear in ``(X, Z, 1)``, so the whole road plane maps through a
single 3x3 homography::

    H_img_from_ground = K @ [ r1 | r3 | h * r2 ]

where ``r1, r2, r3`` are the columns of ``R_x(theta)``: dropping the ``Y``
coordinate (constant at ``h`` on the road) collapses the 3x4 projection into a
3x3 matrix.  Composing that with a metres-to-pixels scaling of the BEV canvas
gives the warp.

Caveat worth stating plainly: IPM assumes the world *is* a plane.  Anything with
height - a cone, a barrier, another vehicle - gets smeared along its viewing ray.
The smear is a correct rendering of a wrong assumption, which is why detected
objects are drawn as markers at their estimated ground position instead of being
read off the warped image.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..distance.camera import CameraModel


class BEVTransform:
    """Maps between image pixels, ground metres, and a bird's-eye raster."""

    def __init__(self, camera: CameraModel, *, x_range=(-10.0, 10.0),
                 z_range=(0.5, 40.0), px_per_m: float = 12.0) -> None:
        self.camera = camera
        self.x_range = x_range
        self.z_range = z_range
        self.px_per_m = float(px_per_m)
        self.width = int(round((x_range[1] - x_range[0]) * px_per_m))
        self.height = int(round((z_range[1] - z_range[0]) * px_per_m))

    # ------------------------------------------------------------- matrices
    def H_image_from_ground(self) -> np.ndarray:
        """3x3 taking ground ``(X, Z, 1)`` in metres to image ``(u, v, 1)``."""
        cam = self.camera
        t = cam.pitch_rad
        c, s = math.cos(t), math.sin(t)
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, c, -s],
                      [0.0, s, c]], dtype=np.float64)   # R_x(theta): level -> camera
        r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
        M = np.stack([r1, r3, cam.camera_height_m * r2], axis=1)
        return cam.K @ M

    def H_bev_from_ground(self) -> np.ndarray:
        """Ground metres -> BEV pixels (x right, z up the canvas)."""
        x0, _ = self.x_range
        _, z1 = self.z_range
        s = self.px_per_m
        return np.array([[s, 0.0, -x0 * s],
                         [0.0, -s, z1 * s],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def H_bev_from_image(self) -> np.ndarray:
        return self.H_bev_from_ground() @ np.linalg.inv(self.H_image_from_ground())

    # ---------------------------------------------------------------- warps
    def warp(self, image: np.ndarray, *, border_value=(20, 20, 20)) -> np.ndarray:
        """Warp a frame into the bird's-eye raster."""
        cam = self.camera.rescaled(image.shape[1], image.shape[0])
        H = BEVTransform(cam, x_range=self.x_range, z_range=self.z_range,
                         px_per_m=self.px_per_m).H_bev_from_image()
        return cv2.warpPerspective(image, H, (self.width, self.height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=border_value)

    def ground_to_bev(self, x_m: float, z_m: float) -> tuple:
        p = self.H_bev_from_ground() @ np.array([x_m, z_m, 1.0])
        return int(round(p[0] / p[2])), int(round(p[1] / p[2]))

    # --------------------------------------------------------------- drawing
    def draw_grid(self, bev: np.ndarray, step_m: float = 5.0,
                  colour=(70, 70, 70)) -> np.ndarray:
        out = bev.copy()
        z = math.ceil(self.z_range[0] / step_m) * step_m
        while z <= self.z_range[1]:
            _, y = self.ground_to_bev(0.0, z)
            cv2.line(out, (0, y), (out.shape[1], y), colour, 1)
            cv2.putText(out, f"{z:.0f}m", (4, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
            z += step_m
        x = math.ceil(self.x_range[0] / step_m) * step_m
        while x <= self.x_range[1]:
            px, _ = self.ground_to_bev(x, self.z_range[0])
            cv2.line(out, (px, 0), (px, out.shape[0]), colour, 1)
            x += step_m
        return out

    def draw_objects(self, bev: np.ndarray, obstacles, colours=None) -> np.ndarray:
        """Place detected objects at their estimated ground position."""
        from ..viz import CLASS_COLOURS, DEFAULT_COLOUR

        colours = colours or CLASS_COLOURS
        out = bev.copy()
        for obs in obstacles:
            fwd = getattr(obs, "forward_m", None)
            lat = getattr(obs, "lateral_m", None)
            if fwd is None or lat is None or not (np.isfinite(fwd) and np.isfinite(lat)):
                continue
            if not (self.z_range[0] <= fwd <= self.z_range[1]):
                continue
            if not (self.x_range[0] <= lat <= self.x_range[1]):
                continue
            x, y = self.ground_to_bev(lat, fwd)
            col = colours.get(obs.class_name, DEFAULT_COLOUR)
            sigma = getattr(obs, "sigma_m", float("nan"))
            if np.isfinite(sigma) and sigma > 0:
                _, y_lo = self.ground_to_bev(lat, max(self.z_range[0], fwd - sigma))
                _, y_hi = self.ground_to_bev(lat, min(self.z_range[1], fwd + sigma))
                cv2.line(out, (x, y_lo), (x, y_hi),
                         tuple(int(c * 0.6) for c in col), 2)
            cv2.circle(out, (x, y), 5, col, -1)
            cv2.putText(out, f"{obs.distance_m:.1f}m", (x + 7, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return out

    def draw_robot(self, bev: np.ndarray) -> np.ndarray:
        out = bev.copy()
        x, y = self.ground_to_bev(0.0, self.z_range[0])
        cv2.drawMarker(out, (x, min(y, out.shape[0] - 6)), (240, 240, 240),
                       cv2.MARKER_TRIANGLE_UP, 16, 2)
        return out


def homography_from_points(src_pts, dst_pts) -> np.ndarray:
    """Fallback for an uncalibrated camera: four ground correspondences.

    Use this when the intrinsics are unknown but four points with known ground
    coordinates are visible (lane-marking corners, a taped rectangle).
    """
    src = np.asarray(src_pts, np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, np.float32).reshape(-1, 1, 2)
    if len(src) < 4:
        raise ValueError("need at least 4 correspondences")
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("homography estimation failed")
    return H
