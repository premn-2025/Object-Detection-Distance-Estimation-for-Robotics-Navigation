"""Drawing detections + ranges onto frames.

The required annotation is ``Cone, 1.5m``; everything else here (uncertainty,
cue provenance, bird's-eye panel) is opt-in so the default output stays legible.
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR. Cones warm, barriers amber, stop signs red - matched to the real objects
# so a frame is readable without consulting a legend.
CLASS_COLOURS = {
    "cone": (0, 140, 255),
    "barrier": (0, 215, 255),
    "stop_sign": (60, 60, 220),
}
DEFAULT_COLOUR = (0, 255, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _colour(name: str):
    return CLASS_COLOURS.get(name, DEFAULT_COLOUR)


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def draw_detections(frame: np.ndarray, boxes, class_names, estimates=None, scores=None, *,
                    show_sigma: bool = False, show_method: bool = False,
                    show_score: bool = False, thickness: int = 2) -> np.ndarray:
    """Annotate a copy of ``frame``.

    Args:
        boxes: iterable of (x1, y1, x2, y2).
        class_names: per-box class name.
        estimates: per-box :class:`DistanceEstimate` (or None to omit ranges).
        scores: per-box detector confidence.
    """
    out = frame.copy()
    n = len(class_names)
    estimates = estimates if estimates is not None else [None] * n
    scores = scores if scores is not None else [None] * n

    # far objects first, so a nearby label is never hidden behind a distant one
    order = sorted(range(n), key=lambda i: -(getattr(estimates[i], "distance_m", 0.0) or 0.0))

    scale = max(0.45, min(0.85, out.shape[1] / 1600.0))
    placed: list = []          # label boxes already drawn, for collision avoidance
    for i in order:
        x1, y1, x2, y2 = (int(round(v)) for v in boxes[i])
        name = class_names[i]
        col = _colour(name)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thickness)

        parts = [pretty(name)]
        est = estimates[i]
        if est is not None and getattr(est, "valid", False):
            txt = f"{est.distance_m:.1f}m"
            if show_sigma and np.isfinite(est.sigma_m):
                txt += f" +/-{est.sigma_m:.1f}"
            parts.append(txt)
            if show_method:
                parts.append(est.method)
        elif est is not None:
            parts.append("range n/a")
        if show_score and scores[i] is not None:
            parts.append(f"{scores[i]:.2f}")
        label = ", ".join(parts)

        (tw, th), base = cv2.getTextSize(label, FONT, scale, 1)
        lh = th + base + 4
        ty = y1 - 4 if y1 - lh > 0 else y2 + lh
        # Cone tapers put many boxes side by side; stack the labels instead of
        # letting them overprint each other into an unreadable smear.
        step = lh + 2
        for _ in range(6):
            rect = (x1, ty - lh, x1 + tw + 6, ty)
            if not any(_overlaps(rect, r) for r in placed):
                break
            ty = ty - step if ty - step - lh > 0 else ty + step
        ty = int(np.clip(ty, lh, out.shape[0] - 2))
        x_lab = int(np.clip(x1, 0, max(0, out.shape[1] - tw - 8)))
        rect = (x_lab, ty - lh, x_lab + tw + 6, ty)
        placed.append(rect)

        cv2.rectangle(out, (rect[0], rect[1]), (rect[2], rect[3]), col, -1)
        cv2.line(out, (x_lab + 2, ty), (x1 + 2, y1), col, 1, cv2.LINE_AA)
        cv2.putText(out, label, (x_lab + 3, ty - base), FONT, scale, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return out


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_horizon(frame: np.ndarray, camera, colour=(180, 180, 180)) -> np.ndarray:
    """Debug overlay: the horizon implied by the calibration."""
    out = frame.copy()
    cam = camera.rescaled(out.shape[1], out.shape[0])
    v = int(round(cam.horizon_row))
    if 0 <= v < out.shape[0]:
        cv2.line(out, (0, v), (out.shape[1], v), colour, 1, cv2.LINE_AA)
        cv2.putText(out, "horizon", (8, max(14, v - 6)), FONT, 0.5, colour, 1, cv2.LINE_AA)
    return out


def draw_range_rulers(frame: np.ndarray, camera, ranges=(5, 10, 20, 40),
                      colour=(120, 200, 120)) -> np.ndarray:
    """Draw the image rows where the ground plane is N metres ahead.

    A quick sanity check on the calibration: if a cone sitting on the 10 m line
    is not reported at ~10 m, the camera pose is wrong, not the detector.
    """
    out = frame.copy()
    cam = camera.rescaled(out.shape[1], out.shape[0])
    for z in ranges:
        v = cam.forward_to_row(float(z), cam.camera_height_m)
        if not np.isfinite(v):
            continue
        v = int(round(v))
        if 0 <= v < out.shape[0]:
            cv2.line(out, (0, v), (out.shape[1], v), colour, 1, cv2.LINE_AA)
            cv2.putText(out, f"{z} m", (out.shape[1] - 60, v - 4), FONT, 0.5,
                        colour, 1, cv2.LINE_AA)
    return out


def bev_panel(estimates, class_names, size=(420, 520), max_range_m: float = 40.0,
              half_width_m: float = 12.0) -> np.ndarray:
    """Top-down plot of where the detected objects are relative to the robot."""
    w, h = size
    panel = np.full((h, w, 3), 28, np.uint8)

    def to_px(lateral, forward):
        x = int(w / 2 + lateral / half_width_m * (w / 2))
        y = int(h - 20 - forward / max_range_m * (h - 40))
        return x, y

    for z in range(5, int(max_range_m) + 1, 5):
        _, y = to_px(0, z)
        cv2.line(panel, (0, y), (w, y), (55, 55, 55), 1)
        cv2.putText(panel, f"{z}m", (4, y - 3), FONT, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.line(panel, (w // 2, 0), (w // 2, h), (55, 55, 55), 1)

    rx, ry = to_px(0, 0)
    cv2.drawMarker(panel, (rx, ry), (240, 240, 240), cv2.MARKER_TRIANGLE_UP, 16, 2)
    cv2.putText(panel, "robot", (rx - 22, ry + 16), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    for est, name in zip(estimates, class_names):
        if est is None or not getattr(est, "valid", False):
            continue
        if not (np.isfinite(est.forward_m) and np.isfinite(est.lateral_m)):
            continue
        x, y = to_px(est.lateral_m, est.forward_m)
        if not (0 <= x < w and 0 <= y < h):
            continue
        col = _colour(name)
        # uncertainty along the range axis, drawn to scale
        if np.isfinite(est.sigma_m) and est.sigma_m > 0:
            _, y_lo = to_px(est.lateral_m, max(0.0, est.forward_m - est.sigma_m))
            _, y_hi = to_px(est.lateral_m, est.forward_m + est.sigma_m)
            cv2.line(panel, (x, y_lo), (x, y_hi), tuple(int(c * 0.55) for c in col), 2)
        cv2.circle(panel, (x, y), 5, col, -1)
        cv2.putText(panel, f"{est.distance_m:.1f}", (x + 8, y + 4), FONT, 0.42,
                    col, 1, cv2.LINE_AA)

    cv2.putText(panel, "bird's-eye view", (8, 18), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def hstack_panel(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    """Glue a side panel onto a frame, matching heights."""
    h = frame.shape[0]
    scale = h / panel.shape[0]
    panel = cv2.resize(panel, (int(panel.shape[1] * scale), h))
    return np.hstack([frame, panel])
