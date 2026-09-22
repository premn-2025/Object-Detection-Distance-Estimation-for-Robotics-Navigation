"""Tracking cones across frames, and what tracking buys a navigation stack.

Two uses, in increasing order of usefulness to a robot:

1. **Identity.**  A detector fires independently every frame.  Associating boxes
   over time turns "seven cones" into "cone #3, still there" and lets a planner
   reason about a lane of cones instead of a cloud of boxes.

2. **Range rate and time-to-collision.**  A single frame gives a noisy distance.
   The *sequence* gives closing speed, and ``TTC = Z / (-dZ/dt)`` is what
   actually drives a stop decision.  Distances are fitted over a short window
   with least squares rather than differenced between adjacent frames, because
   differencing a noisy signal amplifies exactly the noise that matters.

Association uses IoU plus a Lucas-Kanade flow prediction: pure IoU breaks when
the robot moves quickly and boxes stop overlapping between frames, while flow
predicts where the box *should* have gone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def sparse_flow(prev_gray: np.ndarray, gray: np.ndarray, points: np.ndarray):
    """Lucas-Kanade forward-backward check. Returns (new_points, valid_mask)."""
    if len(points) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), bool)
    p0 = np.asarray(points, np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)
    fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 3.0)
    return p1.reshape(-1, 2), ok


def dense_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Farneback dense flow - used for the visualisation, not the tracker."""
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    box: list
    age: int = 0
    hits: int = 1
    misses: int = 0
    distances: deque = field(default_factory=lambda: deque(maxlen=12))
    times: deque = field(default_factory=lambda: deque(maxlen=12))
    centres: list = field(default_factory=list)

    @property
    def centre(self):
        return (0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3]))

    def range_rate(self):
        """Least-squares dZ/dt over the window, m/s. Negative = approaching."""
        if len(self.distances) < 4:
            return None
        t = np.asarray(self.times, float)
        z = np.asarray(self.distances, float)
        ok = np.isfinite(z)
        if ok.sum() < 4 or np.ptp(t[ok]) < 1e-6:
            return None
        slope, _ = np.polyfit(t[ok], z[ok], 1)
        return float(slope)

    def time_to_collision(self):
        """``TTC = Z / -(dZ/dt)`` in seconds; None when not closing."""
        rate = self.range_rate()
        if rate is None or rate >= -1e-3:
            return None
        z = next((d for d in reversed(self.distances) if np.isfinite(d)), None)
        if z is None:
            return None
        return float(z / -rate)


class FlowTracker:
    """IoU + optical-flow association across frames."""

    def __init__(self, iou_threshold: float = 0.2, max_misses: int = 8,
                 use_flow: bool = True) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.use_flow = use_flow
        self.tracks: list = []
        self._next_id = 1
        self._prev_gray = None

    def _predict(self, gray: np.ndarray) -> None:
        """Shift every track's box by the measured flow at its centre."""
        if not self.use_flow or self._prev_gray is None or not self.tracks:
            return
        pts = np.array([t.centre for t in self.tracks], np.float32)
        moved, ok = sparse_flow(self._prev_gray, gray, pts)
        for trk, p, good in zip(self.tracks, moved, ok):
            if not good:
                continue
            dx, dy = p[0] - trk.centre[0], p[1] - trk.centre[1]
            trk.box = [trk.box[0] + dx, trk.box[1] + dy, trk.box[2] + dx, trk.box[3] + dy]

    def update(self, frame: np.ndarray, obstacles, timestamp: float) -> list:
        """Associate this frame's obstacles with existing tracks."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._predict(gray)

        unmatched = list(range(len(obstacles)))
        for trk in self.tracks:
            trk.age += 1
            best, best_iou = None, self.iou_threshold
            for i in unmatched:
                obs = obstacles[i]
                if obs.class_name != trk.class_name:
                    continue
                v = iou(trk.box, obs.box)
                if v > best_iou:
                    best, best_iou = i, v
            if best is None:
                trk.misses += 1
                continue
            obs = obstacles[best]
            trk.box = list(obs.box)
            trk.hits += 1
            trk.misses = 0
            trk.distances.append(obs.distance_m)
            trk.times.append(timestamp)
            trk.centres.append(trk.centre)
            unmatched.remove(best)

        for i in unmatched:
            obs = obstacles[i]
            trk = Track(self._next_id, obs.class_name, list(obs.box))
            trk.distances.append(obs.distance_m)
            trk.times.append(timestamp)
            trk.centres.append(trk.centre)
            self.tracks.append(trk)
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self._prev_gray = gray
        return self.tracks

    def confirmed(self, min_hits: int = 3) -> list:
        return [t for t in self.tracks if t.hits >= min_hits and t.misses == 0]


def draw_tracks(frame: np.ndarray, tracks, *, min_hits: int = 3,
                trail: int = 20) -> np.ndarray:
    """Overlay track ids, trails and TTC where it is defined."""
    from ..viz import CLASS_COLOURS, DEFAULT_COLOUR

    out = frame.copy()
    for trk in tracks:
        if trk.hits < min_hits:
            continue
        col = CLASS_COLOURS.get(trk.class_name, DEFAULT_COLOUR)
        x1, y1, x2, y2 = (int(round(v)) for v in trk.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        pts = trk.centres[-trail:]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)
        label = f"#{trk.track_id}"
        z = next((d for d in reversed(trk.distances) if np.isfinite(d)), None)
        if z is not None:
            label += f" {z:.1f}m"
        ttc = trk.time_to_collision()
        if ttc is not None and ttc < 20:
            label += f" TTC {ttc:.1f}s"
        cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 2, cv2.LINE_AA)
    return out
