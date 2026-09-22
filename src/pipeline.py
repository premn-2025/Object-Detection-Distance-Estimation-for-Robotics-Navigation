"""Detect -> range -> annotate, in one object.

This is the piece a robot would actually import::

    nav = NavigationPipeline.from_config("runs/nav/weights/best.pt")
    result = nav(frame_bgr)
    for obs in result.obstacles:
        print(obs.class_name, obs.distance_m, obs.lateral_m)

``result.annotated`` carries the ``Cone, 1.5m`` overlay; ``result.obstacles`` is
the structured output a planner would consume.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .distance.camera import CameraModel
from .distance.estimator import DistanceEstimate, DistanceEstimator
from .models.detector import BaseDetector, Detections, build_detector
from .viz import bev_panel, draw_detections, hstack_panel


@dataclass
class Obstacle:
    """One detected navigation obstacle with its range."""

    class_name: str
    score: float
    box: list                      # xyxy in original-image pixels
    distance_m: float = float("nan")
    sigma_m: float = float("nan")
    forward_m: float = float("nan")
    lateral_m: float = float("nan")
    range_method: str = "none"
    flags: list = field(default_factory=list)

    @property
    def annotation(self) -> str:
        """The label drawn on the frame, e.g. ``Cone, 1.5m``."""
        pretty = self.class_name.replace("_", " ").title()
        if not np.isfinite(self.distance_m):
            return pretty
        return f"{pretty}, {self.distance_m:.1f}m"


@dataclass
class FrameResult:
    obstacles: list
    annotated: np.ndarray = None
    detect_ms: float = 0.0
    range_ms: float = 0.0

    @property
    def fps(self) -> float:
        total = self.detect_ms + self.range_ms
        return 1000.0 / total if total > 0 else float("inf")

    def to_json(self) -> str:
        return json.dumps([asdict(o) for o in self.obstacles], indent=2, default=float)


class NavigationPipeline:
    """Detector + distance estimator + renderer."""

    def __init__(self, detector: BaseDetector, estimator: DistanceEstimator, *,
                 draw: bool = True, show_sigma: bool = False, show_method: bool = False,
                 show_score: bool = False, with_bev: bool = False) -> None:
        self.detector = detector
        self.estimator = estimator
        self.draw = draw
        self.show_sigma = show_sigma
        self.show_method = show_method
        self.show_score = show_score
        self.with_bev = with_bev

    @classmethod
    def from_config(cls, weights, *, camera_config="configs/camera.yaml",
                    device: str = "cpu", imgsz: int = 640, conf: float = 0.25,
                    iou: float = 0.45, class_names=None, half: bool = False,
                    **kw) -> "NavigationPipeline":
        det = build_detector(weights, device=device, imgsz=imgsz, conf=conf, iou=iou,
                             class_names=class_names, half=half)
        est = DistanceEstimator.from_yaml(camera_config)
        return cls(det, est, **kw)

    @property
    def camera(self) -> CameraModel:
        return self.estimator.camera

    def __call__(self, frame: np.ndarray) -> FrameResult:
        return self.process(frame)

    def process(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        dets: Detections = self.detector.infer(frame)
        t1 = time.perf_counter()

        names = dets.names()
        estimates = [self.estimator.estimate(box, name, frame.shape[:2])
                     for box, name in zip(dets.boxes, names)]
        t2 = time.perf_counter()

        obstacles = []
        for box, name, score, est in zip(dets.boxes, names, dets.scores, estimates):
            obstacles.append(Obstacle(
                class_name=name, score=float(score), box=[float(v) for v in box],
                distance_m=est.distance_m, sigma_m=est.sigma_m,
                forward_m=est.forward_m, lateral_m=est.lateral_m,
                range_method=est.method, flags=list(est.flags)))
        # nearest first: that is the order a planner cares about
        order = np.argsort([o.distance_m if np.isfinite(o.distance_m) else 1e9
                            for o in obstacles])
        obstacles = [obstacles[i] for i in order]
        estimates = [estimates[i] for i in order]

        annotated = None
        if self.draw:
            annotated = draw_detections(
                frame, [o.box for o in obstacles], [o.class_name for o in obstacles],
                estimates, [o.score for o in obstacles],
                show_sigma=self.show_sigma, show_method=self.show_method,
                show_score=self.show_score)
            if self.with_bev:
                panel = bev_panel(estimates, [o.class_name for o in obstacles])
                annotated = hstack_panel(annotated, panel)

        return FrameResult(obstacles=obstacles, annotated=annotated,
                           detect_ms=(t1 - t0) * 1e3, range_ms=(t2 - t1) * 1e3)

    def process_video(self, src, dst=None, *, max_frames: int = 0, stride: int = 1,
                      on_frame=None) -> list:
        """Run over a video file (or image folder) and optionally write an overlay."""
        import cv2

        src = str(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video source: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        results = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    res = self.process(frame)
                    results.append(res)
                    if on_frame is not None:
                        on_frame(idx, res)
                    if dst is not None and res.annotated is not None:
                        if writer is None:
                            h, w = res.annotated.shape[:2]
                            Path(dst).parent.mkdir(parents=True, exist_ok=True)
                            writer = cv2.VideoWriter(str(dst),
                                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                                     fps / stride, (w, h))
                        writer.write(res.annotated)
                    if max_frames and len(results) >= max_frames:
                        break
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        return results
