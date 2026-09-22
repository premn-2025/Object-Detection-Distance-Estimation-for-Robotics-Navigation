"""Detector backends behind one interface.

The point of this module is that the *same* calling code drives the FP32
PyTorch model, the ONNX FP32/FP16 export and the INT8-quantised ONNX model, so
the edge-optimisation benchmark compares like with like instead of comparing
two different pre/post-processing stacks.

    det = build_detector("runs/nav/weights/best.pt", device="cuda")
    dets = det.infer(frame_bgr)          # -> Detections in original-image pixels
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detections:
    """Boxes in original-image pixel coordinates."""

    boxes: np.ndarray            # (N, 4) float32, xyxy
    scores: np.ndarray           # (N,)   float32
    class_ids: np.ndarray        # (N,)   int32
    class_names: list

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def names(self) -> list:
        return [self.class_names[i] for i in self.class_ids]

    @classmethod
    def empty(cls, class_names) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                   np.zeros((0,), np.int32), list(class_names))


# --------------------------------------------------------------------------- #
# pre / post processing shared by the ONNX backends
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    """Resize preserving aspect ratio and pad to a square. Returns (img, scale, (dw, dh))."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def decode_yolo(pred: np.ndarray, r: float, pad, orig_shape, class_names,
                conf: float = 0.25, iou: float = 0.45) -> Detections:
    """Decode an ultralytics YOLOv8/11 head: (1, 4+nc, A) with xywh in input pixels."""
    p = pred[0] if pred.ndim == 3 else pred
    if p.shape[0] < p.shape[1]:          # (4+nc, A) -> (A, 4+nc)
        p = p.transpose()
    boxes_xywh, cls_scores = p[:, :4], p[:, 4:]
    best = cls_scores.argmax(axis=1)
    conf_scores = cls_scores[np.arange(cls_scores.shape[0]), best]
    keep = conf_scores > conf
    if not keep.any():
        return Detections.empty(class_names)
    boxes_xywh, conf_scores, best = boxes_xywh[keep], conf_scores[keep], best[keep]

    cx, cy, bw, bh = boxes_xywh.T
    x1, y1 = cx - bw / 2, cy - bh / 2
    rects = np.stack([x1, y1, bw, bh], axis=1)

    idx = cv2.dnn.NMSBoxes(rects.tolist(), conf_scores.tolist(), conf, iou)
    if len(idx) == 0:
        return Detections.empty(class_names)
    idx = np.asarray(idx).reshape(-1)

    x1, y1 = x1[idx], y1[idx]
    x2, y2 = x1 + bw[idx], y1 + bh[idx]
    # undo letterbox
    left, top = pad
    x1 = (x1 - left) / r
    x2 = (x2 - left) / r
    y1 = (y1 - top) / r
    y2 = (y2 - top) / r
    h, w = orig_shape[:2]
    boxes = np.stack([x1.clip(0, w - 1), y1.clip(0, h - 1),
                      x2.clip(0, w - 1), y2.clip(0, h - 1)], axis=1).astype(np.float32)
    return Detections(boxes, conf_scores[idx].astype(np.float32),
                      best[idx].astype(np.int32), list(class_names))


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class BaseDetector:
    name = "base"
    class_names: list = []

    def infer(self, image: np.ndarray) -> Detections:
        raise NotImplementedError

    def warmup(self, n: int = 3, size: int = 640) -> None:
        dummy = np.zeros((size, size, 3), np.uint8)
        for _ in range(n):
            self.infer(dummy)

    @property
    def size_mb(self) -> float:
        p = Path(getattr(self, "weights", ""))
        return p.stat().st_size / 1e6 if p.exists() else float("nan")


class UltralyticsDetector(BaseDetector):
    """FP32/FP16 PyTorch (or any format ultralytics can load natively)."""

    def __init__(self, weights, device: str = "cpu", imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, half: bool = False) -> None:
        from ultralytics import YOLO

        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half and device != "cpu"
        self.class_names = [self.model.names[i] for i in sorted(self.model.names)]
        self.name = f"{Path(self.weights).stem}@{device}{'-fp16' if self.half else ''}"

    def infer(self, image: np.ndarray) -> Detections:
        # only pass `half` when it is actually wanted: recent ultralytics warns
        # about the argument on every call, even when it is False
        kw = {"half": True} if self.half else {}
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               device=self.device, verbose=False, **kw)[0]
        b = r.boxes
        if b is None or b.shape[0] == 0:
            return Detections.empty(self.class_names)
        return Detections(b.xyxy.cpu().numpy().astype(np.float32),
                          b.conf.cpu().numpy().astype(np.float32),
                          b.cls.cpu().numpy().astype(np.int32),
                          self.class_names)


class OnnxDetector(BaseDetector):
    """ONNX Runtime backend - the one that actually runs the quantised models."""

    def __init__(self, weights, class_names, providers=None, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, threads: int = 0) -> None:
        import onnxruntime as ort

        self.weights = str(weights)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(self.weights, so,
                                            providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else imgsz
        self.dtype = (np.float16 if "float16" in self.session.get_inputs()[0].type
                      else np.float32)
        self.class_names = list(class_names)
        self.conf, self.iou = conf, iou
        prov = self.session.get_providers()[0].replace("ExecutionProvider", "").lower()
        self.name = f"{Path(self.weights).stem}@onnx-{prov}"

    def infer(self, image: np.ndarray) -> Detections:
        img, r, pad = letterbox(image, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(self.dtype) / self.dtype(255.0)
        out = self.session.run(None, {self.input_name: blob})[0]
        return decode_yolo(np.asarray(out, np.float32), r, pad, image.shape,
                           self.class_names, self.conf, self.iou)


def build_detector(weights, *, device: str = "cpu", class_names=None, imgsz: int = 640,
                   conf: float = 0.25, iou: float = 0.45, half: bool = False,
                   threads: int = 0) -> BaseDetector:
    """Pick a backend from the file extension."""
    weights = str(weights)
    if weights.endswith(".onnx"):
        if class_names is None:
            raise ValueError("class_names is required for ONNX backends")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device != "cpu" else ["CPUExecutionProvider"])
        return OnnxDetector(weights, class_names, providers=providers, imgsz=imgsz,
                            conf=conf, iou=iou, threads=threads)
    return UltralyticsDetector(weights, device=device, imgsz=imgsz, conf=conf,
                               iou=iou, half=half)


def time_inference(det: BaseDetector, frames, runs: int = 50, warmup: int = 5) -> dict:
    """Median-based FPS measurement. Median, not mean - one GC pause should not
    decide the headline number."""
    if not len(frames):
        raise ValueError("no frames given")
    for i in range(warmup):
        det.infer(frames[i % len(frames)])
    times = []
    for i in range(runs):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        det.infer(f)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return {
        "backend": det.name,
        "runs": runs,
        "ms_median": float(np.median(times) * 1e3),
        "ms_mean": float(times.mean() * 1e3),
        "ms_p90": float(np.percentile(times, 90) * 1e3),
        "fps": float(1.0 / np.median(times)),
        "size_mb": det.size_mb,
    }
