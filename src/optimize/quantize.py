"""INT8 quantisation of the exported ONNX detector.

Two schemes, because they fail differently on a detection head:

*dynamic*  weights INT8, activations quantised on the fly.  No calibration data,
           no accuracy cliff, but the activation quantisation cost eats much of
           the speed-up.

*static*   weights and activations INT8, activation ranges measured on real
           frames.  Faster, but a detection head is sensitive: the box-regression
           branch outputs coordinates in pixel units spanning three orders of
           magnitude, and forcing that through one INT8 scale destroys
           localisation.  ``exclude_output_ops`` keeps the last few nodes in
           float, which is what makes the static path usable at all.

Calibration frames come from the *training* split - using val frames to pick
quantisation ranges would leak the evaluation set into the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ImageCalibrationReader:
    """Feeds real, letterboxed frames to ONNX Runtime's static quantiser."""

    def __init__(self, image_paths, input_name: str, imgsz: int = 640, limit: int = 128):
        from onnxruntime.quantization import CalibrationDataReader  # noqa: F401

        self.image_paths = list(image_paths)[:limit]
        self.input_name = input_name
        self.imgsz = imgsz
        self._iter = None

    def _load(self):
        import cv2

        from ..models.detector import letterbox

        for p in self.image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            lb, _, _ = letterbox(img, self.imgsz)
            blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            yield {self.input_name: blob}

    def get_next(self):
        if self._iter is None:
            self._iter = self._load()
        return next(self._iter, None)

    def rewind(self):
        self._iter = None


def quantize_dynamic_int8(onnx_fp32, out_path=None) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_fp32 = Path(onnx_fp32)
    out_path = Path(out_path or onnx_fp32.with_name(onnx_fp32.stem + "_int8_dynamic.onnx"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_fp32), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def _tail_node_names(onnx_fp32: Path, n: int = 40) -> list:
    """Names of the last ``n`` nodes - the detection head's output arithmetic."""
    import onnx

    model = onnx.load(str(onnx_fp32))
    nodes = [nd.name for nd in model.graph.node if nd.name]
    return nodes[-n:]


def quantize_static_int8(onnx_fp32, calibration_images, *, imgsz: int = 640,
                         limit: int = 128, per_channel: bool = True,
                         exclude_output_ops: int = 40, out_path=None) -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.preprocess import quant_pre_process

    onnx_fp32 = Path(onnx_fp32)
    out_path = Path(out_path or onnx_fp32.with_name(onnx_fp32.stem + "_int8_static.onnx"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prepped = onnx_fp32.with_name(onnx_fp32.stem + "_prep.onnx")
    quant_pre_process(str(onnx_fp32), str(prepped), skip_symbolic_shape=True)

    sess = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    reader = ImageCalibrationReader(calibration_images, input_name, imgsz, limit)
    nodes_to_exclude = _tail_node_names(prepped, exclude_output_ops) if exclude_output_ops else []

    quantize_static(
        str(prepped), str(out_path), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    prepped.unlink(missing_ok=True)
    return out_path


def model_size_mb(path) -> float:
    return Path(path).stat().st_size / 1e6
