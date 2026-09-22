"""Export a trained checkpoint to the formats the edge benchmark compares."""

from __future__ import annotations

from pathlib import Path


def export_onnx(weights, *, imgsz: int = 640, half: bool = False, opset: int = 13,
                simplify: bool = True, dynamic: bool = False, out_path=None) -> Path:
    """Export to ONNX via ultralytics.

    ``opset`` is pinned at 13 because that is the lowest version ONNX Runtime's
    static quantiser handles cleanly for this graph; newer opsets export fine but
    introduce ops the QDQ pass then refuses to fold.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    produced = model.export(format="onnx", imgsz=imgsz, half=half, opset=opset,
                            simplify=simplify, dynamic=dynamic, verbose=False)
    produced = Path(produced)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if produced.resolve() != out_path.resolve():
            out_path.write_bytes(produced.read_bytes())
        return out_path
    return produced


def class_names(weights) -> list:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    return [m.names[i] for i in sorted(m.names)]
