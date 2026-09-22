# Edge optimisation benchmark

```json
{
  "python": "3.11.0",
  "platform": "Windows-10-10.0.26200-SP0",
  "processor": "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel",
  "torch": "2.7.1+cu118",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "onnxruntime": "1.30.0",
  "ort_providers": [
    "AzureExecutionProvider",
    "CPUExecutionProvider"
  ],
  "cpu_count": 24
}
```

## CPU

| variant | device | size (MB) | latency p50 (ms) | FPS | speed-up | mAP50-95 | mAP50 | notes |
|---|---|---|---|---|---|---|---|---|
| yolo11s FP32 (PyTorch) | cpu | 19.16 | 38.4 | 26.0 | 1.00x | 0.4057 | 0.6007 | baseline |
| yolo11s FP32 (ONNX Runtime) | cpu | 37.93 | 40.0 | 25.0 | 0.96x | 0.4065 | 0.6035 | graph-optimised |
| yolo11s INT8 dynamic (ONNX) | cpu | 9.87 | 489.5 | 2.0 | 0.08x | 0.3969 | 0.5993 | weights INT8 |
| yolo11s INT8 static (ONNX) | cpu | 11.44 | 32.7 | 30.6 | 1.18x | 0.4003 | 0.5985 | weights+activations INT8, QDQ |
| yolo11s 30% unstructured-sparse | cpu | 19.19 | 45.2 | 22.1 | 0.85x | 0.4042 | 0.6011 | sparse weights, dense kernels - no speed-up expected |
| yolo11s 50% unstructured-sparse | cpu | 19.19 | 48.0 | 20.8 | 0.80x | 0.3744 | 0.5804 | sparse weights, dense kernels - no speed-up expected |
| yolo11n FP32 (PyTorch) | cpu | 5.45 | 26.6 | 37.6 | 1.44x | 0.3578 | 0.5531 | lightweight backbone |

## GPU

| variant | device | size (MB) | latency p50 (ms) | FPS | speed-up | mAP50-95 | mAP50 | notes |
|---|---|---|---|---|---|---|---|---|
| yolo11s FP32 (PyTorch) | 0 | 19.16 | 6.6 | 151.9 | 1.00x | 0.4064 | 0.6007 | baseline |
| yolo11s FP16 (PyTorch) | 0 | 19.16 | 7.1 | 141.8 | 0.93x | 0.4061 | 0.5999 | half precision |
| yolo11n FP32 (PyTorch) | 0 | 5.45 | 7.1 | 140.6 | 0.93x | 0.3578 | 0.5530 | lightweight backbone |
