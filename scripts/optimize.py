"""Edge-optimisation pipeline: export, quantise, prune, and benchmark.

    python scripts/optimize.py --weights runs/stage2/nav_yolo11s/weights/best.pt --all

Steps (each can be run alone):
  --export    ONNX FP32 + FP16
  --quantize  INT8 dynamic and INT8 static (calibrated on training frames)
  --prune     structured channel pruning + short fine-tune
  --bench     time and score every artefact that exists, on CPU and GPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimize import benchmark as bench  # noqa: E402
from src.optimize import prune as prune_mod  # noqa: E402
from src.optimize import quantize as quant  # noqa: E402
from src.optimize.export import class_names, export_onnx  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--nano-weights", default=None,
                   help="the lightweight-backbone model, benchmarked alongside")
    p.add_argument("--data", default="data/processed/nav/data.yaml")
    p.add_argument("--out-dir", default="outputs/optimized")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=60)
    p.add_argument("--calib-images", default="data/processed/nav/images/train",
                   help="static-quantisation calibration frames (TRAIN split: using "
                        "val frames to pick INT8 ranges leaks the evaluation set)")
    p.add_argument("--bench-images", default="data/processed/nav/images/val",
                   help="frames the latency benchmark runs on")
    p.add_argument("--calib-limit", type=int, default=128)
    p.add_argument("--prune-ratio", type=float, default=0.30)
    p.add_argument("--prune-epochs", type=int, default=12)
    p.add_argument("--prune-device", default="cpu",
                   help="torch-pruning traces the graph with a forward pass; on CPU that is very slow for this model")
    p.add_argument("--no-accuracy", action="store_true")

    p.add_argument("--all", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--unstructured-prune", action="store_true")
    p.add_argument("--unstructured-ratios", type=float, nargs="+",
                   default=[0.3, 0.5])
    p.add_argument("--bench", action="store_true")
    args = p.parse_args(argv)

    if args.all:
        args.export = args.quantize = args.prune = args.bench = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names(args.weights)
    report = {"weights": args.weights, "classes": names, "steps": {}}
    report_path = out_dir / "optimization_report.json"

    fp32_onnx = out_dir / "model_fp32.onnx"
    fp16_onnx = out_dir / "model_fp16.onnx"
    int8_dyn = out_dir / "model_int8_dynamic.onnx"
    int8_sta = out_dir / "model_int8_static.onnx"
    pruned_pt = out_dir / "model_pruned.pt"

    if args.export:
        print("[opt] exporting ONNX ...")
        export_onnx(args.weights, imgsz=args.imgsz, half=False, out_path=fp32_onnx)
        report["steps"]["export_fp32"] = {"path": str(fp32_onnx),
                                          "size_mb": quant.model_size_mb(fp32_onnx)}
        try:
            export_onnx(args.weights, imgsz=args.imgsz, half=True, out_path=fp16_onnx)
            report["steps"]["export_fp16"] = {"path": str(fp16_onnx),
                                              "size_mb": quant.model_size_mb(fp16_onnx)}
        except Exception as exc:                  # noqa: BLE001
            report["steps"]["export_fp16"] = {"error": str(exc)[:200]}
        report_path.write_text(json.dumps(report, indent=2, default=float))

    if args.quantize:
        if not fp32_onnx.exists():
            print("[opt] no FP32 ONNX yet - exporting first")
            export_onnx(args.weights, imgsz=args.imgsz, out_path=fp32_onnx)
        print("[opt] INT8 dynamic ...")
        try:
            quant.quantize_dynamic_int8(fp32_onnx, int8_dyn)
            report["steps"]["int8_dynamic"] = {"path": str(int8_dyn),
                                               "size_mb": quant.model_size_mb(int8_dyn)}
        except Exception as exc:                  # noqa: BLE001
            report["steps"]["int8_dynamic"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print("[opt] INT8 static (calibrating on training frames) ...")
        try:
            calib = sorted(Path(args.calib_images).glob("*.jpg"))[:args.calib_limit]
            quant.quantize_static_int8(fp32_onnx, calib, imgsz=args.imgsz,
                                       limit=args.calib_limit, out_path=int8_sta)
            report["steps"]["int8_static"] = {"path": str(int8_sta),
                                              "size_mb": quant.model_size_mb(int8_sta),
                                              "calibration_images": len(calib)}
        except Exception as exc:                  # noqa: BLE001
            report["steps"]["int8_static"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        report_path.write_text(json.dumps(report, indent=2, default=float))

    if args.unstructured_prune:
        print(f"[opt] unstructured L1 pruning at {args.prune_ratio} ...")
        rows = []
        for amount in args.unstructured_ratios:
            out_pt = out_dir / f"model_unstructured_{int(amount*100)}.pt"
            info = prune_mod.global_unstructured_prune(args.weights, amount=amount,
                                                       out_path=out_pt)
            try:
                from src.train.trainer import evaluate

                m = evaluate(out_pt, args.data, device="0", imgsz=args.imgsz)
                info["mAP50-95"] = m["mAP50-95"]
                info["mAP50"] = m["mAP50"]
            except Exception as exc:              # noqa: BLE001
                info["eval_error"] = str(exc)[:200]
            info["size_mb"] = quant.model_size_mb(out_pt)
            rows.append(info)
            print(f"  amount={amount}: sparsity={info['conv_weight_sparsity']:.3f} "
                  f"mAP50-95={info.get('mAP50-95')}")
        report["steps"]["unstructured_prune"] = rows
        report_path.write_text(json.dumps(report, indent=2, default=float),
                               encoding="utf-8")

    if args.prune:
        print(f"[opt] structured pruning at ratio {args.prune_ratio} ...")
        try:
            info = prune_mod.structured_prune(args.weights, amount=args.prune_ratio,
                                              imgsz=args.imgsz, out_path=pruned_pt,
                                              device=args.prune_device)
            print(json.dumps(info, indent=2, default=float))
            print("[opt] fine-tuning the pruned model ...")
            ft = prune_mod.finetune(pruned_pt, args.data, epochs=args.prune_epochs,
                                    imgsz=args.imgsz, name="pruned_finetune")
            info["finetune"] = ft
            best = Path(ft["save_dir"]) / "weights" / "best.pt"
            if best.exists():
                info["finetuned_weights"] = str(best)
            report["steps"]["structured_prune"] = info
        except Exception as exc:                  # noqa: BLE001
            import traceback

            report["steps"]["structured_prune"] = {
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "traceback": traceback.format_exc()[-1200:]}
            print(report["steps"]["structured_prune"]["error"])
        report_path.write_text(json.dumps(report, indent=2, default=float))

    if args.bench:
        print("[opt] benchmarking ...")
        import torch

        gpu = "0" if torch.cuda.is_available() else None
        frames = bench.load_frames(args.bench_images, n=32)

        variants = [
            bench.Variant("yolo11s FP32 (PyTorch)", args.weights, device="cpu",
                          imgsz=args.imgsz, notes="baseline"),
        ]
        if gpu:
            variants += [
                bench.Variant("yolo11s FP32 (PyTorch)", args.weights, device=gpu,
                              imgsz=args.imgsz, notes="baseline"),
                bench.Variant("yolo11s FP16 (PyTorch)", args.weights, device=gpu,
                              half=True, imgsz=args.imgsz, notes="half precision"),
            ]
        if fp32_onnx.exists():
            variants.append(bench.Variant("yolo11s FP32 (ONNX Runtime)", str(fp32_onnx),
                                          device="cpu", class_names=names,
                                          imgsz=args.imgsz, notes="graph-optimised"))
        if int8_dyn.exists():
            variants.append(bench.Variant("yolo11s INT8 dynamic (ONNX)", str(int8_dyn),
                                          device="cpu", class_names=names,
                                          imgsz=args.imgsz, notes="weights INT8"))
        if int8_sta.exists():
            variants.append(bench.Variant("yolo11s INT8 static (ONNX)", str(int8_sta),
                                          device="cpu", class_names=names,
                                          imgsz=args.imgsz,
                                          notes="weights+activations INT8, QDQ"))
        pruned_ft = report.get("steps", {}).get("structured_prune", {}).get(
            "finetuned_weights")
        if pruned_ft and Path(pruned_ft).exists():
            variants.append(bench.Variant(
                f"yolo11s pruned {int(args.prune_ratio*100)}% (PyTorch)", pruned_ft,
                device="cpu", imgsz=args.imgsz, notes="structured, fine-tuned"))
            if gpu:
                variants.append(bench.Variant(
                    f"yolo11s pruned {int(args.prune_ratio*100)}% (PyTorch)", pruned_ft,
                    device=gpu, imgsz=args.imgsz, notes="structured, fine-tuned"))
        # Unstructured-sparse models are included precisely to show they are NOT
        # faster: same tensor shapes, same dense kernels, same latency.
        for amount in (0.3, 0.5):
            sp = out_dir / f"model_unstructured_{int(amount*100)}.pt"
            if sp.exists():
                variants.append(bench.Variant(
                    f"yolo11s {int(amount*100)}% unstructured-sparse", str(sp),
                    device="cpu", imgsz=args.imgsz,
                    notes="sparse weights, dense kernels - no speed-up expected"))
        if args.nano_weights and Path(args.nano_weights).exists():
            variants.append(bench.Variant("yolo11n FP32 (PyTorch)", args.nano_weights,
                                          device="cpu", imgsz=args.imgsz,
                                          notes="lightweight backbone"))
            if gpu:
                variants.append(bench.Variant("yolo11n FP32 (PyTorch)", args.nano_weights,
                                              device=gpu, imgsz=args.imgsz,
                                              notes="lightweight backbone"))

        rows = bench.run_suite(variants, frames, args.data, runs=args.runs,
                               measure_accuracy=not args.no_accuracy)
        host = bench.host_info()
        js, md = bench.save(rows, host, out_dir="outputs/benchmark")
        cpu_rows = [r for r in rows if r["device"] == "cpu"]
        gpu_rows = [r for r in rows if r["device"] != "cpu"]
        text = ["# Edge optimisation benchmark", "",
                "```json", json.dumps(host, indent=2), "```", ""]
        if cpu_rows:
            text += ["## CPU", "", bench.to_markdown(
                cpu_rows, "yolo11s FP32 (PyTorch)"), ""]
        if gpu_rows:
            text += ["## GPU", "", bench.to_markdown(
                gpu_rows, "yolo11s FP32 (PyTorch)"), ""]
        md.write_text("\n".join(text))
        print(f"[opt] benchmark -> {js} and {md}")
        report["steps"]["benchmark"] = {"json": str(js), "markdown": str(md)}

    report_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"[opt] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
