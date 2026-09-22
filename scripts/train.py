"""Training entry point.

    python scripts/train.py stage1                 # BDD100K domain adaptation
    python scripts/train.py stage1_nano            # same, for the edge backbone
    python scripts/train.py stage2                 # cone / barrier / stop_sign
    python scripts/train.py stage2 --from-coco     # ablation: skip stage 1
    python scripts/train.py stage2_nano            # lightweight edge backbone
    python scripts/train.py eval --weights ... --data ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.trainer import evaluate, load_config, summarise, train  # noqa: E402

STAGE1_BEST = Path("runs/stage1/bdd_yolo11s/weights/best.pt")


def _stage1_weights(cfg: dict, explicit: str = None) -> str:
    if explicit:
        return explicit
    if STAGE1_BEST.exists():
        return str(STAGE1_BEST)
    raise FileNotFoundError(
        f"{STAGE1_BEST} not found - run `python scripts/train.py stage1` first, "
        f"or pass --from-coco to train stage 2 directly from COCO weights")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage",
                   choices=["stage1", "stage1_nano", "stage2", "stage2_nano", "eval"])
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--weights", default=None, help="override the starting checkpoint")
    p.add_argument("--from-coco", action="store_true",
                   help="stage 2 ablation: start from COCO instead of stage-1 weights")
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None, help="where to write the metrics json")
    p.add_argument("--resume", default=None,
                   help="resume an interrupted run from its last.pt checkpoint")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    common = cfg["common"]

    if args.resume:
        # ultralytics restores the original arguments from the checkpoint, so the
        # resumed run is a continuation of the same schedule, not a new one
        from ultralytics import YOLO

        from src.train.trainer import resolve_device

        preset = common.get("hard_augment", "off")
        if preset and preset != "off":
            from src.train.augment import install_hard_augmentation

            install_hard_augmentation(preset)
        print(f"[train] resuming from {args.resume}")
        results = YOLO(args.resume).train(resume=True, device=resolve_device(args.device))
        summary = summarise(results, args.out or f"outputs/metrics/{args.stage}_resumed.json")
        print(json.dumps(summary, indent=2))
        return 0

    if args.stage == "eval":
        if not args.weights or not args.data:
            p.error("eval needs --weights and --data")
        res = evaluate(args.weights, args.data, device=args.device,
                       imgsz=args.imgsz or common["imgsz"],
                       out_path=args.out or "outputs/metrics/eval.json")
        print(json.dumps(res, indent=2))
        return 0

    stage_cfg = dict(cfg[args.stage])
    overrides = {}
    for key in ("epochs", "batch", "imgsz", "data"):
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val

    if args.stage in ("stage1", "stage1_nano"):
        start = args.weights or stage_cfg.get("model")
        name = args.name or stage_cfg["name"]
    else:
        if args.from_coco:
            base = "yolo11n.pt" if args.stage == "stage2_nano" else "yolo11s.pt"
            start = args.weights or base
            name = args.name or f"{stage_cfg['name']}_from_coco"
        else:
            start = _stage1_weights(cfg, args.weights)
            name = args.name or stage_cfg["name"]

    _, results = train(stage_cfg, common, model=start, device=args.device,
                       name=name, overrides=overrides)
    summary = summarise(results, args.out or f"outputs/metrics/{args.stage}_{name}.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
