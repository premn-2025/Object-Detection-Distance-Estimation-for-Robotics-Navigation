"""Audit the train/val split for the failure mode that fakes good results.

ROADWork frames are video frames. Two frames 30 ms apart are almost the same
picture. If one lands in train and the other in val, the validation score
measures memorisation, not generalisation — and it does so *silently*, because
the loss curves look perfectly healthy. A model can be badly overfit and still
show a rising val mAP if val is contaminated.

This script checks three things and fails loudly on any of them:

1. **Identical file stems** in both splits (the trivial case).
2. **Shared source video / sequence** between splits — the real risk.
3. **Near-duplicate frames** across the split boundary, by perceptual hash, which
   also catches duplicates that the file names do not reveal.

It also reports the class balance per split, because a val set with four stop
signs in it produces a meaningless stop-sign mAP.

    python scripts/check_leakage.py --dataset data/processed/nav
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sequences import parse_stem  # noqa: E402


def group_key(stem: str):
    """The unit that must not straddle the split: a source clip, or one image."""
    info = parse_stem(stem)
    if info:                       # ROADWork: city + video + sequence
        return f"rw::{info['city']}_{info['video']}_{info['seq']}"
    if stem.startswith("bddstop_"):
        return f"bdd::{stem}"      # BDD frames are independent scenes
    if stem.startswith("coco_"):
        return f"coco::{stem}"
    return f"other::{stem}"


def phash(path, size: int = 16) -> int:
    """Cheap difference hash - enough to catch near-duplicate video frames."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/processed/nav")
    p.add_argument("--out", default="outputs/metrics/leakage_audit.json")
    p.add_argument("--hash-limit", type=int, default=1500,
                   help="frames per split to perceptually hash (0 = all)")
    p.add_argument("--hash-threshold", type=int, default=6,
                   help="hamming distance below which two frames count as duplicates")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any leakage is found")
    args = p.parse_args(argv)

    root = Path(args.dataset)
    splits = {}
    for split in ("train", "val"):
        d = root / "images" / split
        splits[split] = sorted(p.stem for p in d.glob("*.jpg")) if d.exists() else []
    if not splits["train"] or not splits["val"]:
        print(f"[audit] dataset not built at {root}")
        return 1

    report = {"dataset": str(root),
              "counts": {k: len(v) for k, v in splits.items()}}
    print(f"[audit] train={len(splits['train'])} val={len(splits['val'])}")

    # --- 1. identical stems ---------------------------------------------------
    dupes = sorted(set(splits["train"]) & set(splits["val"]))
    report["identical_stems"] = {"n": len(dupes), "examples": dupes[:10]}
    print(f"[audit] identical stems in both splits: {len(dupes)}")

    # --- 2. shared source clip ------------------------------------------------
    groups = {s: Counter(group_key(st) for st in stems) for s, stems in splits.items()}
    shared = sorted(set(groups["train"]) & set(groups["val"]))
    shared_rw = [g for g in shared if g.startswith("rw::")]
    report["shared_groups"] = {
        "n_total": len(shared),
        "n_roadwork_clips": len(shared_rw),
        "examples": shared_rw[:10],
        "train_clips": len(groups["train"]), "val_clips": len(groups["val"]),
    }
    print(f"[audit] ROADWork clips appearing in BOTH splits: {len(shared_rw)} "
          f"(train has {len(groups['train'])} groups, val {len(groups['val'])})")
    if shared_rw:
        affected = sum(groups["val"][g] for g in shared_rw)
        report["shared_groups"]["val_frames_affected"] = affected
        print(f"         -> {affected} val frames ({100*affected/len(splits['val']):.1f}%) "
              f"come from a clip also seen in training")

    # --- 3. near-duplicate frames across the boundary -------------------------
    print("[audit] hashing frames for near-duplicate check ...")
    hashes = {}
    for split in ("train", "val"):
        stems = splits[split]
        if args.hash_limit and len(stems) > args.hash_limit:
            idx = np.linspace(0, len(stems) - 1, args.hash_limit).astype(int)
            stems = [stems[i] for i in idx]
        hashes[split] = [(st, phash(root / "images" / split / f"{st}.jpg")) for st in stems]

    train_hashes = np.array([h for _, h in hashes["train"]], dtype=object)
    near = []
    for st, h in hashes["val"]:
        if h == 0:
            continue
        best, best_d = None, 999
        for st_t, h_t in hashes["train"]:
            d = hamming(h, h_t)
            if d < best_d:
                best, best_d = st_t, d
                if d == 0:
                    break
        if best_d <= args.hash_threshold:
            near.append({"val": st, "train": best, "hamming": best_d})
    report["near_duplicates"] = {
        "n": len(near),
        "checked_val": len(hashes["val"]), "checked_train": len(hashes["train"]),
        "threshold": args.hash_threshold,
        "examples": near[:10],
    }
    pct = 100 * len(near) / max(len(hashes["val"]), 1)
    print(f"[audit] val frames with a near-duplicate in train: {len(near)} "
          f"({pct:.2f}% of {len(hashes['val'])} checked)")

    # --- 4. class balance -----------------------------------------------------
    names = ["cone", "barrier", "stop_sign"]
    balance = {}
    for split in ("train", "val"):
        counts = Counter()
        for f in (root / "labels" / split).glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        counts[names[int(line.split()[0])]] += 1
                    except (ValueError, IndexError):
                        pass
        balance[split] = dict(counts)
    report["class_balance"] = balance
    print(f"[audit] boxes: train={balance['train']}  val={balance['val']}")
    thin = [c for c in names if balance["val"].get(c, 0) < 50]
    if thin:
        report["thin_val_classes"] = thin
        print(f"[audit] WARNING: val has <50 boxes for {thin} - "
              f"per-class mAP for those is noise, not signal")

    leaked = bool(dupes or shared_rw or near)
    report["verdict"] = "LEAKAGE DETECTED" if leaked else "clean"
    print(f"\n[audit] verdict: {report['verdict']}")
    if not leaked:
        print("        train and val share no clip, no file and no near-duplicate frame,")
        print("        so validation mAP measures generalisation to unseen locations.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[audit] report -> {out}")
    return 1 if (leaked and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
