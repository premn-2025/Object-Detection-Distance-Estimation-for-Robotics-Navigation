"""Rebuild the train/val split so that no video clip straddles it.

`scripts/check_leakage.py` found that ROADWork's official ``gps_split`` shares no
*images* between train and val, but does share 222 *video clips* â€" 35 % of the
validation frames came from a clip the model had also trained on. Frames 30 ms
apart are nearly the same picture, so that validation score was measuring
memorisation and would have read several points too high, with completely healthy
looking loss curves.

This regroups every image by its source clip and assigns whole clips to one side
of the split. Two properties matter:

* **group-disjoint** â€" a clip is entirely in train or entirely in val;
* **deterministic** â€" the assignment is a hash of the clip name, so re-running
  the pipeline, or re-mining with a different threshold, cannot silently move an
  image across the boundary (the bug that produced the duplicate files).

Clips are filled greedily by size into the validation side until the target
frame fraction is reached, so the split hits the requested ratio rather than
whatever the hash happens to give.

    python scripts/resplit.py --dataset data/processed/nav --val-fraction 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sequences import parse_stem  # noqa: E402


SEQ_STEM = re.compile(r"^rw_(?P<prefix>.+)_(?P<idx>\d+)$")


def build_groups(stems, *, run_gap: int = 10, max_run: int = 250,
                 embargo: int = 8) -> tuple:
    """Map every stem to a (source, group) that never straddles the split.

    Three naming schemes appear in ROADWork and all three are temporal:

    * ``rw_<city>_<video>_<seq>_<frame>`` - grouped by city+video+sequence.
    * ``rw_pgh04_1478`` - one long continuous Pittsburgh drive, consecutive
      indices (median gap 1).
    * ``rw_IMG_3931`` - photo bursts, consecutive with large gaps between
      sessions.

    The last two are grouped by *contiguous index run*: sort by index and start a
    new run wherever the gap exceeds ``run_gap``. A run longer than ``max_run``
    is chunked further, and ``embargo`` frames at each chunk boundary are dropped
    from the dataset entirely - otherwise the last frame of a train chunk sits
    directly beside the first frame of a val chunk, which is the leak this whole
    exercise is about. Losing a few percent of frames is the right price.

    Returns ``(assignment, dropped)``.
    """
    groups = {}
    dropped = set()
    seq_by_prefix = defaultdict(list)

    for stem in stems:
        info = parse_stem(stem)
        if info:
            groups[stem] = ("roadwork", f"{info['city']}_{info['video']}_{info['seq']}")
            continue
        if stem.startswith("bddstop_"):
            groups[stem] = ("bdd", stem)
            continue
        if stem.startswith("coco_"):
            groups[stem] = ("coco", stem)
            continue
        m = SEQ_STEM.match(stem)
        if m:
            seq_by_prefix[m.group("prefix")].append((int(m.group("idx")), stem))
        else:
            groups[stem] = ("other", stem)

    for prefix, items in seq_by_prefix.items():
        items.sort()
        run_id, chunk_start = 0, 0
        run = [items[0]]
        for prev, cur in zip(items[:-1], items[1:]):
            if cur[0] - prev[0] > run_gap or len(run) - chunk_start >= max_run:
                _close_run(run[chunk_start:], prefix, run_id, groups, dropped, embargo)
                run_id += 1
                chunk_start = len(run)
            run.append(cur)
        _close_run(run[chunk_start:], prefix, run_id, groups, dropped, embargo)

    return groups, dropped


def _close_run(items, prefix, run_id, groups, dropped, embargo):
    """Assign one contiguous run, embargoing its trailing frames."""
    if not items:
        return
    keep = items[:-embargo] if embargo and len(items) > 2 * embargo else items
    for _, stem in keep:
        groups[stem] = ("roadwork_seq", f"{prefix}#run{run_id}")
    for _, stem in items[len(keep):]:
        dropped.add(stem)


def _dhash(path, size: int = 8) -> int:
    """64-bit difference hash of an image."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = 0
    for b in (small[:, 1:] > small[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return bits


def merge_duplicate_groups(located, keyed, root, *, threshold: int = 3,
                           verbose: bool = True) -> dict:
    """Union groups that contain visually identical images.

    ROADWork ships some captures twice under different naming schemes - the
    ``pgh0N_####`` series and the ``IMG_####`` series overlap, byte-for-byte
    identical frames with unrelated names. Name-based grouping cannot see that,
    so the duplicates land on opposite sides of the split and the leak survives.

    Near-duplicates are found with multi-index hashing: two 64-bit hashes within
    Hamming distance ``threshold`` must agree on at least one of four 16-bit
    chunks (pigeonhole), so bucketing by chunk turns an O(n^2) scan into a few
    thousand comparisons. Matching groups are then merged with union-find.
    """
    from collections import defaultdict

    hashes = {}
    for stem, split in located.items():
        h = _dhash(root / "images" / split / f"{stem}.jpg")
        if h >= 0:
            hashes[stem] = h

    parent = {k: k for k in set(keyed.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets = defaultdict(list)
    for stem, h in hashes.items():
        for i in range(4):
            buckets[(i, (h >> (16 * i)) & 0xFFFF)].append(stem)

    merged = 0
    seen = set()
    for bucket in buckets.values():
        if len(bucket) < 2 or len(bucket) > 400:
            continue
        for i, a in enumerate(bucket):
            for b in bucket[i + 1:]:
                pair = (a, b) if a < b else (b, a)
                if pair in seen:
                    continue
                seen.add(pair)
                if bin(hashes[a] ^ hashes[b]).count("1") > threshold:
                    continue
                ga, gb = keyed.get(a), keyed.get(b)
                if ga and gb and find(ga) != find(gb):
                    union(ga, gb)
                    merged += 1
    if verbose and merged:
        print(f"[resplit] merged {merged} group pairs holding visually identical "
              f"images under different names")
    return {stem: find(g) for stem, g in keyed.items()}


def hash_rank(key: str, seed: int = 0) -> float:
    h = hashlib.blake2s(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/processed/nav")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", default="outputs/metrics/resplit.json")
    args = p.parse_args(argv)

    root = Path(args.dataset)

    # ---- inventory: stem -> (current split, group), deduplicating stale copies
    located: dict = {}
    duplicates = []
    for split in ("train", "val"):
        for img in sorted((root / "images" / split).glob("*.jpg")):
            if img.stem in located:
                duplicates.append(img.stem)
            located[img.stem] = split
    print(f"[resplit] {len(located)} unique images "
          f"({len(duplicates)} appeared in both splits)")

    keyed, dropped = build_groups(sorted(located))
    if dropped:
        print(f"[resplit] embargoing {len(dropped)} frames at run boundaries "
              f"so train and val chunks are never adjacent")
    keyed = merge_duplicate_groups(located, keyed, root)
    groups = defaultdict(list)
    for stem in located:
        if stem in dropped:
            continue
        groups[keyed[stem]].append(stem)
    # ---- assign whole groups, per source, greedily to hit the target fraction
    assignment: dict = {}
    summary = {}
    for src in sorted({s for s, _ in groups}):
        keys = [k for k in groups if k[0] == src]
        total = sum(len(groups[k]) for k in keys)
        target = args.val_fraction * total
        # deterministic order, then greedy fill: reproducible and hits the ratio
        keys.sort(key=lambda k: hash_rank(k[1], args.seed))
        val_n = 0
        n_val_groups = 0
        for k in keys:
            if val_n < target:
                for stem in groups[k]:
                    assignment[stem] = "val"
                val_n += len(groups[k])
                n_val_groups += 1
            else:
                for stem in groups[k]:
                    assignment[stem] = "train"
        summary[src] = {"groups": len(keys), "val_groups": n_val_groups,
                        "images": total, "val_images": val_n,
                        "val_fraction": round(val_n / max(total, 1), 4)}
        print(f"[resplit] {src:9s} {total:5d} images in {len(keys):5d} groups "
              f"-> val {val_n} ({100*val_n/max(total,1):.1f}%) from "
              f"{n_val_groups} groups")

    moves = [(s, located[s], assignment[s]) for s in located
             if s in assignment and located[s] != assignment[s]]
    print(f"[resplit] {len(moves)} images change split")
    if args.dry_run:
        print("[resplit] dry run - nothing written")
        return 0

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # embargoed frames leave the dataset entirely
    for stem in dropped:
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            for split in ("train", "val"):
                f = root / kind / split / f"{stem}{ext}"
                if f.exists():
                    f.unlink()

    for stem, old, new in moves:
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            src_p = root / kind / old / f"{stem}{ext}"
            dst_p = root / kind / new / f"{stem}{ext}"
            if src_p.exists():
                shutil.move(str(src_p), str(dst_p))
    # remove any stale copy left in the other split by an earlier run
    removed = 0
    for stem, split in assignment.items():
        other = "val" if split == "train" else "train"
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            stale = root / kind / other / f"{stem}{ext}"
            if stale.exists():
                stale.unlink()
                removed += 1
    if removed:
        print(f"[resplit] removed {removed} stale duplicate files")

    from scripts.prepare_data import recount

    import yaml

    names = [yaml.safe_load((root / "data.yaml").read_text())["names"][i] for i in range(3)]
    stats = recount(root, names)
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    report = {"val_fraction_target": args.val_fraction, "per_source": summary,
              "images_moved": len(moves), "stale_removed": removed,
              "embargoed_frames": len(dropped), "stats": stats}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[resplit] new composition:")
    for split, v in stats.items():
        print(f"  {split}: images={v['images']} neg={v['negative_images']} "
              f"boxes={v['boxes']}")
    print(f"[resplit] report -> {args.report}")
    print("[resplit] now re-run: python scripts/check_leakage.py --dataset "
          f"{args.dataset} --strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
