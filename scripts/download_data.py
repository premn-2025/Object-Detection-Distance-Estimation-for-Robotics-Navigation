"""Fetch every dataset this project needs, resumably.

    python scripts/download_data.py --all
    python scripts/download_data.py --roadwork          # just one source

Sizes: ROADWork images 10.6 GB, ROADWork annotations 67 MB, BDD100K val 570 MB,
COCO annotations 253 MB.  Everything lands in ``data/raw/``.

Downloads go through ``scripts/fetch.py`` rather than a plain ``requests.get``
because a single long-lived connection to these CDNs degrades badly partway
through a multi-gigabyte transfer; ranged parallel chunks keep the link busy and
make an interrupted transfer resumable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch import download  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"

SOURCES = {
    "roadwork-annotations": {
        "url": HF.format(repo="anuragxel/roadwork-dataset", name="annotations.zip"),
        "dest": "data/raw/annotations.zip",
        "size": "67 MB",
        "note": "COCO-format boxes for the ROADWork images",
        "workers": 6, "chunk_mb": 16,
    },
    "roadwork": {
        "url": HF.format(repo="anuragxel/roadwork-dataset", name="images.zip"),
        "dest": "data/raw/images.zip",
        "size": "10.6 GB",
        "note": "8550 dash-cam work-zone frames at 1920x1080",
        "workers": 16, "chunk_mb": 32,
    },
    "bdd100k": {
        "url": HF.format(repo="hirundo-io/bdd100k-val",
                         name="bdd100k_val_hirundo.zip"),
        "dest": "data/raw/bdd100k_val.zip",
        "size": "570 MB",
        "note": "BDD100K official val split, 10k frames + box labels",
        "workers": 8, "chunk_mb": 16,
    },
    "coco-annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "dest": "data/raw/coco_ann.zip",
        "size": "253 MB",
        "note": "only instances_*2017.json is used, for the stop-sign class",
        "workers": 6, "chunk_mb": 16,
    },
}

# smallest first, so a slow link still produces something usable early
ORDER = ["roadwork-annotations", "bdd100k", "coco-annotations", "roadwork"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true")
    for key in SOURCES:
        p.add_argument(f"--{key}", action="store_true",
                       help=f"{SOURCES[key]['size']} - {SOURCES[key]['note']}")
    p.add_argument("--list", action="store_true", help="show sources and exit")
    p.add_argument("--workers", type=int, default=None, help="override worker count")
    args = p.parse_args(argv)

    if args.list:
        for key in ORDER:
            s = SOURCES[key]
            print(f"  {key:22s} {s['size']:>8s}  {s['dest']}")
            print(f"  {'':22s}          {s['note']}")
        return 0

    selected = [k for k in ORDER
                if args.all or getattr(args, k.replace("-", "_"), False)]
    if not selected:
        p.error("nothing selected; pass --all or one of the source flags (--list to see them)")

    for key in selected:
        s = SOURCES[key]
        dest = Path(s["dest"])
        print(f"\n=== {key} ({s['size']}) -> {dest} ===")
        try:
            download(s["url"], dest, workers=args.workers or s["workers"],
                     chunk_mb=s["chunk_mb"])
            print(f"[ok] {dest} ({dest.stat().st_size/1e9:.2f} GB)")
        except Exception as exc:                  # noqa: BLE001
            print(f"[fail] {key}: {type(exc).__name__}: {exc}")
            print("       re-run the same command to resume from where it stopped")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
