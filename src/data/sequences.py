"""Recovering temporal order from ROADWork file names.

ROADWork frames are named ``<city>_<video_id>_<sequence_id>_<frame_id>.jpg`` and
the converter prefixes them with ``rw_``.  The dataset is shipped as loose images
with no manifest, but that name is enough to reassemble the original clips, which
is what the optical-flow and motion-stereo demos need.

Frame ids are timestamps in milliseconds, so consecutive frames in a sequence are
typically 30 ms apart - close enough for Lucas-Kanade, far enough apart to give a
usable stereo baseline at driving speed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# rw_<city>_<32-hex video id>_<6-digit seq>_<frame id>
PATTERN = re.compile(r"^rw_(?P<city>[a-z_]+)_(?P<vid>[0-9a-f]{8,})_(?P<seq>\d+)_(?P<frame>\d+)$")


def parse_stem(stem: str):
    m = PATTERN.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {"city": d["city"], "video": d["vid"], "seq": d["seq"],
            "frame": int(d["frame"]), "stem": stem}


def find_sequences(image_dir, min_length: int = 4) -> list:
    """Group images into time-ordered clips. Longest first."""
    image_dir = Path(image_dir)
    groups = defaultdict(list)
    for p in image_dir.glob("rw_*.jpg"):
        info = parse_stem(p.stem)
        if info is None:
            continue
        groups[(info["city"], info["video"], info["seq"])].append((info["frame"], p))

    sequences = []
    for key, items in groups.items():
        if len(items) < min_length:
            continue
        items.sort()
        sequences.append({
            "key": "_".join(key),
            "frames": [p for _, p in items],
            "frame_ids": [f for f, _ in items],
            "length": len(items),
        })
    sequences.sort(key=lambda s: -s["length"])
    return sequences


def frame_interval_s(frame_ids, fps_hint: float = 30.0) -> float:
    """Median spacing between frames, in seconds.

    ROADWork frame ids are millisecond timestamps; if they do not look like that
    (non-monotonic, or absurd spacing) fall back to ``fps_hint``.
    """
    if len(frame_ids) < 2:
        return 1.0 / fps_hint
    diffs = [b - a for a, b in zip(frame_ids[:-1], frame_ids[1:]) if b > a]
    if not diffs:
        return 1.0 / fps_hint
    diffs.sort()
    median_ms = diffs[len(diffs) // 2]
    dt = median_ms / 1000.0
    return dt if 0.005 <= dt <= 2.0 else 1.0 / fps_hint


def load_sequence(sequence, limit: int = 0):
    """Yield ``(path, bgr_frame)`` in temporal order."""
    import cv2

    frames = sequence["frames"][:limit] if limit else sequence["frames"]
    for p in frames:
        img = cv2.imread(str(p))
        if img is not None:
            yield p, img
