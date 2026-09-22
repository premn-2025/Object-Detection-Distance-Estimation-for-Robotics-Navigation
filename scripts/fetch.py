"""Resumable parallel range-downloader.

A single long-lived HTTP connection to the dataset CDNs used here collapses to a
fraction of the available bandwidth after a few hundred megabytes, while several
shorter-lived ranged connections keep saturating the link. This downloads a file
as independent byte ranges, records which chunks are done, and can be re-run to
finish an interrupted transfer.

    python scripts/fetch.py <url> <dest> [--workers 12] [--chunk-mb 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests


def content_length(url: str, session: requests.Session) -> int:
    r = session.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    n = r.headers.get("Content-Length")
    if n is None:
        raise RuntimeError("server did not report Content-Length; cannot range-download")
    return int(n)


def download(url: str, dest: Path, workers: int = 12, chunk_mb: int = 32,
             retries: int = 8, quiet: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    state_path = dest.with_suffix(dest.suffix + ".parts")

    with requests.Session() as probe:
        total = content_length(url, probe)

    chunk = chunk_mb * 1024 * 1024
    ranges = [(i, s, min(s + chunk, total) - 1)
              for i, s in enumerate(range(0, total, chunk))]

    done: set = set()
    if state_path.exists() and dest.exists() and dest.stat().st_size == total:
        try:
            st = json.loads(state_path.read_text())
            if st.get("url") == url and st.get("total") == total:
                done = set(st.get("done", []))
        except (json.JSONDecodeError, OSError):
            done = set()

    if not dest.exists() or dest.stat().st_size != total:
        with open(dest, "wb") as f:       # preallocate so threads can seek freely
            f.truncate(total)
        done = set()

    todo = [r for r in ranges if r[0] not in done]
    if not todo:
        if not quiet:
            print(f"[fetch] already complete: {dest} ({total/1e9:.2f} GB)")
        return dest

    lock = threading.Lock()
    progress = {"bytes": sum(min(chunk, total - i * chunk) for i in done), "t0": time.time()}

    def save_state():
        state_path.write_text(json.dumps({"url": url, "total": total, "done": sorted(done)}))

    def work(item):
        idx, start, end = item
        for attempt in range(retries):
            try:
                with requests.Session() as s:     # a fresh connection per chunk
                    r = s.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(30, 120))
                    if r.status_code not in (206, 200):
                        raise requests.RequestException(f"status {r.status_code}")
                    buf = bytearray()
                    for block in r.iter_content(1024 * 256):
                        buf.extend(block)
                    if len(buf) != end - start + 1:
                        raise requests.RequestException(
                            f"short chunk {len(buf)} != {end - start + 1}")
                with lock:
                    with open(dest, "r+b") as f:
                        f.seek(start)
                        f.write(buf)
                    done.add(idx)
                    progress["bytes"] += len(buf)
                    save_state()
                    if not quiet:
                        el = time.time() - progress["t0"]
                        mb = progress["bytes"] / 1e6
                        rate = mb / max(el, 1e-6)
                        eta = (total / 1e6 - mb) / max(rate, 1e-6) / 60
                        sys.stdout.write(
                            f"\r[fetch] {mb:8.0f}/{total/1e6:.0f} MB  "
                            f"{rate:5.2f} MB/s  eta {eta:5.1f} min  "
                            f"({len(done)}/{len(ranges)} chunks)")
                        sys.stdout.flush()
                return True
            except (requests.RequestException, OSError):
                time.sleep(min(2 ** attempt, 20))
        return False

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, todo))
    if not quiet:
        print()
    if not all(results):
        raise RuntimeError(f"{results.count(False)} chunks failed; re-run to resume")
    save_state()
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("dest")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--chunk-mb", type=int, default=32)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    download(a.url, Path(a.dest), a.workers, a.chunk_mb, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
