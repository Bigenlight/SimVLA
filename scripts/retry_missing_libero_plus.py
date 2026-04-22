"""Resume-download: compare the task_ab_subset.json episode list against the
current on-disk files and download any missing parquet / mp4 with single-thread
exponential backoff (to avoid HF 429 rate limit).

Run inside simvla-http container the same way as download_libero_plus_taskAB.py.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "Sylvest/libero_plus_lerobot"
ROOT = Path("/libero_plus/data/libero_plus_lerobot")
CHUNK_SIZE = 1000


def expected_files() -> list[str]:
    subset = json.loads((ROOT / "meta" / "task_ab_subset.json").read_text())
    files: list[str] = []
    for key in ("task_a", "task_b"):
        for ep in subset[key]["episode_indices"]:
            c = ep // CHUNK_SIZE
            files.append(f"data/chunk-{c:03d}/episode_{ep:06d}.parquet")
            files.append(f"videos/chunk-{c:03d}/observation.images.front/episode_{ep:06d}.mp4")
            files.append(f"videos/chunk-{c:03d}/observation.images.wrist/episode_{ep:06d}.mp4")
    return files


def main() -> None:
    all_files = expected_files()
    missing = [p for p in all_files if not (ROOT / p).exists()]
    print(f"total expected: {len(all_files)}  missing: {len(missing)}")
    if not missing:
        print("nothing to do — all files present")
        return

    for i, rel in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {rel}")
        for attempt in range(12):
            try:
                hf_hub_download(REPO, rel, repo_type="dataset", local_dir=str(ROOT))
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg:
                    wait = min(120.0, 8.0 * (1.5 ** attempt)) + random.uniform(0, 4)
                    print(f"   429 backoff {wait:.1f}s (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    print(f"   hard fail: {e}")
                    break
        else:
            print("   gave up")

    # Final count
    all_files = expected_files()
    missing = [p for p in all_files if not (ROOT / p).exists()]
    print(f"done. final missing: {len(missing)}")
    for p in missing:
        print(f"   still missing: {p}")


if __name__ == "__main__":
    main()
