"""Download only Task A + Task B episodes from `Sylvest/libero_plus_lerobot`.

Target layout under LIBERO-plus/data/libero_plus_lerobot:
    meta/{info,tasks,episodes,episodes_stats}.{json,jsonl}
    norm_stats.json
    data/chunk-XXX/episode_YYYYYY.parquet
    videos/chunk-XXX/observation.images.{front,wrist}/episode_YYYYYY.mp4

Task A = KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
Task B = KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it

Run inside the simvla-http container (which already has huggingface_hub):
    docker run --rm --network host \
      -v /home/theo/workspace/SimVLA:/app \
      -v /home/theo/workspace/LIBERO-plus:/libero_plus \
      -v /home/theo/.cache/huggingface:/hf_cache \
      -e HF_HOME=/hf_cache \
      --entrypoint python bigenlight/simvla-http:latest \
      /app/scripts/download_libero_plus_taskAB.py
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "Sylvest/libero_plus_lerobot"
CHUNK_SIZE = 1000  # from meta/info.json

TASK_A_STR = "turn on the stove and put the moka pot on it"
TASK_B_STR = "put the black bowl in the bottom drawer of the cabinet and close it"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/libero_plus/data/libero_plus_lerobot",
        help="Local destination root (inside container convention).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers for snapshot_download.",
    )
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # --- 1) Metadata first (small — ~37 MB incl episodes_stats) --------------
    meta_files = [
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "norm_stats.json",
        "README.md",
    ]
    print(f"[1/3] downloading meta → {root}")
    for f in meta_files:
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(root))
        except Exception as e:  # README etc. may not exist; don't fail
            print(f"   skip {f}: {e}")

    # --- 2) Resolve target episodes ------------------------------------------
    tasks = [json.loads(l) for l in open(root / "meta" / "tasks.jsonl")]
    task_a = next((t for t in tasks if t["task"] == TASK_A_STR), None)
    task_b = next((t for t in tasks if t["task"] == TASK_B_STR), None)
    assert task_a is not None, f"Task A not in tasks.jsonl: {[t['task'] for t in tasks]}"
    assert task_b is not None, f"Task B not in tasks.jsonl"
    target_task_strs = {TASK_A_STR, TASK_B_STR}
    print(f"   Task A → task_index={task_a['task_index']}")
    print(f"   Task B → task_index={task_b['task_index']}")

    episodes = [json.loads(l) for l in open(root / "meta" / "episodes.jsonl")]
    # episodes.jsonl shape: {episode_index, tasks:[str], length}
    ep_a = [e for e in episodes if e["tasks"] and e["tasks"][0] == TASK_A_STR]
    ep_b = [e for e in episodes if e["tasks"] and e["tasks"][0] == TASK_B_STR]
    print(f"   Task A episodes: {len(ep_a)}  |  Task B episodes: {len(ep_b)}")
    assert ep_a and ep_b, "no episodes matched — check task string spelling"

    # --- 3) Build allow_patterns ---------------------------------------------
    patterns: list[str] = []
    for e in ep_a + ep_b:
        ep_idx = e["episode_index"]
        chunk = ep_idx // CHUNK_SIZE
        patterns.append(f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet")
        patterns.append(
            f"videos/chunk-{chunk:03d}/observation.images.front/episode_{ep_idx:06d}.mp4"
        )
        patterns.append(
            f"videos/chunk-{chunk:03d}/observation.images.wrist/episode_{ep_idx:06d}.mp4"
        )
    # Also save the filtered task subset for later metadata generation
    subset = {
        "task_a": {
            "task_index": task_a["task_index"],
            "task_str": TASK_A_STR,
            "episode_indices": [e["episode_index"] for e in ep_a],
            "lengths": [e["length"] for e in ep_a],
        },
        "task_b": {
            "task_index": task_b["task_index"],
            "task_str": TASK_B_STR,
            "episode_indices": [e["episode_index"] for e in ep_b],
            "lengths": [e["length"] for e in ep_b],
        },
        "source_repo": REPO,
    }
    (root / "meta" / "task_ab_subset.json").write_text(json.dumps(subset, indent=2))
    print(f"   wrote subset index → meta/task_ab_subset.json")

    # --- 4) Bulk download via direct parallel hf_hub_download -----------------
    # NOTE: snapshot_download(allow_patterns=[...]) is O(N_repo_files × N_patterns)
    # on the client side (28k × 1900 for this repo ≈ CPU-bound for minutes). Direct
    # parallel hf_hub_download is ~100× faster for targeted subsets. hf_hub_download
    # also skips files whose etag already matches the local cache — so re-runs only
    # fetch what's missing.
    #
    # HF has a per-minute xet-read-token rate limit; we back off on 429 with jitter.
    import random
    import time

    def _dl(rel_path: str, max_retries: int = 6) -> str:
        last = None
        for attempt in range(max_retries):
            try:
                return hf_hub_download(
                    REPO, rel_path, repo_type="dataset", local_dir=str(root)
                )
            except Exception as e:
                msg = str(e)
                last = e
                if "429" in msg or "Too Many Requests" in msg:
                    # exponential backoff with jitter: 4s, 8s, 16s, 32s, 60s, 60s
                    wait = min(60.0, 4.0 * (2 ** attempt)) + random.uniform(0, 2)
                    time.sleep(wait)
                    continue
                # non-retryable error
                raise
        raise last  # type: ignore[misc]

    print(f"[2/3] parallel hf_hub_download ({len(patterns)} files, workers={args.max_workers})")
    n_done = 0
    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(_dl, p): p for p in patterns}
        for fut in as_completed(futures):
            try:
                fut.result()
                n_done += 1
            except Exception as e:
                n_fail += 1
                if n_fail < 10:
                    print(f"   FAIL {futures[fut]}: {e}")
            if n_done % 50 == 0 and n_done > 0:
                print(f"   progress: {n_done}/{len(patterns)} (fails={n_fail})")
    print(f"   downloaded {n_done}/{len(patterns)} (fails={n_fail})")

    # --- 5) Verify ------------------------------------------------------------
    n_parquet = sum(1 for _ in root.glob("data/**/*.parquet"))
    n_mp4 = sum(1 for _ in root.glob("videos/**/*.mp4"))
    print(f"[3/3] done.  parquet={n_parquet}  mp4={n_mp4}  (root={root})")
    expected_parquet = len(ep_a) + len(ep_b)
    expected_mp4 = expected_parquet * 2
    if n_parquet != expected_parquet or n_mp4 != expected_mp4:
        print(
            f"WARNING: expected parquet={expected_parquet} mp4={expected_mp4} — "
            "rerun if short (snapshot_download retries on its own)."
        )


if __name__ == "__main__":
    main()
