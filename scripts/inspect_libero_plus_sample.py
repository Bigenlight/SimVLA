"""Inspect one Task A episode from `Sylvest/libero_plus_lerobot` to pin down the
exact schema before writing the parquet→HDF5 conversion script.

Answers these questions:
  (1) observation.state: is dim 3-5 Euler or axis-angle?   (critical for SimVLA)
  (2) gripper range: ±1 binary, or continuous 0..0.04?
  (3) mp4 codec + decode works with pyav inside the simvla container?
  (4) image shape after decode: matches meta/info.json?
  (5) action chunk alignment: parquet frame_index[-1] == episode length - 1?

Run inside the simvla-http container with LIBERO-plus bind-mounted at /libero_plus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def first_episode_paths(root: Path, task_key: str = "task_a"):
    subset = json.loads((root / "meta" / "task_ab_subset.json").read_text())
    meta = subset[task_key]
    ep_idx = meta["episode_indices"][0]
    chunk = ep_idx // 1000
    parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    mp4_front = (
        root / "videos" / f"chunk-{chunk:03d}" / "observation.images.front"
        / f"episode_{ep_idx:06d}.mp4"
    )
    mp4_wrist = (
        root / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist"
        / f"episode_{ep_idx:06d}.mp4"
    )
    return ep_idx, meta["task_str"], parquet, mp4_front, mp4_wrist


def inspect_parquet(pq_path: Path):
    print("=" * 70)
    print(f"PARQUET  {pq_path.name}")
    print("=" * 70)
    tbl = pq.read_table(str(pq_path))
    print(f"columns: {tbl.column_names}")
    print(f"schema:\n{tbl.schema}")
    # sort by frame_index
    frame_index = tbl.column("frame_index").to_numpy()
    order = np.argsort(frame_index)
    frame_index = frame_index[order]
    print(f"n_frames = {len(frame_index)}")
    print(f"frame_index[0..5] = {frame_index[:5].tolist()}")
    print(f"frame_index[-5:] = {frame_index[-5:].tolist()}")

    # observation.state and action are list<item: float32> in parquet — each cell is a list
    state_rows = tbl.column("observation.state").to_pylist()
    action_rows = tbl.column("action").to_pylist()
    state = np.asarray([state_rows[i] for i in order], dtype=np.float32)  # [T, 8]
    action = np.asarray([action_rows[i] for i in order], dtype=np.float32)  # [T, 7]
    print(f"state shape={state.shape} dtype={state.dtype}")
    print(f"action shape={action.shape} dtype={action.dtype}")

    print("\n--- state per-dim stats (min, max, mean, std) ---")
    for i in range(state.shape[1]):
        col = state[:, i]
        print(
            f"  state[:,{i}] "
            f"min={col.min():+.4f} max={col.max():+.4f} "
            f"mean={col.mean():+.4f} std={col.std():.4f}"
        )
    print("\n--- action per-dim stats ---")
    for i in range(action.shape[1]):
        col = action[:, i]
        print(
            f"  action[:,{i}] "
            f"min={col.min():+.4f} max={col.max():+.4f} "
            f"mean={col.mean():+.4f} std={col.std():.4f}"
        )

    # Gripper heuristic: action[:, 6] is gripper → is it binary ±1?
    gripper = action[:, 6]
    uniq = np.unique(np.round(gripper, 2))
    print(f"\n  action[:,6] (gripper) unique(2dp) head: {uniq[:15]}")
    print(f"  action[:,6] discrete? abs>=0.9: {(np.abs(gripper) >= 0.9).mean()*100:.0f}%")

    # Euler vs axis-angle check for state[:, 3:6]
    ori = state[:, 3:6]
    norms = np.linalg.norm(ori, axis=1)
    print(f"\n  state[:,3:6] (orientation) ||v||  min={norms.min():.4f} "
          f"max={norms.max():.4f} mean={norms.mean():.4f}")
    print("  Interpretation hint:")
    print("    - Euler angles: each component ∈ [-π, π], no obvious constraint on norm")
    print("    - Axis-angle:   norm = rotation magnitude, typically < π ≈ 3.14")

    return state, action


def inspect_mp4(mp4_path: Path, n_frames: int = 3):
    print("\n" + "=" * 70)
    print(f"MP4  {mp4_path.name}")
    print("=" * 70)
    import av

    with av.open(str(mp4_path)) as c:
        stream = c.streams.video[0]
        print(f"codec={stream.codec_context.name}  "
              f"pix_fmt={stream.codec_context.pix_fmt}  "
              f"fps={float(stream.average_rate):.2f}  "
              f"n_frames={stream.frames}  "
              f"size={stream.codec_context.width}x{stream.codec_context.height}")

        # Decode first n_frames to confirm
        frames = []
        for i, frame in enumerate(c.decode(stream)):
            arr = frame.to_ndarray(format="rgb24")
            frames.append(arr)
            if i + 1 == n_frames:
                break
        for i, arr in enumerate(frames):
            print(f"  frame[{i}] shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/libero_plus/data/libero_plus_lerobot")
    p.add_argument("--task", choices=["task_a", "task_b"], default="task_a")
    args = p.parse_args()

    root = Path(args.root)
    ep_idx, task_str, pq_path, mp4_front, mp4_wrist = first_episode_paths(root, args.task)
    print(f"### Inspecting {args.task} ep_idx={ep_idx}  task='{task_str}'")
    print(f"    parquet:   {pq_path}  ({pq_path.stat().st_size/1024:.1f} KB)")
    print(f"    front mp4: {mp4_front}  ({mp4_front.stat().st_size/1024:.1f} KB)")
    print(f"    wrist mp4: {mp4_wrist}  ({mp4_wrist.stat().st_size/1024:.1f} KB)")

    state, action = inspect_parquet(pq_path)
    inspect_mp4(mp4_front)
    inspect_mp4(mp4_wrist)

    # Also peek at info.json feature names
    info = json.loads((root / "meta" / "info.json").read_text())
    print("\n=== meta/info.json observation.state + action features ===")
    print(json.dumps(info.get("features", {}).get("observation.state", {}), indent=2))
    print(json.dumps(info.get("features", {}).get("action", {}), indent=2))


if __name__ == "__main__":
    main()
