"""Convert `Sylvest/libero_plus_lerobot` Task A / Task B episodes into SimVLA-
compatible LIBERO HDF5 files.

Output layout (one HDF5 file per task, matches SimVLA's existing LIBERO convention
and `create_libero_meta.py` filename parser — `SCENE\\d+_` prefix + `_demo.hdf5`
suffix are stripped to derive the task string):

    <out_dir>/KITCHEN_SCENE3_turn_on_the_stove_..._demo.hdf5
    <out_dir>/KITCHEN_SCENE4_put_the_black_bowl_..._demo.hdf5

Each file has:
    data/demo_<i>/actions                  [T, 7]  float32
    data/demo_<i>/obs/agentview_rgb        [T, H, W, 3]  uint8  (NO rotation — loader rotates)
    data/demo_<i>/obs/eye_in_hand_rgb      [T, H, W, 3]  uint8
    data/demo_<i>/obs/ee_pos               [T, 3]  float32
    data/demo_<i>/obs/ee_ori               [T, 3]  float32  EULER xyz (axis-angle → euler)
    data/demo_<i>/obs/gripper_states       [T, 2]  float32

Rationale: `datasets/domain_handler/libero_hdf5.py` reads `obs/ee_ori` as Euler
and then does Euler → axis-angle internally (libero_hdf5.py:206) to yield the 8D
proprio `[ee_pos(3), axisangle(3), gripper(2)]`. The LeRobot dataset stores
state[:,3:6] already as AXIS-ANGLE. To reuse the existing loader verbatim we
round-trip axis-angle → euler → axis-angle. Singularities (gimbal lock) are
unlikely for LIBERO table-top manipulation trajectories.

Run inside simvla-http container:
    docker run --rm --network host \\
      -v /home/theo/workspace/SimVLA:/app \\
      -v /home/theo/workspace/LIBERO-plus:/libero_plus \\
      --entrypoint python bigenlight/simvla-http:latest \\
      /app/scripts/lerobot_to_libero_hdf5.py
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("convert")


LEROBOT_ROOT_DEFAULT = "/libero_plus/data/libero_plus_lerobot"
OUT_DIR_DEFAULT = "/libero_plus/data/libero_plus_hdf5"
CHUNK_SIZE = 1000

# Final HDF5 filename stem → what the task-name derivation strips back to
#   KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
#     → strip _demo.hdf5 → KITCHEN_SCENE3_turn_on_the_stove_...
#     → strip SCENE\d+_  → KITCHEN_turn_on_the_stove_...
#     → replace _ with ' '
# Matches create_libero_meta.py:26-35.
TASK_A_STEM = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
TASK_B_STEM = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
)


# --------------------------------------------------------------------------- #
# rotvec → Euler  (scipy-free to avoid adding a heavy dep in the container)   #
# --------------------------------------------------------------------------- #
def rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    """Rodrigues. rv shape [..., 3] → rotmat [..., 3, 3]."""
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    small = theta < 1e-8
    # Safe axis for small theta; value unused due to sin≈0
    axis = np.where(small, np.array([1.0, 0.0, 0.0]), rv / np.where(small, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(theta).squeeze(-1)
    s = np.sin(theta).squeeze(-1)
    C = 1.0 - c

    R = np.stack(
        [
            np.stack([c + x * x * C,     x * y * C - z * s, x * z * C + y * s], axis=-1),
            np.stack([y * x * C + z * s, c + y * y * C,     y * z * C - x * s], axis=-1),
            np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], axis=-1),
        ],
        axis=-2,
    )
    # Replace identity for tiny-rotation cases
    small_sq = small.squeeze(-1)
    if small_sq.any():
        eye = np.eye(3)
        R = np.where(small_sq[..., None, None], eye, R)
    return R


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Intrinsic xyz Euler (matches scipy.Rotation.as_euler('xyz')). Input [...,3,3]."""
    sy = R[..., 0, 2]
    sy = np.clip(sy, -1.0, 1.0)
    y = np.arcsin(sy)
    # gimbal-safe branches
    gimbal = np.abs(sy) > 1.0 - 1e-6
    x_normal = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    z_normal = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    x_gimbal = np.arctan2(R[..., 2, 1], R[..., 1, 1])
    z_gimbal = np.zeros_like(x_gimbal)
    x = np.where(gimbal, x_gimbal, x_normal)
    z = np.where(gimbal, z_gimbal, z_normal)
    return np.stack([x, y, z], axis=-1)


def axisangle_to_euler(rv: np.ndarray) -> np.ndarray:
    return rotmat_to_euler_xyz(rotvec_to_rotmat(rv))


# --------------------------------------------------------------------------- #
# mp4 decoding                                                                 #
# --------------------------------------------------------------------------- #
def decode_mp4(path: Path, n_expected: int) -> np.ndarray:
    """Decode an mp4 to [T, H, W, 3] uint8."""
    import av  # lazy so ProcessPoolExecutor can fork cleanly

    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        frames: List[np.ndarray] = []
        for frame in c.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))

    arr = np.stack(frames, axis=0)  # [T, H, W, 3]
    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"frame count mismatch at {path.name}: got {arr.shape[0]} expected {n_expected}"
        )
    return arr


# --------------------------------------------------------------------------- #
# parquet → per-episode dict                                                   #
# --------------------------------------------------------------------------- #
def load_episode(
    lerobot_root: Path, ep_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunk = ep_idx // CHUNK_SIZE
    pq_path = (
        lerobot_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    )
    mp4_front = (
        lerobot_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.front"
        / f"episode_{ep_idx:06d}.mp4"
    )
    mp4_wrist = (
        lerobot_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.wrist"
        / f"episode_{ep_idx:06d}.mp4"
    )

    tbl = pq.read_table(str(pq_path))
    frame_index = tbl.column("frame_index").to_numpy()
    order = np.argsort(frame_index)

    state_rows = tbl.column("observation.state").to_pylist()
    action_rows = tbl.column("action").to_pylist()
    state = np.asarray([state_rows[i] for i in order], dtype=np.float32)  # [T, 8]
    action = np.asarray([action_rows[i] for i in order], dtype=np.float32)  # [T, 7]

    T = state.shape[0]
    front = decode_mp4(mp4_front, n_expected=T)
    wrist = decode_mp4(mp4_wrist, n_expected=T)
    return state, action, front, wrist


def _worker(args: Tuple[str, int, int]) -> Tuple[int, Dict]:
    """Run in a subprocess: decode + convert, return demo payload for HDF5 write."""
    lerobot_root_str, ep_idx, demo_slot = args
    state, action, front, wrist = load_episode(Path(lerobot_root_str), ep_idx)

    ee_pos = state[:, 0:3].astype(np.float32)
    axisangle = state[:, 3:6].astype(np.float32)
    ee_ori_euler = axisangle_to_euler(axisangle.astype(np.float64)).astype(np.float32)
    gripper = state[:, 6:8].astype(np.float32)

    return demo_slot, {
        "actions": action,
        "agentview_rgb": front,
        "eye_in_hand_rgb": wrist,
        "ee_pos": ee_pos,
        "ee_ori": ee_ori_euler,
        "gripper_states": gripper,
    }


def write_task_hdf5(
    lerobot_root: Path,
    out_path: Path,
    episode_indices: List[int],
    max_workers: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".hdf5.partial")
    logger.info("writing %s (%d episodes)", out_path.name, len(episode_indices))

    # Prefill expected structure
    with h5py.File(tmp_path, "w") as hf:
        grp_data = hf.create_group("data")
        # Reserve slots so writes can happen in arrival order
        for i in range(len(episode_indices)):
            grp_data.create_group(f"demo_{i}")

        # Parallel decode + sequential HDF5 write
        jobs = [(str(lerobot_root), ep, i) for i, ep in enumerate(episode_indices)]
        n_done = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_worker, job) for job in jobs]
            for fut in as_completed(futures):
                demo_slot, d = fut.result()
                g = grp_data[f"demo_{demo_slot}"]
                g.create_dataset("actions", data=d["actions"], compression="gzip", compression_opts=4)
                obs = g.create_group("obs")
                obs.create_dataset(
                    "agentview_rgb", data=d["agentview_rgb"], compression="gzip", compression_opts=4
                )
                obs.create_dataset(
                    "eye_in_hand_rgb",
                    data=d["eye_in_hand_rgb"],
                    compression="gzip",
                    compression_opts=4,
                )
                obs.create_dataset("ee_pos", data=d["ee_pos"])
                obs.create_dataset("ee_ori", data=d["ee_ori"])
                obs.create_dataset("gripper_states", data=d["gripper_states"])
                n_done += 1
                if n_done % 20 == 0:
                    logger.info("  %d/%d done", n_done, len(episode_indices))

    tmp_path.rename(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("  wrote %s  (%.1f MB)", out_path.name, size_mb)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-root", default=LEROBOT_ROOT_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--task", choices=["a", "b", "both"], default="both", help="Which task(s) to convert."
    )
    p.add_argument(
        "--max-eps-per-task",
        type=int,
        default=None,
        help="Limit number of episodes per task (smoke-test only).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lerobot_root = Path(args.lerobot_root)
    out_dir = Path(args.out_dir)

    subset = json.loads((lerobot_root / "meta" / "task_ab_subset.json").read_text())

    def _limit(eps: List[int]) -> List[int]:
        return eps if args.max_eps_per_task is None else eps[: args.max_eps_per_task]

    if args.task in ("a", "both"):
        write_task_hdf5(
            lerobot_root,
            out_dir / f"{TASK_A_STEM}_demo.hdf5",
            _limit(subset["task_a"]["episode_indices"]),
            max_workers=args.max_workers,
        )
    if args.task in ("b", "both"):
        write_task_hdf5(
            lerobot_root,
            out_dir / f"{TASK_B_STEM}_demo.hdf5",
            _limit(subset["task_b"]["episode_indices"]),
            max_workers=args.max_workers,
        )

    logger.info("done. output dir: %s", out_dir)


if __name__ == "__main__":
    main()
