"""Evaluate SimVLA on LIBERO-Plus Task A + Task B ONLY (no perturbation variants).

Directly instantiates `OffScreenRenderEnv` from the base BDDL paths for the two
target tasks, skipping LIBERO-Plus's perturbation _add_N variants. Talks to a
SimVLA HTTP server (default port 8700) that already implements the unified VLA
protocol (scripts/serve_simvla_http.py). Reuses Libero-pro_benchmark's
vla_client.VLAClient via sys.path injection — no code duplication.

Must run inside the `libero` conda env of the bigenlight/simvla-train container
(robosuite 1.4.0, LIBERO-Plus installed). The simvla HTTP server must be
running (probably in a second container or process).

Outputs:
    <output_dir>/<timestamp>/summary.json
    <output_dir>/<timestamp>/videos/*.mp4
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import imageio
import numpy as np

logger = logging.getLogger("eval_taskAB")


TASK_A_NAME = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
TASK_B_NAME = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
)
TASK_A_LANG = "turn on the stove and put the moka pot on it"
TASK_B_LANG = "put the black bowl in the bottom drawer of the cabinet and close it"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # open gripper, no arm motion


@dataclass
class TrialResult:
    task: str
    trial: int
    done: bool
    steps: int
    latencies_ms: List[float] = field(default_factory=list)
    video_path: Optional[str] = None


def _load_vla_client(vla_client_path: str):
    """Reuse `Libero-pro_benchmark/scripts/vla_client.py` — same HTTP client as
    the LIBERO-pro benchmark, keeping protocol parity.
    """
    parent = str(Path(vla_client_path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from vla_client import VLAClient  # type: ignore

    return VLAClient


def _build_states(obs: dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    def put(k: str, v):
        if v is not None:
            out[f"observation.state.{k}"] = np.asarray(v, dtype=np.float32).reshape(-1)

    put("eef_pos", obs.get("robot0_eef_pos"))
    put("eef_quat", obs.get("robot0_eef_quat"))  # xyzw
    put("gripper_qpos", obs.get("robot0_gripper_qpos"))
    put("gripper_qvel", obs.get("robot0_gripper_qvel"))
    put("joint_pos", obs.get("robot0_joint_pos"))
    put("joint_vel", obs.get("robot0_joint_vel"))
    return out


def _build_images(obs: dict) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    if "agentview_image" in obs:
        images["static"] = np.ascontiguousarray(obs["agentview_image"])
    if "robot0_eye_in_hand_image" in obs:
        images["wrist"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    return images


def _assemble_action(action_dict: Dict[str, np.ndarray]) -> np.ndarray:
    pos = action_dict["action.eef_pos"]
    rot = action_dict.get("action.eef_euler")
    if rot is None:
        rot = action_dict.get("action.eef_axis_angle")
    if rot is None:
        rot = np.zeros((pos.shape[0], 3), dtype=np.float32)
    grip = action_dict.get("action.gripper")
    if grip is None:
        grip = np.full((pos.shape[0], 1), -1.0, dtype=np.float32)
    n = pos.shape[0]
    assert rot.shape[0] == n and grip.shape[0] == n, "chunk length mismatch"
    return np.concatenate([pos, rot, grip], axis=1).astype(np.float32)


def _rollout(env, client, task_lang: str, max_steps: int, replan_steps: int,
             num_steps_wait: int, save_video: bool) -> TrialResult:
    obs = env.reset()
    client.reset()

    replay: List[np.ndarray] = []
    latencies: List[float] = []
    action_plan: Deque[np.ndarray] = collections.deque()

    # Warm-up no-ops (for objects to settle, mirrors libero_vla_eval.py)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    done = False
    steps = 0
    while steps < max_steps:
        if not action_plan:
            images = _build_images(obs)
            states = _build_states(obs)
            action_or_dict, lat = client.predict(images, states, task_lang)
            latencies.append(lat)
            chunk = _assemble_action(action_or_dict) if isinstance(action_or_dict, dict) \
                else np.atleast_2d(np.asarray(action_or_dict, dtype=np.float32))
            for i in range(min(len(chunk), max(1, replan_steps))):
                action_plan.append(chunk[i])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if save_video:
            replay.append(obs["agentview_image"][::-1])
        steps += 1
        if done:
            break

    return TrialResult(
        task="", trial=-1, done=bool(done), steps=steps, latencies_ms=latencies,
    ), replay


def _resolve_bddl(libero_plus_root: Path, task_name: str) -> Path:
    """Return the base (unperturbed) BDDL path for a task name."""
    cands = [
        libero_plus_root / "libero/libero/bddl_files/libero_10" / f"{task_name}.bddl",
    ]
    for c in cands:
        if c.exists():
            return c
    # Fallback: search
    matches = list(
        (libero_plus_root / "libero/libero/bddl_files").rglob(f"{task_name}.bddl")
    )
    if not matches:
        raise FileNotFoundError(f"BDDL not found for {task_name}")
    # Prefer an exact non-perturbed match
    exact = [m for m in matches if m.stem == task_name]
    return (exact or matches)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vla-url", default=os.environ.get("VLA_SERVER_URL", "http://localhost:8700"))
    p.add_argument("--libero-plus-root", default="/libero_plus")
    p.add_argument(
        "--vla-client-path",
        default="/libero_pro_benchmark/scripts/vla_client.py",
        help="Absolute path to Libero-pro_benchmark/scripts/vla_client.py (bind-mount at runtime).",
    )
    p.add_argument("--output-dir", default="/app/runs/eval_taskAB")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--tasks", default="A,B", help="Comma list; subset of {A,B}.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    libero_plus_root = Path(args.libero_plus_root)
    VLAClient = _load_vla_client(args.vla_client_path)
    client = VLAClient(args.vla_url, timeout=120.0)
    logger.info("Waiting for VLA server at %s ...", args.vla_url)
    server_info = client.wait_until_ready(max_wait=300.0, poll_interval=3.0)
    logger.info("Server ready: %s", server_info)

    from libero.libero.envs import OffScreenRenderEnv  # lazy

    tasks_to_run = []
    if "A" in args.tasks.upper():
        tasks_to_run.append(("A", TASK_A_NAME, TASK_A_LANG))
    if "B" in args.tasks.upper():
        tasks_to_run.append(("B", TASK_B_NAME, TASK_B_LANG))

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"eval_taskAB_{run_tag}"
    video_dir = run_dir / "videos"
    (video_dir if not args.no_video else run_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    results_list: List[dict] = []
    total_eps = 0
    total_succ = 0

    for key, name, lang in tasks_to_run:
        bddl = _resolve_bddl(libero_plus_root, name)
        logger.info("Task %s: %s\n    BDDL: %s", key, lang, bddl)
        for trial in range(args.num_trials):
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=args.resolution,
                camera_widths=args.resolution,
            )
            try:
                env.seed(args.seed + trial)
                tr, replay = _rollout(
                    env, client, lang,
                    max_steps=args.max_steps,
                    replan_steps=args.replan_steps,
                    num_steps_wait=args.num_steps_wait,
                    save_video=not args.no_video,
                )
                tr.task = key
                tr.trial = trial
                if not args.no_video and replay:
                    out_mp4 = (
                        video_dir
                        / f"task{key}_t{trial}_{'success' if tr.done else 'failure'}.mp4"
                    )
                    imageio.mimwrite(str(out_mp4), replay, fps=10)
                    tr.video_path = str(out_mp4)
                logger.info(
                    "  trial %d: done=%s steps=%d lat_avg=%.1fms",
                    trial, tr.done, tr.steps,
                    float(np.mean(tr.latencies_ms)) if tr.latencies_ms else 0.0,
                )
                results_list.append(asdict(tr))
                total_eps += 1
                total_succ += int(tr.done)
            finally:
                env.close()

    summary = {
        "vla_url": args.vla_url,
        "server_info": server_info,
        "num_trials_per_task": args.num_trials,
        "tasks_run": [k for k, _, _ in tasks_to_run],
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "success_rate": total_succ / max(1, total_eps),
        "trials": results_list,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("=" * 60)
    logger.info(
        "DONE  success=%d/%d (%.1f%%)  out=%s",
        total_succ, total_eps, 100.0 * summary["success_rate"], run_dir,
    )


if __name__ == "__main__":
    main()
