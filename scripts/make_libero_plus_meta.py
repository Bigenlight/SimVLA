"""Generate the SimVLA meta JSON for LIBERO-Plus Task A + Task B.

Writes `/app/datasets/metas/libero_plus_taskAB.json` that points at the HDF5
files produced by `lerobot_to_libero_hdf5.py`. Also registers the new subset
names via `DATA_WEIGHTS` patch file (see bottom).

Run inside the simvla conda env:
    python /app/scripts/make_libero_plus_meta.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HDF5_DIR_DEFAULT = "/libero_plus/data/libero_plus_hdf5"
OUT_JSON_DEFAULT = "/app/datasets/metas/libero_plus_taskAB.json"

TASK_A_STEM = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
TASK_B_STEM = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
)
TASK_A_STR = "turn on the stove and put the moka pot on it"
TASK_B_STR = "put the black bowl in the bottom drawer of the cabinet and close it"


def count_demos(h5_path: Path) -> int:
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        return len(list(f["data"].keys()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-dir", default=HDF5_DIR_DEFAULT)
    p.add_argument("--output", default=OUT_JSON_DEFAULT)
    args = p.parse_args()

    hdf5_dir = Path(args.hdf5_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("libero_plus_taskA", TASK_A_STR, hdf5_dir / f"{TASK_A_STEM}_demo.hdf5"),
        ("libero_plus_taskB", TASK_B_STR, hdf5_dir / f"{TASK_B_STEM}_demo.hdf5"),
    ]
    datalist = []
    total = 0
    subsets = []
    subset_stats: dict = {}
    for subset, task_str, h5 in pairs:
        assert h5.exists(), f"HDF5 missing: {h5}"
        n = count_demos(h5)
        datalist.append(
            {"path": str(h5), "task": task_str, "subset": subset, "num_demos": n}
        )
        subsets.append(subset)
        subset_stats[subset] = {"num_files": 1, "num_episodes": n}
        total += n

    meta = {
        "dataset_name": "libero_hdf5",
        "data_dir": str(hdf5_dir),
        "datalist": datalist,
        "num_files": len(datalist),
        "num_episodes": total,
        "subsets": subsets,
        "subset_stats": subset_stats,
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,
        "action_dim": 7,
        "fps": 10,
    }
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path} — {total} demos across {len(datalist)} task(s)")

    # Also dump the DATA_WEIGHTS patch (applied by training launcher, see comment).
    print(
        "\nNOTE: `datasets/domain_config.py` DATA_WEIGHTS dict needs these keys:\n"
        f"    'libero_plus_taskA': 1.0,\n"
        f"    'libero_plus_taskB': 1.0,\n"
        "otherwise SmolVLMDataReader.__iter__ will raise KeyError."
    )


if __name__ == "__main__":
    main()
