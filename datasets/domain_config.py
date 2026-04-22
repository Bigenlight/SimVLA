from __future__ import annotations

DATA_WEIGHTS = {
    # LIBERO datasets
    "libero_hdf5": 1.0,
    "libero_10": 1.0,
    "libero_90": 1.0,
    "libero_goal": 1.0,
    "libero_object": 1.0,
    "libero_spatial": 1.0,
    # LIBERO-Plus task-specific subsets (this fork)
    "libero_plus_taskA": 1.0,
    "libero_plus_taskB": 1.0,
}

DATA_DOMAIN_ID = {
    # LIBERO
    "libero": 0,
    "libero_hdf5": 0,
    "libero_10": 0,
    "libero_90": 0,
    "libero_goal": 0,
    "libero_object": 0,
    "libero_spatial": 0,
    # LIBERO-Plus share the same domain id as LIBERO (same obs/action space)
    "libero_plus_taskA": 0,
    "libero_plus_taskB": 0,
}
