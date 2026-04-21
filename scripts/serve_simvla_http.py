"""FastAPI HTTP server exposing SimVLA (SmolVLM LIBERO finetune) via the unified VLA protocol.

Implements the three-endpoint contract defined in VLA_COMMUNICATION_PROTOCOL.md:

    GET  /health  →  {status, model, action_type, action_keys, n_action_steps}
    POST /reset   →  {status}
    POST /act     →  {action.eef_pos, action.eef_euler, action.gripper, latency_ms}

The server accepts generic observations keyed by `observation.images.{cam}` (base64 PNG)
and `observation.state.{field}` (flat list), plus a `task` string. It then performs the
SimVLA-LIBERO specific preprocessing (180° rotation, bilinear resize to 224x224, ImageNet
norm via SmolVLM processor) so that every benchmark can talk to this server without
knowing anything about SimVLA internals.

State assembly mirrors the native SimVLA client:
    state = [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D

Action output is a 10-step chunk of 7D deltas:
    action[:, 0:3] → action.eef_pos    (delta xyz)
    action[:, 3:6] → action.eef_euler   (delta axis-angle placed in the euler slot;
                                         the LIBERO OSC_POSE controller concatenates
                                         [pos, rot, grip] → 7D directly, so this is
                                         exactly what LIBERO expects)
    action[:, 6:7] → action.gripper

Run:
    python scripts/serve_simvla_http.py --port 8700 --dummy
    python scripts/serve_simvla_http.py --port 8700 \
        --checkpoint /simvla_ckpt \
        --norm-stats /app/norm_stats/libero_norm.json
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

# Make `models.*` importable when the server is launched from /app (SimVLA repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("serve_simvla_http")

# ----------------------------------------------------------------------------- #
# Globals populated at startup
# ----------------------------------------------------------------------------- #
model = None            # SmolVLMVLA
processor = None        # SmolVLMVLAProcessor
device = "cpu"
is_dummy: bool = False
action_horizon: int = 10
action_dim: int = 7
state_dim: int = 8
image_size: int = 384   # internal model input size (after ImageNet norm + resize)
pre_resize: int = 224   # size images are resized to before handing to processor transform

app = FastAPI()


# ----------------------------------------------------------------------------- #
# State utilities
# ----------------------------------------------------------------------------- #
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation.

    Copied verbatim from SimVLA's libero_client._quat2axisangle so the server matches
    the convention used at training time (robosuite).
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _as_np(x, dtype=np.float32) -> np.ndarray:
    """Coerce a list / tuple / scalar / np.ndarray into a flat np.ndarray."""
    if x is None:
        return np.zeros(0, dtype=dtype)
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr


def _assemble_state(payload: dict) -> np.ndarray:
    """Build the 8D state vector [eef_pos(3), axis_angle(3), gripper_qpos(2)].

    Any missing key is zero-padded so the server never crashes on degenerate payloads.
    `observation.state.eef_quat` is expected in [x, y, z, w] order (matches robosuite
    and LIBERO-pro's client).
    """
    eef_pos = _as_np(payload.get("observation.state.eef_pos"))
    if eef_pos.size < 3:
        eef_pos = np.pad(eef_pos, (0, 3 - eef_pos.size))
    eef_pos = eef_pos[:3]

    quat = _as_np(payload.get("observation.state.eef_quat"))
    if quat.size < 4:
        # No rotation info → identity quaternion (w=1) → zero axis-angle.
        axisangle = np.zeros(3, dtype=np.float32)
    else:
        axisangle = _quat2axisangle(quat[:4].astype(np.float64)).astype(np.float32)

    grip = _as_np(payload.get("observation.state.gripper_qpos"))
    if grip.size < 2:
        grip = np.pad(grip, (0, 2 - grip.size))
    grip = grip[:2]

    state = np.concatenate([eef_pos, axisangle, grip]).astype(np.float32)
    if state.size < state_dim:
        state = np.pad(state, (0, state_dim - state.size))
    return state[:state_dim]


# ----------------------------------------------------------------------------- #
# Image utilities
# ----------------------------------------------------------------------------- #
def _b64_to_pil(b64_str: str) -> Image.Image:
    """base64-encoded PNG → PIL RGB image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _resize_with_pad_square(img: Image.Image, size: int) -> Image.Image:
    """Pad to square with black, then bilinear resize to (size, size).

    Replicates openpi_client.image_tools.resize_with_pad without the dependency.
    For inputs that are already square (e.g. 256x256 from LIBERO) the padding is a
    no-op and this reduces to a plain PIL resize.
    """
    w, h = img.size
    if w != h:
        side = max(w, h)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(img, ((side - w) // 2, (side - h) // 2))
        img = padded
    return img.resize((size, size), Image.BILINEAR)


def _prepare_libero_image(b64_str: str) -> np.ndarray:
    """base64 PNG → HxWx3 uint8, 180° rotated and resized to `pre_resize`.

    The client (Libero-pro benchmark's libero_vla_eval.py) sends raw un-rotated images,
    so we apply the 180° rotation here to match what SimVLA saw at training time.
    """
    pil = _b64_to_pil(b64_str)
    arr = np.array(pil)                       # HxWx3 uint8
    arr = np.ascontiguousarray(arr[::-1, ::-1])  # 180° rotate (vertical + horizontal flip)
    out = _resize_with_pad_square(Image.fromarray(arr), pre_resize)
    return np.array(out, dtype=np.uint8)


def _preprocess_images_for_model(image0: np.ndarray, image1: Optional[np.ndarray]):
    """Mirror `serve_smolvlm_libero.preprocess_images`: resize to `image_size`, ImageNet
    normalize, pad to 3 views (third slot zeroed out, mask=False).

    image1 may be None if the client omits the wrist camera — in that case we still emit
    a zeroed third slot and mark both [True, False, False] in the mask.
    """
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img0_t = transform(Image.fromarray(image0.astype(np.uint8)))
    if image1 is not None:
        img1_t = transform(Image.fromarray(image1.astype(np.uint8)))
        mask = [True, True, False]
    else:
        img1_t = torch.zeros_like(img0_t)
        mask = [True, False, False]

    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([mask])
    return images.unsqueeze(0), image_mask


# ----------------------------------------------------------------------------- #
# Model loading
# ----------------------------------------------------------------------------- #
def _load_model(checkpoint: str, norm_stats: Optional[str], smolvlm_model_path: str):
    """Load SimVLA (SmolVLMVLA) + its processor from a checkpoint dir.

    norm_stats are attached via `model.action_space.load_norm_stats` when provided,
    matching serve_smolvlm_libero.load_model.
    """
    import os

    import torch

    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    global model, processor, device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SimVLA checkpoint %s (device=%s)", checkpoint, device)
    m = SmolVLMVLA.from_pretrained(checkpoint)
    m = m.to(device)
    m.eval()

    logger.info("Loading processor from %s", smolvlm_model_path)
    p = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path)

    if norm_stats and os.path.exists(norm_stats):
        logger.info("Loading norm stats from %s", norm_stats)
        m.action_space.load_norm_stats(norm_stats)
    elif norm_stats:
        logger.warning("norm_stats path %s does not exist; skipping", norm_stats)
    else:
        logger.warning("No norm_stats provided; running with identity normalization.")

    model, processor = m, p
    logger.info("SimVLA loaded. image_size=%d  action_horizon=%d", image_size, action_horizon)


# ----------------------------------------------------------------------------- #
# HTTP endpoints
# ----------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {
        "status": "ok" if (model is not None or is_dummy) else "loading",
        "model": "simvla_dummy" if is_dummy else "simvla",
        "action_type": "relative",
        "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
        "n_action_steps": action_horizon,
    }


@app.post("/reset")
async def reset():
    # SimVLA is stateless per-request; action chunking lives on the client side.
    return {"status": "reset"}


@app.post("/act")
async def act(payload: dict):
    t0 = time.time()

    # Pick static/agentview camera. Falls back to the first available camera key.
    b64_static = payload.get("observation.images.static")
    if b64_static is None:
        for k, v in payload.items():
            if k.startswith("observation.images.") and not k.endswith(".wrist"):
                b64_static = v
                break

    b64_wrist = payload.get("observation.images.wrist")
    task = str(payload.get("task", ""))

    if is_dummy:
        actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
    elif b64_static is None:
        # No image → zero chunk. Keeps the server crash-free on degenerate payloads.
        actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
    else:
        try:
            import torch

            img_static = _prepare_libero_image(b64_static)
            img_wrist = _prepare_libero_image(b64_wrist) if b64_wrist is not None else None

            images, image_mask = _preprocess_images_for_model(img_static, img_wrist)
            images = images.to(device)
            image_mask = image_mask.to(device)

            state = _assemble_state(payload)
            proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            lang = processor.encode_language([task])
            lang = {k: v.to(device) for k, v in lang.items()}

            with torch.no_grad():
                out = model.generate_actions(
                    input_ids=lang["input_ids"],
                    image_input=images,
                    image_mask=image_mask,
                    proprio=proprio,
                    steps=action_horizon,
                )
            actions = out.cpu().numpy()[0]  # [H, 7]
            actions = np.asarray(actions, dtype=np.float32)
        except Exception:
            # Mirror serve_smolvlm_libero.infer's except block: log and return zeros
            # so the eval loop doesn't crash mid-episode.
            logger.exception("Inference failed; returning zero action chunk.")
            actions = np.zeros((action_horizon, action_dim), dtype=np.float32)

    # Split into unified sub-keys. Protocol requires 2-D lists, shape [N_steps, D].
    pos = actions[:, 0:3].tolist()
    euler = actions[:, 3:6].tolist()
    grip = actions[:, 6:7].tolist()
    return {
        "action.eef_pos": pos,
        "action.eef_euler": euler,
        "action.gripper": grip,
        "latency_ms": (time.time() - t0) * 1000.0,
    }


# ----------------------------------------------------------------------------- #
# Entry point
# ----------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SimVLA checkpoint directory (e.g. /simvla_ckpt).")
    parser.add_argument("--norm-stats", default="/app/norm_stats/libero_norm.json",
                        help="Path to action/state normalization stats JSON.")
    parser.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="SmolVLM model path or HF repo (used for the processor only).")
    parser.add_argument("--action-horizon", type=int, default=10,
                        help="Number of action steps returned per /act call.")
    parser.add_argument("--image-size", type=int, default=384,
                        help="Model-internal image input size (post-ImageNet-norm resize).")
    parser.add_argument("--dummy", action="store_true",
                        help="Skip model loading; return zero actions. Useful for protocol smoke tests.")
    args = parser.parse_args()

    global is_dummy, action_horizon, image_size
    is_dummy = args.dummy
    action_horizon = args.action_horizon
    image_size = args.image_size

    if args.dummy:
        logger.warning("Running in --dummy mode (zero actions, no checkpoint).")
    else:
        _load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
