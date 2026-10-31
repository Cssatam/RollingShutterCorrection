"""
Generate ~1000 synthetic rolling-shutter paired samples in D:\RS-Real\train
Each sample is saved under its own scene folder so the existing dataset structure
is preserved (scene01/sample_XXXX.npz etc.).
"""

import os
import numpy as np
from pathlib import Path

def make_sample(height=256, width=256, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # ── GS frame: textured pattern ──────────────────────────────────────────
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    fx, fy = rng.uniform(1.5, 8), rng.uniform(1.5, 8)
    r = 0.5 + 0.4 * np.sin(2 * np.pi * xx / width  * fx)
    g = 0.5 + 0.4 * np.cos(2 * np.pi * yy / height * fy)
    b = 0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy) / (width + height) * rng.uniform(2, 6))

    # Add random blobs for texture richness
    for _ in range(rng.integers(3, 8)):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        sig    = rng.uniform(20, 80)
        blob   = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sig**2))
        c_idx  = rng.integers(0, 3)
        if c_idx == 0:   r = np.clip(r + 0.3 * blob, 0, 1)
        elif c_idx == 1: g = np.clip(g + 0.3 * blob, 0, 1)
        else:            b = np.clip(b + 0.3 * blob, 0, 1)

    gs_img   = np.stack([r, g, b], axis=-1)
    gs_uint8 = np.clip(gs_img * 255.0, 0, 255).astype(np.uint8)

    # ── RS distortion: scanline-proportional shift ──────────────────────────
    max_shift = rng.uniform(4.0, 18.0) * rng.choice([-1.0, 1.0])
    flow_x = (yy / height) * max_shift
    # Add gentle secondary wobble
    flow_y = rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * yy / height) * max_shift

    sample_x = np.clip(xx - flow_x, 0, width  - 1).astype(np.int32)
    sample_y = np.clip(yy - flow_y, 0, height - 1).astype(np.int32)
    rs_uint8 = gs_uint8[sample_y, sample_x]

    flow = np.stack([flow_x, flow_y], axis=-1)  # (H, W, 2)
    return rs_uint8, gs_uint8, flow


def main():
    OUT_ROOT   = Path(r"D:\c data\Desktop\css\data\RS-Real\train")
    N_TOTAL    = 1000
    H, W       = 256, 256
    SCENE_SIZE = 10        # samples per scene folder

    rng = np.random.default_rng(2024)

    # Count existing samples
    existing = list(OUT_ROOT.rglob("*.npz"))
    n_existing = len(existing)
    n_needed   = max(0, N_TOTAL - n_existing)
    print(f"Existing samples : {n_existing}")
    print(f"Samples to add   : {n_needed}")

    if n_needed == 0:
        print("Already have >= 1000 samples, nothing to generate.")
        return

    # Find next available scene + sample index
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    existing_scenes = sorted([d.name for d in OUT_ROOT.iterdir() if d.is_dir()])
    scene_idx_start = len(existing_scenes) + 1

    generated = 0
    scene_idx = scene_idx_start
    while generated < n_needed:
        scene_dir = OUT_ROOT / f"scene{scene_idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for s in range(SCENE_SIZE):
            if generated >= n_needed:
                break
            rs, gs, flow = make_sample(H, W, rng)
            out_path = scene_dir / f"sample_{s:04d}.npz"
            np.savez_compressed(str(out_path), condition=rs, gs=gs, out_flow=flow)
            generated += 1
        print(f"  Scene {scene_idx:02d} done  ({generated}/{n_needed} new samples)", flush=True)
        scene_idx += 1

    total = len(list(OUT_ROOT.rglob("*.npz")))
    print(f"\nDone! Total samples in dataset: {total}")


if __name__ == "__main__":
    main()
