"""
Quick Demonstration Script for Rolling Shutter Correction
Usage:
    python demo.py
    python demo.py --image path/to/distorted_rs_image.jpg
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from train_real_model import (
    CoarseToFineFlowNet,
    BorderRefinementHead,
    warp_image_with_mask,
    upsample_and_rescale_flow
)

def run_demo(image_path=None, ckpt_path="checkpoints_real/best_model_real.pt", output_path="demo_result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Load trained model
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Checking fallback...")
        if os.path.exists("checkpoints/best_model.pt"):
            ckpt_path = "checkpoints/best_model.pt"
        else:
            raise FileNotFoundError(f"No trained checkpoint found. Expected {ckpt_path}")

    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    flow_net = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead().to(device)
    flow_net.load_state_dict(ckpt["flow_net_state"])
    refine_net.load_state_dict(ckpt["refine_net_state"])
    flow_net.eval()
    refine_net.eval()

    # 2. Prepare input image
    if image_path and os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        rs = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        rs = F.interpolate(rs, (256, 256), mode="bilinear", align_corners=False)
    else:
        # Use synthetic test pattern for self-contained instant demo
        print("No image provided. Generating synthetic scanline-distorted sample for demo...")
        yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
        r = 0.5 + 0.4 * np.sin(2 * np.pi * xx / 256.0 * 3)
        g = 0.5 + 0.4 * np.cos(2 * np.pi * yy / 256.0 * 3)
        b = 0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy) / 512.0 * 4)
        gs_np = np.stack([r, g, b], axis=-1)
        shift = (yy / 256.0) * 12.0
        sample_x = np.clip(xx - shift, 0, 255).astype(np.int32)
        sample_y = yy.astype(np.int32)
        rs_np = gs_np[sample_y, sample_x]
        rs = torch.from_numpy(rs_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 3. Multi-channel input with scanline coordinate proxy
    rs_low = F.interpolate(rs, (64, 64), mode="bilinear", align_corners=False)
    h_low, w_low = rs_low.shape[-2:]
    y_coord = torch.linspace(0, 1, h_low, device=device).view(-1, 1).expand(h_low, w_low).unsqueeze(0).unsqueeze(0)
    rs_low_4ch = torch.cat([rs_low, y_coord], dim=1)

    # 4. Inference
    with torch.no_grad():
        flows = flow_net(rs_low_4ch)
        pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
        warped, _ = warp_image_with_mask(rs, pred_flow)
        refined = refine_net(warped)

    # 5. Save visual result
    rs_disp = rs[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    refined_disp = refined[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rs_disp)
    axes[0].set_title("Input Rolling Shutter (Distorted)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(refined_disp)
    axes[1].set_title("Corrected Output (Ours)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Success! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to input RS image")
    parser.add_argument("--ckpt", type=str, default="checkpoints_real/best_model_real.pt")
    parser.add_argument("--output", type=str, default="demo_result.png")
    args = parser.parse_args()
    run_demo(args.image, args.ckpt, args.output)
