"""
Visualize Real-World RS Correction Results on Official Benchmark Samples
"""

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import sys
sys.path.append(r"D:\c data\Desktop\css")
from train_real_model import CoarseToFineFlowNet, BorderRefinementHead, warp_image_with_mask, upsample_and_rescale_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flow_to_rgb(flow):
    """Convert (2, H, W) flow to HSV RGB visualization."""
    fx, fy = flow[0], flow[1]
    rad = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    max_rad = max(rad.max(), 1.0)
    
    h = (angle / np.pi + 1.0) / 2.0
    s = np.clip(rad / max_rad * 1.5, 0, 1)
    v = np.ones_like(rad)

    import colorsys
    rgb = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    for y in range(flow.shape[1]):
        for x in range(flow.shape[2]):
            rgb[y, x] = colorsys.hsv_to_rgb(h[y, x], s[y, x], v[y, x])
    return rgb

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)

def main():
    CKPT_PATH = r"D:\c data\Desktop\css\checkpoints_real\best_model_real.pt"
    TEST_DIR = Path(r"D:\RS-Real\test\npz")
    OUT_PNG = r"D:\c data\Desktop\css\checkpoints_real\real_benchmark_visual_comparison.png"

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    flow_net = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead().to(device)
    flow_net.load_state_dict(ckpt["flow_net_state"])
    refine_net.load_state_dict(ckpt["refine_net_state"])
    flow_net.eval()
    refine_net.eval()

    test_files = sorted(TEST_DIR.glob("*.npz"))
    # Pick 4 diverse real scenes
    indices = [15, 120, 350, 780]
    chosen = [test_files[i % len(test_files)] for i in indices]

    N = len(chosen)
    fig, axes = plt.subplots(N, 6, figsize=(26, 4.5 * N + 1))
    fig.patch.set_facecolor("#121218")

    col_titles = [
        "Rolling Shutter (Input)",
        "Predicted Optical Flow",
        "Warped Intermediate",
        "Refined Output (Ours)",
        "Ground Truth (Global Shutter)",
        "Error Map"
    ]
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)

    with torch.no_grad():
        for row_i, p in enumerate(chosen):
            with np.load(p, allow_pickle=False) as d:
                rs_np = d["condition"]
                gs_np = d["gs"]
                gt_flow_np = d["out_flow"]

            rs = torch.from_numpy(rs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            gs = torch.from_numpy(gs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            rs_low = F.interpolate(rs, (64, 64), mode="bilinear", align_corners=False)
            h_low, w_low = rs_low.shape[-2:]
            y_coord = torch.linspace(0, 1, h_low, device=device).view(-1, 1).expand(h_low, w_low).unsqueeze(0).unsqueeze(0)
            rs_low_4ch = torch.cat([rs_low, y_coord], dim=1)

            flows = flow_net(rs_low_4ch)
            pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
            warped, _ = warp_image_with_mask(rs, pred_flow)
            refined = refine_net(warped)

            p_in = calc_psnr(rs, gs)
            p_ref = calc_psnr(refined, gs)

            rs_disp = rs[0].permute(1, 2, 0).cpu().numpy()
            flow_disp = flow_to_rgb(pred_flow[0].cpu().numpy())
            warp_disp = warped[0].permute(1, 2, 0).cpu().numpy()
            ref_disp = refined[0].permute(1, 2, 0).cpu().numpy()
            gs_disp = gs[0].permute(1, 2, 0).cpu().numpy()
            err_disp = np.abs(ref_disp - gs_disp).mean(axis=-1)

            imgs = [rs_disp, flow_disp, warp_disp, ref_disp, gs_disp, err_disp]
            cmaps = [None, None, None, None, None, "hot"]
            sub_labels = [
                f"{p.stem} (PSNR: {p_in:.1f} dB)",
                f"Peak shift: {float(pred_flow[0].abs().max()):.1f} px",
                "Warped (raw disocclusion)",
                f"PSNR: {p_ref:.1f} dB (Inpainted)",
                "True GS Sensor Frame",
                f"Residual (Mean: {float(err_disp.mean()):.3f})"
            ]

            for col_i in range(6):
                ax = axes[row_i, col_i]
                if cmaps[col_i] is not None:
                    ax.imshow(imgs[col_i], cmap=cmaps[col_i], vmin=0, vmax=0.25)
                else:
                    ax.imshow(np.clip(imgs[col_i], 0, 1))
                ax.set_xlabel(sub_labels[col_i], color="white", fontsize=10, labelpad=6)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333344")

    fig.suptitle(
        "Real-World Rolling Shutter Correction (Trained with Direct Flow Supervision on RS-Real)\n"
        "Benchmark Test Set: 999 Real Dual-Sensor Camera Samples",
        color="white", fontsize=15, fontweight="bold", y=0.99
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved visualization to: {OUT_PNG}")

if __name__ == "__main__":
    main()
