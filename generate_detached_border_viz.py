"""
Generate side-by-side comparison illustrating detached black borders vs stretched borders.
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
from train_real_model import CoarseToFineFlowNet, upsample_and_rescale_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)

def main():
    CKPT_PATH = r"D:\c data\Desktop\css\checkpoints_real\best_model_real.pt"
    TEST_DIR = Path(r"D:\RS-Real\test\npz")
    OUT_PNG = r"D:\c data\Desktop\css\checkpoints_real\detached_black_border_comparison.png"

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    flow_net = CoarseToFineFlowNet().to(device)
    flow_net.load_state_dict(ckpt["flow_net_state"])
    flow_net.eval()

    test_files = sorted(TEST_DIR.glob("*.npz"))
    # Include scene03_0589 (sample 120) from user's image + other scenes with clear border shifts
    indices = [120, 15, 350, 780]
    chosen = [test_files[i % len(test_files)] for i in indices]

    N = len(chosen)
    fig, axes = plt.subplots(N, 4, figsize=(20, 4.8 * N))
    fig.patch.set_facecolor("#111116")

    col_titles = [
        "Rolling Shutter (Input)",
        "Stretched / Slid Border (padding_mode='border')",
        "Detached Black Border (padding_mode='zeros') [NEW]",
        "True GS Sensor Frame (Target)"
    ]
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)

    with torch.no_grad():
        for row_i, p in enumerate(chosen):
            with np.load(p, allow_pickle=False) as d:
                rs_np = d["condition"]
                gs_np = d["gs"]

            rs = torch.from_numpy(rs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            gs = torch.from_numpy(gs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            rs_low = F.interpolate(rs, (64, 64), mode="bilinear", align_corners=False)
            h_low, w_low = rs_low.shape[-2:]
            y_coord = torch.linspace(0, 1, h_low, device=device).view(-1, 1).expand(h_low, w_low).unsqueeze(0).unsqueeze(0)
            rs_low_4ch = torch.cat([rs_low, y_coord], dim=1)

            flows = flow_net(rs_low_4ch)
            pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))

            b, _, h, w = rs.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            sample_x = xx.unsqueeze(0).expand(b, -1, -1) + pred_flow[:, 0]
            sample_y = yy.unsqueeze(0).expand(b, -1, -1) + pred_flow[:, 1]
            norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
            norm_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1)

            # 1. Border stretching
            w_border = F.grid_sample(rs, grid, mode="bilinear", padding_mode="border", align_corners=True)
            # 2. Detached black border (zeros)
            w_zeros = F.grid_sample(rs, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

            p_in = calc_psnr(rs, gs)
            p_border = calc_psnr(w_border, gs)
            p_zeros = calc_psnr(w_zeros, gs)

            imgs = [
                rs[0].permute(1, 2, 0).cpu().numpy(),
                w_border[0].permute(1, 2, 0).cpu().numpy(),
                w_zeros[0].permute(1, 2, 0).cpu().numpy(),
                gs[0].permute(1, 2, 0).cpu().numpy()
            ]

            labels = [
                f"{p.stem}\nPSNR: {p_in:.1f} dB",
                f"Border Stretching\nPSNR: {p_border:.2f} dB",
                f"Detached Black\nPSNR: {p_zeros:.2f} dB (+{p_zeros - p_border:.2f} dB)",
                "True GS Sensor Frame\n(Note black boundary)"
            ]

            for col_i in range(4):
                ax = axes[row_i, col_i]
                ax.imshow(np.clip(imgs[col_i], 0, 1))
                ax.set_xlabel(labels[col_i], color="white", fontsize=10, labelpad=8)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#444455")

    fig.suptitle(
        "Rolling Shutter Detached Black Border vs Edge Stretching\n"
        "Demonstrating clean zero-padded disocclusion matching the True GS sensor frame",
        color="white", fontsize=15, fontweight="bold", y=0.99
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
