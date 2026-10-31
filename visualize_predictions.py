import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from train_model import CoarseToFineFlowNet, RollingShutterDataset, upsample_and_rescale_flow, warp_image, calculate_psnr, calculate_ssim

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    data_dir = r"D:\RS-Real\train"
    
    dataset = RollingShutterDataset(data_dir, flow_size=64, image_size=256)
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    
    model = CoarseToFineFlowNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    sample_indices = [0, 5, 12]

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            rs_low, rs_full, gs_full = dataset[idx]
            rs_low = rs_low.unsqueeze(0).to(device)
            rs_full = rs_full.unsqueeze(0).to(device)
            gs_full = gs_full.unsqueeze(0).to(device)

            flow_low = model(rs_low)
            flow_full = upsample_and_rescale_flow(flow_low, rs_full.shape[-2:])
            corrected = warp_image(rs_full, flow_full)

            psnr = calculate_psnr(corrected, gs_full)
            ssim = calculate_ssim(corrected, gs_full)

            rs_disp = rs_full[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            corr_disp = corrected[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            gs_disp = gs_full[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

            # Flow magnitude visualization
            flow_mag = torch.norm(flow_full[0], dim=0).cpu().numpy()

            axes[row, 0].imshow(rs_disp)
            axes[row, 0].set_title(f"Input RS (Distorted)")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(flow_mag, cmap="plasma")
            axes[row, 1].set_title("Predicted Flow Magnitude")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(corr_disp)
            axes[row, 2].set_title(f"Corrected (PSNR: {psnr:.2f}dB)")
            axes[row, 2].axis("off")

            axes[row, 3].imshow(gs_disp)
            axes[row, 3].set_title("Target GS (Ground Truth)")
            axes[row, 3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "visual_comparison.png")
    plt.savefig(out_path, dpi=150)
    print("Visual comparison plot saved to:", out_path)

if __name__ == "__main__":
    visualize()
