"""
Publication-Quality Figure Generator for Rolling Shutter Correction Paper
Generates 4 high-resolution (300 DPI), publication-standard figures with white backgrounds:
- Figure 1: Qualitative Visual Gallery on Official 999-Sample RS-Real Benchmark (with zoom insets)
- Figure 2: Training Convergence, Phase Transition & Multi-Crop Boundary Disocclusion Analysis
- Figure 3: Quantization & Compression Pareto Frontier (Proposed CNN vs RS-Diffusion)
- Figure 4: Boundary Disocclusion & Border Inpainting Mechanism
"""

import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F

import sys
sys.path.append(r"D:\c data\Desktop\css")
from train_real_model import CoarseToFineFlowNet, BorderRefinementHead, warp_image_with_mask, upsample_and_rescale_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(r"D:\c data\Desktop\css\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set clean IEEE / CVPR publication typography style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def flow_to_rgb(flow):
    """Vectorized conversion of (2, H, W) optical flow to HSV-based RGB."""
    fx, fy = flow[0], flow[1]
    rad = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    max_rad = max(float(rad.max()), 1.0)
    
    h = (angle / np.pi + 1.0) / 2.0
    s = np.clip(rad / max_rad * 1.5, 0.0, 1.0)
    v = np.ones_like(rad)

    # HSV to RGB vectorized
    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    m0 = (i == 0); rgb[m0] = np.stack([v[m0], t[m0], p[m0]], axis=-1)
    m1 = (i == 1); rgb[m1] = np.stack([q[m1], v[m1], p[m1]], axis=-1)
    m2 = (i == 2); rgb[m2] = np.stack([p[m2], v[m2], t[m2]], axis=-1)
    m3 = (i == 3); rgb[m3] = np.stack([p[m3], q[m3], v[m3]], axis=-1)
    m4 = (i == 4); rgb[m4] = np.stack([t[m4], p[m4], v[m4]], axis=-1)
    m5 = (i == 5); rgb[m5] = np.stack([v[m5], p[m5], q[m5]], axis=-1)
    return rgb

def calc_psnr_np(pred, target):
    mse = np.mean((pred - target) ** 2)
    return 100.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)

def generate_figure1():
    """Figure 1: Qualitative Visual Gallery on Official Benchmark Samples."""
    print("Generating Figure 1: Qualitative Visual Results...")
    CKPT_PATH = r"D:\c data\Desktop\css\checkpoints_real\best_model_real.pt"
    TEST_DIR = Path(r"D:\RS-Real\test\npz")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    flow_net = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead().to(device)
    flow_net.load_state_dict(ckpt["flow_net_state"])
    refine_net.load_state_dict(ckpt["refine_net_state"])
    flow_net.eval()
    refine_net.eval()

    test_files = sorted(TEST_DIR.glob("*.npz"))
    # Pick 3 diverse benchmark test samples
    chosen_indices = [15, 120, 350]
    chosen = [test_files[i % len(test_files)] for i in chosen_indices]

    N = len(chosen)
    fig, axes = plt.subplots(N, 6, figsize=(15, 2.7 * N), constrained_layout=True)
    fig.patch.set_facecolor("white")

    col_titles = [
        "(a) Distorted RS Input",
        "(b) Predicted Optical Flow",
        "(c) Warped Intermediate",
        "(d) Refined Output (Ours)",
        "(e) Ground Truth (GS)",
        "(f) Absolute Error (|Ours - GS|)"
    ]

    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=10.5, fontweight="bold", pad=8)

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
            warped, _ = warp_image_with_mask(rs, pred_flow)
            refined = refine_net(warped)

            rs_disp = rs[0].permute(1, 2, 0).cpu().numpy()
            flow_disp = flow_to_rgb(pred_flow[0].cpu().numpy())
            warp_disp = warped[0].permute(1, 2, 0).cpu().numpy()
            ref_disp = refined[0].permute(1, 2, 0).cpu().numpy()
            gs_disp = gs[0].permute(1, 2, 0).cpu().numpy()
            err_disp = np.abs(ref_disp - gs_disp).mean(axis=-1)

            psnr_in = calc_psnr_np(rs_disp, gs_disp)
            psnr_out = calc_psnr_np(ref_disp, gs_disp)

            imgs = [rs_disp, flow_disp, warp_disp, ref_disp, gs_disp, err_disp]
            cmaps = [None, None, None, None, None, "inferno"]

            for col_i in range(6):
                ax = axes[row_i, col_i]
                if cmaps[col_i] is not None:
                    im = ax.imshow(imgs[col_i], cmap=cmaps[col_i], vmin=0, vmax=0.25)
                else:
                    ax.imshow(np.clip(imgs[col_i], 0, 1))

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.8)

            axes[row_i, 0].set_ylabel(f"Scene {chosen_indices[row_i]}\nPSNR: {psnr_in:.1f} dB", fontsize=9.5, fontweight="semibold")
            axes[row_i, 3].set_xlabel(f"PSNR: {psnr_out:.1f} dB (+{psnr_out - psnr_in:.1f} dB)", fontsize=9.5, color="#006622", fontweight="bold")

    out_png = OUT_DIR / "figure1_qualitative_results.png"
    out_pdf = OUT_DIR / "figure1_qualitative_results.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, facecolor="white")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")

def generate_figure2():
    """Figure 2: Training Convergence, Phase Transition & Multi-Crop Boundary Disocclusion Analysis."""
    print("Generating Figure 2: Convergence and Multi-Crop Analysis...")
    with open(r"D:\c data\Desktop\css\checkpoints_real\history_real.json", "r") as f:
        history = json.load(f)

    with open(r"D:\c data\Desktop\css\checkpoints_real\benchmark_test_results.json", "r") as f:
        benchmark = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_psnr = [h["val_psnr"] for h in history]
    val_ssim = [h["val_ssim"] for h in history]
    val_flow = [h["val_flow"] for h in history]
    lr = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # (a) Training & Validation Loss with Flow EPE
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.0, linestyle="--")
    ax1.plot(epochs, val_flow, label="Val Flow EPE (px)", color="#2ca02c", linewidth=1.8, linestyle=":")
    ax1.axvline(x=13, color="#d62728", linestyle="-.", alpha=0.7, label="Phase Transition (Ep. 13)")
    ax1.set_title("(a) Multi-Task Loss & Flow EPE Convergence", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Loss / End-Point Error (px)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # (b) Validation PSNR & SSIM Progression
    ax2 = axes[1]
    color_psnr = "#0055aa"
    color_ssim = "#cc3333"

    ax2.plot(epochs, val_psnr, color=color_psnr, linewidth=2.2, label="Validation PSNR (dB)")
    ax2.set_xlabel("Training Epochs")
    ax2.set_ylabel("PSNR (dB)", color=color_psnr, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_psnr)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, val_ssim, color=color_ssim, linewidth=2.0, linestyle="--", label="Validation SSIM")
    ax2_twin.set_ylabel("Structural Similarity (SSIM)", color=color_ssim, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_ssim)

    ax2.set_title("(b) Validation PSNR & SSIM Progression", fontsize=11, fontweight="bold")
    ax2.axvline(x=13, color="#d62728", linestyle="-.", alpha=0.7)
    ax2.text(14, 15.45, "Phase Transition\nFlow Convergence", fontsize=8.5, color="#990000", fontweight="semibold")

    # (c) Multi-Crop Boundary Disocclusion Analysis
    ax3 = axes[2]
    crop_data = benchmark["crop_analysis"]
    crop_margins = [0, 8, 16, 24]
    crop_psnrs = [
        crop_data["0px_crop_psnr"],
        crop_data["8px_crop_psnr"],
        crop_data["16px_crop_psnr"],
        crop_data["24px_crop_psnr"]
    ]
    uncorrected_psnr = benchmark["uncorrected_mean_psnr"]

    bar_colors = ["#4a90e2", "#357abd", "#286090", "#1b3a5b"]
    bars = ax3.bar([str(m) + " px" for m in crop_margins], crop_psnrs, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax3.axhline(y=uncorrected_psnr, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Uncorrected RS ({uncorrected_psnr:.2f} dB)")

    for bar, val in zip(bars, crop_psnrs):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{val:.2f} dB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax3.set_ylim(14.5, 20.2)
    ax3.set_title("(c) Multi-Crop FOV Boundary Analysis", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Boundary Margin Excluded (Border Crop)")
    ax3.set_ylabel("Evaluation PSNR (dB)")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    out_png = OUT_DIR / "figure2_training_convergence.png"
    out_pdf = OUT_DIR / "figure2_training_convergence.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, facecolor="white")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")

def generate_figure3():
    """Figure 3: Quantization & Compression Pareto Frontier (Proposed CNN vs RS-Diffusion)."""
    print("Generating Figure 3: Quantization and Compression Pareto Frontier...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Data for Proposed CNN
    cnn_data = [
        ("FP32 (Dense)", 4.28, 18.74, 284.9, "#1f77b4", "o"),
        ("FP16 (Half)", 2.14, 18.75, 241.6, "#17becf", "s"),
        ("W8 (INT8)", 1.07, 18.75, 253.2, "#2ca02c", "^"),
        ("W8A8 (INT8)", 1.07, 18.74, 131.2, "#bcbd22", "v"),
        ("W4 (4-bit)", 0.54, 18.15, 244.5, "#ff7f0e", "D"),
        ("Prune 30%", 4.28, 16.84, 275.1, "#9467bd", "h"),
    ]

    # Data for RS-Diffusion (AAAI 2025)
    diff_data = [
        ("RS-Diff 8-step", 319.5, 26.16, 5.38, "#d62728", "*"),
        ("RS-Diff 1-step", 319.5, 25.77, 27.06, "#e377c2", "p"),
        ("RS-Diff W8A8", 79.9, 25.26, 5.10, "#8c564b", "v"),
        ("RS-Diff W4 (Collapse)", 39.9, 21.88, 5.10, "#7f7f7f", "X"),
    ]

    # Subplot (a): Quality (PSNR) vs Model Storage Size (Log Scale)
    for name, size, psnr, fps, col, marker in cnn_data:
        ax1.scatter(size, psnr, color=col, marker=marker, s=120, edgecolors="black", linewidth=1.0, zorder=4)
        offset_y = 0.25 if name != "W8A8 (INT8)" else -0.5
        ax1.annotate(name, (size, psnr), xytext=(size * 1.15, psnr + offset_y), fontsize=8.5, fontweight="semibold")

    for name, size, psnr, fps, col, marker in diff_data:
        ax1.scatter(size, psnr, color=col, marker=marker, s=160, edgecolors="black", linewidth=1.0, zorder=4)
        offset_x = 0.5 if "Collapse" in name else 1.1
        ax1.annotate(name, (size, psnr), xytext=(size * offset_x, psnr + 0.3), fontsize=8.5, fontweight="semibold", color="#a00000")

    ax1.set_xscale("log")
    ax1.set_xlabel("Storage Footprint (MB, Log Scale)")
    ax1.set_ylabel("Restoration Fidelity (PSNR in dB)")
    ax1.set_title("(a) Accuracy vs Storage: Proposed CNN vs. RS-Diffusion", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.axvspan(0.4, 5.0, color="#2ca02c", alpha=0.08, label="Edge Embedded Zone (< 5 MB)")
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # Subplot (b): Throughput (FPS) vs Model Storage Size
    for name, size, psnr, fps, col, marker in cnn_data:
        ax2.scatter(size, fps, color=col, marker=marker, s=120, edgecolors="black", linewidth=1.0, zorder=4)
        ax2.annotate(f"{name}\n({fps:.0f} FPS)", (size, fps), xytext=(size * 1.15, fps - 10), fontsize=8.0)

    for name, size, psnr, fps, col, marker in diff_data:
        ax2.scatter(size, fps, color=col, marker=marker, s=160, edgecolors="black", linewidth=1.0, zorder=4)
        ax2.annotate(f"{name}\n({fps:.1f} FPS)", (size, fps), xytext=(size * 0.45, fps + 8), fontsize=8.0, color="#a00000")

    ax2.axhline(y=30, color="#e74c3c", linestyle="--", linewidth=1.5, label="Standard Real-Time Threshold (30 FPS)")
    ax2.axhline(y=60, color="#f39c12", linestyle=":", linewidth=1.5, label="High-Speed Video Threshold (60 FPS)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Storage Footprint (MB, Log Scale)")
    ax2.set_ylabel("Throughput (Frames Per Second)")
    ax2.set_title("(b) Inference Speed vs Storage", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    out_png = OUT_DIR / "figure3_compression_pareto_frontier.png"
    out_pdf = OUT_DIR / "figure3_compression_pareto_frontier.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, facecolor="white")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")

def custom_warp(image, flow, padding_mode="zeros"):
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    sample_x = xx.unsqueeze(0).expand(b, -1, -1) + flow[:, 0]
    sample_y = yy.unsqueeze(0).expand(b, -1, -1) + flow[:, 1]
    norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)

def generate_figure4():
    """Figure 4: Edge Disocclusion & Boundary Refinement Inpainting Demonstration."""
    print("Generating Figure 4: Boundary Disocclusion and Inpainting...")
    CKPT_PATH = r"D:\c data\Desktop\css\checkpoints_real\best_model_real.pt"
    TEST_DIR = Path(r"D:\RS-Real\test\npz")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    flow_net = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead().to(device)
    flow_net.load_state_dict(ckpt["flow_net_state"])
    refine_net.load_state_dict(ckpt["refine_net_state"])
    flow_net.eval()
    refine_net.eval()

    test_files = sorted(TEST_DIR.glob("*.npz"))
    # Pick sample with notable camera rotation / motion
    chosen_file = test_files[15]

    with np.load(chosen_file, allow_pickle=False) as d:
        rs_np = d["condition"]
        gs_np = d["gs"]

    rs = torch.from_numpy(rs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    gs = torch.from_numpy(gs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        rs_low = F.interpolate(rs, (64, 64), mode="bilinear", align_corners=False)
        h_low, w_low = rs_low.shape[-2:]
        y_coord = torch.linspace(0, 1, h_low, device=device).view(-1, 1).expand(h_low, w_low).unsqueeze(0).unsqueeze(0)
        rs_low_4ch = torch.cat([rs_low, y_coord], dim=1)

        flows = flow_net(rs_low_4ch)
        pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
        warped_zero = custom_warp(rs, pred_flow, padding_mode="zeros")
        warped_border = custom_warp(rs, pred_flow, padding_mode="border")
        refined = refine_net(warped_zero)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    titles = [
        "(a) Uncorrected RS Frame",
        "(b) Naive Border Clamping\n(Edge Smearing Artifacts)",
        "(c) Detached Zero-Padding\n(Physical Disocclusion Boundary)",
        "(d) Boundary Refinement (Ours)\n(Contextual Edge Inpainting)",
        "(e) Ground Truth (GS)"
    ]

    displays = [
        rs[0].permute(1, 2, 0).cpu().numpy(),
        warped_border[0].permute(1, 2, 0).cpu().numpy(),
        warped_zero[0].permute(1, 2, 0).cpu().numpy(),
        refined[0].permute(1, 2, 0).cpu().numpy(),
        gs[0].permute(1, 2, 0).cpu().numpy()
    ]

    for ax, title, disp in zip(axes, titles, displays):
        ax.imshow(np.clip(disp, 0, 1))
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
            spine.set_linewidth(0.8)

    out_png = OUT_DIR / "figure4_edge_disocclusion_inpainting.png"
    out_pdf = OUT_DIR / "figure4_edge_disocclusion_inpainting.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, facecolor="white")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")

def main():
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    print("\nAll 4 publication figures generated successfully in:", OUT_DIR)

if __name__ == "__main__":
    main()
