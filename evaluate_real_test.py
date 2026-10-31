"""
Evaluate the Real-World Trained Model on the Official 999 Real Test Samples
============================================================================
Computes:
1. Full-image PSNR and SSIM across all 999 real test samples
2. Multi-crop border analysis (0px, 8px, 16px, 24px)
3. End-Point Error (EPE) against ground-truth physical optical flow (out_flow)
"""

import math
import json
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(r"D:\c data\Desktop\css")
from train_real_model import CoarseToFineFlowNet, BorderRefinementHead, warp_image_with_mask, upsample_and_rescale_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)

def calc_ssim(pred, target):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(pred, 3, 1, 1)
    mu2 = F.avg_pool2d(target, 3, 1, 1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1mu2 = mu1 * mu2
    s1 = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
    s2 = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
    s12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1mu2
    ssim_map = ((2 * mu1mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()

def evaluate():
    CKPT_PATH = r"D:\c data\Desktop\css\checkpoints_real\best_model_real.pt"
    TEST_DIR = r"D:\RS-Real\test\npz"

    print("=" * 65)
    print("  Official Benchmark Evaluation on 999 Real-World Test Samples")
    print("=" * 65)
    print(f"Loading checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    flow_net = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead().to(device)
    flow_net.load_state_dict(ckpt["flow_net_state"])
    refine_net.load_state_dict(ckpt["refine_net_state"])
    flow_net.eval()
    refine_net.eval()

    test_files = sorted(Path(TEST_DIR).glob("*.npz"))
    print(f"Found {len(test_files)} real benchmark test samples.")

    uncorrected_psnr = []
    uncorrected_ssim = []
    warped_psnr = []
    warped_ssim = []
    refined_psnr = []
    refined_ssim = []
    flow_epe_list = []

    # Crop analysis for refined output
    crop_psnr = {0: [], 8: [], 16: [], 24: []}

    t0 = time.time()
    with torch.no_grad():
        for p in tqdm(test_files, desc="Evaluating Test Benchmark"):
            with np.load(p, allow_pickle=False) as d:
                rs_np = d["condition"]
                gs_np = d["gs"]
                gt_flow_np = d["out_flow"]

            rs = torch.from_numpy(rs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            gs = torch.from_numpy(gs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            gt_flow = torch.from_numpy(gt_flow_np.copy()).float().unsqueeze(0).to(device)

            if rs.shape[-1] != 256 or rs.shape[-2] != 256:
                rs = F.interpolate(rs, (256, 256), mode="bilinear", align_corners=False)
                gs = F.interpolate(gs, (256, 256), mode="bilinear", align_corners=False)
                gt_flow = F.interpolate(gt_flow, (256, 256), mode="bilinear", align_corners=False)

            # Low-res input with Y-coordinate channel
            rs_low = F.interpolate(rs, (64, 64), mode="bilinear", align_corners=False)
            h_low, w_low = rs_low.shape[-2:]
            y_coord = torch.linspace(0, 1, h_low, device=device).view(-1, 1).expand(h_low, w_low).unsqueeze(0).unsqueeze(0)
            rs_low_4ch = torch.cat([rs_low, y_coord], dim=1)

            # Forward pass
            flows = flow_net(rs_low_4ch)
            pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
            warped, _ = warp_image_with_mask(rs, pred_flow)
            refined = refine_net(warped)

            # Uncorrected baseline
            uncorrected_psnr.append(calc_psnr(rs, gs))
            uncorrected_ssim.append(calc_ssim(rs, gs))

            # Warped only
            warped_psnr.append(calc_psnr(warped, gs))
            warped_ssim.append(calc_ssim(warped, gs))

            # Refined output
            refined_psnr.append(calc_psnr(refined, gs))
            refined_ssim.append(calc_ssim(refined, gs))

            # Flow EPE
            epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1)).mean().item()
            flow_epe_list.append(epe)

            # Multi-crop PSNR
            for crop in [0, 8, 16, 24]:
                sl = slice(crop, -crop) if crop > 0 else slice(None)
                crop_psnr[crop].append(calc_psnr(refined[:, :, sl, sl], gs[:, :, sl, sl]))

    results = {
        "num_test_samples": len(test_files),
        "uncorrected_mean_psnr": float(np.mean(uncorrected_psnr)),
        "uncorrected_mean_ssim": float(np.mean(uncorrected_ssim)),
        "warped_mean_psnr": float(np.mean(warped_psnr)),
        "warped_mean_ssim": float(np.mean(warped_ssim)),
        "refined_mean_psnr": float(np.mean(refined_psnr)),
        "refined_mean_ssim": float(np.mean(refined_ssim)),
        "mean_flow_epe_px": float(np.mean(flow_epe_list)),
        "crop_analysis": {
            "0px_crop_psnr": float(np.mean(crop_psnr[0])),
            "8px_crop_psnr": float(np.mean(crop_psnr[8])),
            "16px_crop_psnr": float(np.mean(crop_psnr[16])),
            "24px_crop_psnr": float(np.mean(crop_psnr[24])),
        },
        "evaluation_time_sec": float(time.time() - t0),
    }

    print("\n" + "=" * 65)
    print("  Official Test Benchmark Results (999 Real Camera Samples)")
    print("=" * 65)
    print(f"  Uncorrected RS Input : PSNR = {results['uncorrected_mean_psnr']:.2f} dB | SSIM = {results['uncorrected_mean_ssim']:.4f}")
    print(f"  Warped Only          : PSNR = {results['warped_mean_psnr']:.2f} dB | SSIM = {results['warped_mean_ssim']:.4f}")
    print(f"  Refined (Final Model): PSNR = {results['refined_mean_psnr']:.2f} dB | SSIM = {results['refined_mean_ssim']:.4f}")
    print(f"  Mean Optical Flow EPE: {results['mean_flow_epe_px']:.2f} pixels")
    print("-" * 65)
    print("  Interior Border Analysis (Eliminating CMOS Edge Cutoff):")
    print(f"    0 px crop (Full Frame): {results['crop_analysis']['0px_crop_psnr']:.2f} dB")
    print(f"    8 px crop             : {results['crop_analysis']['8px_crop_psnr']:.2f} dB")
    print(f"   16 px crop             : {results['crop_analysis']['16px_crop_psnr']:.2f} dB")
    print(f"   24 px crop             : {results['crop_analysis']['24px_crop_psnr']:.2f} dB")
    print("=" * 65)

    save_path = Path(r"D:\c data\Desktop\css\checkpoints_real\benchmark_test_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full benchmark results to: {save_path}")

if __name__ == "__main__":
    evaluate()
