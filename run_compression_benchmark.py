"""
Benchmark of Speed, Size, and Quality Trade-offs for the Lightweight CNN
Rolling Shutter Correction Model (CoarseToFineFlowNet + BorderRefinementHead)
Replicating the exact compression methodology of the RS-Diffusion benchmark:
- Baseline fp32
- Fewer steps (N/A for single-pass CNN -> instant 1-pass)
- fp16 (autocast & half)
- W8 (per-channel symmetric int8 weights)
- W8A8 (W8 + dynamic per-tensor int8 activation fake-quant)
- W4 (per-channel 4-bit weights)
- Pruning (30%, 50%, 70% unstructured magnitude pruning)
- Prune (30%, 50%) + W8
"""

import os
import sys
import copy
import math
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(r"D:\c data\Desktop\css")
from train_real_model import (
    CoarseToFineFlowNet,
    BorderRefinementHead,
    warp_image_with_mask,
    upsample_and_rescale_flow
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = r"D:\c data\Desktop\css\checkpoints_real\best_model_real.pt"
TEST_DIR = r"D:\RS-Real\test\npz"
N_SAMPLES = 50

# ----------------- Metrics -----------------
def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)

def calc_ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(pred, 3, 1, 1)
    mu2 = F.avg_pool2d(target, 3, 1, 1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1mu2 = mu1 * mu2
    s1 = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
    s2 = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
    s12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1mu2
    ssim_map = ((2 * mu1mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()

def calc_flow_epe(pred_flow, gt_flow):
    epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1)).mean().item()
    return epe

# ----------------- Quantization Functions -----------------
def quantize_weight_per_channel_sym(w, bits=8):
    # w: (C_out, C_in, k, k) or (C_out, C_in)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    orig_shape = w.shape
    c_out = orig_shape[0]
    w_flat = w.reshape(c_out, -1)
    
    max_val = torch.max(torch.abs(w_flat), dim=1, keepdim=True)[0]
    scale = torch.clamp(max_val / qmax, min=1e-8)
    
    q_w = torch.clamp(torch.round(w_flat / scale), qmin, qmax)
    deq_w = (q_w * scale).reshape(orig_shape)
    return deq_w

def apply_weight_quant(model, bits=8):
    model_q = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in model_q.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(quantize_weight_per_channel_sym(module.weight.data, bits=bits))
    return model_q

class QuantizedActConv2d(nn.Module):
    def __init__(self, conv_layer, bits=8):
        super().__init__()
        self.conv = conv_layer
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))

    def forward(self, x):
        # Dynamic per-tensor activation fake quantization
        scale = torch.clamp(torch.max(torch.abs(x)) / self.qmax, min=1e-8)
        x_q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax) * scale
        return self.conv(x_q)

def apply_w8a8(model):
    model_w8 = apply_weight_quant(model, bits=8)
    def replace_convs(m):
        for name, child in m.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(m, name, QuantizedActConv2d(child, bits=8))
            else:
                replace_convs(child)
    replace_convs(model_w8)
    return model_w8

# ----------------- Pruning Functions -----------------
def apply_magnitude_pruning(model, sparsity=0.3):
    model_p = copy.deepcopy(model)
    total_weights = 0
    total_zeros = 0
    with torch.no_grad():
        # Collect all conv weights
        for name, module in model_p.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Skip first and last conv if desired, or prune all
                w = module.weight.data
                k = int(w.numel() * sparsity)
                if k > 0:
                    thresh = torch.kthvalue(torch.abs(w).view(-1), k)[0]
                    mask = (torch.abs(w) > thresh).float()
                    module.weight.data.mul_(mask)
                    total_zeros += (mask == 0).sum().item()
                    total_weights += mask.numel()
    achieved_sparsity = total_zeros / max(total_weights, 1)
    return model_p, achieved_sparsity

def apply_prune_and_quant(model, sparsity=0.3, bits=8):
    model_p, achieved_sp = apply_magnitude_pruning(model, sparsity=sparsity)
    # Quantize only non-zero weights
    with torch.no_grad():
        for name, module in model_p.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data
                mask = (w != 0).float()
                deq_w = quantize_weight_per_channel_sym(w, bits=bits)
                module.weight.data.copy_(deq_w * mask)
    return model_p, achieved_sp

# ----------------- Main Evaluation Pipeline -----------------
def evaluate_variant(flow_net, refine_net, test_data, is_fp16=False, is_half=False):
    flow_net.eval()
    refine_net.eval()
    if is_half:
        flow_net = flow_net.half()
        refine_net = refine_net.half()
    
    psnr_list = []
    ssim_list = []
    epe_list = []
    latencies = []

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            dummy = test_data[0]
            rs_4ch = dummy["rs_low_4ch"].half() if is_half else dummy["rs_low_4ch"]
            rs = dummy["rs"].half() if is_half else dummy["rs"]
            flows = flow_net(rs_4ch)
            pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
            warped, _ = warp_image_with_mask(rs, pred_flow)
            _ = refine_net(warped)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    with torch.no_grad():
        for item in test_data:
            rs_4ch = item["rs_low_4ch"]
            rs = item["rs"]
            gs = item["gs"]
            gt_flow = item["gt_flow"]

            if is_half:
                rs_4ch = rs_4ch.half()
                rs = rs.half()

            t0 = time.perf_counter()
            if is_fp16:
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                    flows = flow_net(rs_4ch)
                    pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
                    warped, _ = warp_image_with_mask(rs, pred_flow)
                    refined = refine_net(warped)
            else:
                flows = flow_net(rs_4ch)
                pred_flow = upsample_and_rescale_flow(flows[-1], (256, 256))
                warped, _ = warp_image_with_mask(rs, pred_flow)
                refined = refine_net(warped)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            refined_f = refined.float()
            pred_flow_f = pred_flow.float()

            psnr_list.append(calc_psnr(refined_f, gs))
            ssim_list.append(calc_ssim(refined_f, gs))
            epe_list.append(calc_flow_epe(pred_flow_f, gt_flow))

    return {
        "psnr_list": psnr_list,
        "mean_psnr": float(np.mean(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_epe": float(np.mean(epe_list)),
        "median_latency_ms": float(np.median(latencies)),
        "fps": 1000.0 / float(np.median(latencies))
    }

def main():
    print("=" * 70)
    print("  Lightweight CNN Rolling Shutter Model Compression Benchmark")
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Checkpoint: {CKPT_PATH}")
    print("=" * 70)

    # 1. Load trained model
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    base_flow_net = CoarseToFineFlowNet().to(DEVICE)
    base_refine_net = BorderRefinementHead().to(DEVICE)
    base_flow_net.load_state_dict(ckpt["flow_net_state"])
    base_refine_net.load_state_dict(ckpt["refine_net_state"])
    
    total_params = sum(p.numel() for p in base_flow_net.parameters()) + sum(p.numel() for p in base_refine_net.parameters())
    print(f"Total Model Parameters: {total_params:,} ({total_params/1e6:.3f}M)")

    # 2. Select exactly 50 test samples evenly spaced
    all_files = sorted(Path(TEST_DIR).glob("*.npz"))
    indices = np.linspace(0, len(all_files) - 1, min(N_SAMPLES, len(all_files))).astype(int)
    picked_files = [all_files[i] for i in indices]
    print(f"Loaded {len(picked_files)} test samples evenly spaced from {len(all_files)} total test samples.")

    test_data = []
    uncorr_psnr = []
    uncorr_ssim = []

    for p in picked_files:
        with np.load(p, allow_pickle=False) as d:
            rs_np = d["condition"]
            gs_np = d["gs"]
            gt_flow_np = d["out_flow"]

        rs = torch.from_numpy(rs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        gs = torch.from_numpy(gs_np.copy()).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        gt_flow = torch.from_numpy(gt_flow_np.copy()).float().unsqueeze(0).to(DEVICE)

        if rs.shape[-1] != 256 or rs.shape[-2] != 256:
            rs = F.interpolate(rs, (256, 256), mode="bilinear", align_corners=False)
            gs = F.interpolate(gs, (256, 256), mode="bilinear", align_corners=False)
            gt_flow = F.interpolate(gt_flow, (256, 256), mode="bilinear", align_corners=False)

        rs_low = F.interpolate(rs, (64, 64), mode="bilinear", align_corners=False)
        h_low, w_low = rs_low.shape[-2:]
        y_coord = torch.linspace(0, 1, h_low, device=DEVICE).view(-1, 1).expand(h_low, w_low).unsqueeze(0).unsqueeze(0)
        rs_low_4ch = torch.cat([rs_low, y_coord], dim=1)

        uncorr_psnr.append(calc_psnr(rs, gs))
        uncorr_ssim.append(calc_ssim(rs, gs))

        test_data.append({
            "path": str(p),
            "rs": rs,
            "gs": gs,
            "gt_flow": gt_flow,
            "rs_low_4ch": rs_low_4ch
        })

    mean_uncorr_psnr = float(np.mean(uncorr_psnr))
    mean_uncorr_ssim = float(np.mean(uncorr_ssim))
    print(f"Uncorrected RS Input Baseline: PSNR = {mean_uncorr_psnr:.2f} dB, SSIM = {mean_uncorr_ssim:.4f}")

    # 3. Variants setup
    variants = [
        ("Baseline fp32", "fp32", 0.0, 32, False, False),
        ("fp16 (autocast)", "fp16_auto", 0.0, 16, True, False),
        ("fp16 (half weights)", "fp16_half", 0.0, 16, False, True),
        ("W8 weights", "w8", 0.0, 8, False, False),
        ("W8A8 (weights+activations)", "w8a8", 0.0, 8, False, False),
        ("W4 weights", "w4", 0.0, 4, False, False),
        ("Prune 30%", "prune_30", 0.30, 32, False, False),
        ("Prune 50%", "prune_50", 0.50, 32, False, False),
        ("Prune 70%", "prune_70", 0.70, 32, False, False),
        ("Prune 30% + W8", "prune_30_w8", 0.30, 8, False, False),
        ("Prune 50% + W8", "prune_50_w8", 0.50, 8, False, False),
    ]

    results = []
    base_psnr_list = None

    for name, vtype, target_sp, bits, is_auto, is_half in variants:
        print(f"\nRunning variant: {name} ...")
        achieved_sp = 0.0

        if vtype == "fp32":
            f_net = copy.deepcopy(base_flow_net)
            r_net = copy.deepcopy(base_refine_net)
            ideal_size_mb = (total_params * 4) / (1024 ** 2)
        elif vtype == "fp16_auto":
            f_net = copy.deepcopy(base_flow_net)
            r_net = copy.deepcopy(base_refine_net)
            ideal_size_mb = (total_params * 2) / (1024 ** 2)
        elif vtype == "fp16_half":
            f_net = copy.deepcopy(base_flow_net)
            r_net = copy.deepcopy(base_refine_net)
            ideal_size_mb = (total_params * 2) / (1024 ** 2)
        elif vtype == "w8":
            f_net = apply_weight_quant(base_flow_net, bits=8)
            r_net = apply_weight_quant(base_refine_net, bits=8)
            ideal_size_mb = (total_params * 1) / (1024 ** 2)
        elif vtype == "w8a8":
            f_net = apply_w8a8(base_flow_net)
            r_net = apply_w8a8(base_refine_net)
            ideal_size_mb = (total_params * 1) / (1024 ** 2)
        elif vtype == "w4":
            f_net = apply_weight_quant(base_flow_net, bits=4)
            r_net = apply_weight_quant(base_refine_net, bits=4)
            ideal_size_mb = (total_params * 0.5) / (1024 ** 2)
        elif vtype.startswith("prune_") and not vtype.endswith("_w8"):
            f_net, sp_f = apply_magnitude_pruning(base_flow_net, sparsity=target_sp)
            r_net, sp_r = apply_magnitude_pruning(base_refine_net, sparsity=target_sp)
            achieved_sp = (sp_f + sp_r) / 2.0
            # Dense size unchanged in fp32
            ideal_size_mb = (total_params * 4) / (1024 ** 2)
        elif vtype.endswith("_w8"):
            f_net, sp_f = apply_prune_and_quant(base_flow_net, sparsity=target_sp, bits=8)
            r_net, sp_r = apply_prune_and_quant(base_refine_net, sparsity=target_sp, bits=8)
            achieved_sp = (sp_f + sp_r) / 2.0
            # Sparse int8: non-zero bytes + 1-bit mask per parameter
            ideal_size_mb = (total_params * (1.0 - achieved_sp) * 1.0 + total_params * (1.0 / 8.0)) / (1024 ** 2)

        res = evaluate_variant(f_net, r_net, test_data, is_fp16=is_auto, is_half=is_half)

        if vtype == "fp32":
            base_psnr_list = res["psnr_list"]
            delta_str = "0"
        else:
            diffs = np.array(res["psnr_list"]) - np.array(base_psnr_list)
            mean_d = float(np.mean(diffs))
            ci = 1.96 * float(np.std(diffs)) / np.sqrt(len(diffs))
            delta_str = f"{mean_d:+.2f} ± {ci:.2f}"

        comp_ratio = ((total_params * 4) / (1024 ** 2)) / ideal_size_mb

        item = {
            "method": name,
            "steps": "1 (feed-forward)",
            "sparsity": f"{achieved_sp*100:.1f}%",
            "size_mb": round(ideal_size_mb, 2),
            "compression": f"{comp_ratio:.1f}x",
            "latency_ms": round(res["median_latency_ms"], 2),
            "fps": round(res["fps"], 1),
            "psnr": round(res["mean_psnr"], 2),
            "delta_psnr": delta_str,
            "ssim": round(res["mean_ssim"], 4),
            "epe": round(res["mean_epe"], 3),
        }
        results.append(item)
        print(f"  -> PSNR: {item['psnr']} dB ({item['delta_psnr']}) | SSIM: {item['ssim']} | EPE: {item['epe']} px | Latency: {item['latency_ms']} ms ({item['fps']} FPS) | Size: {item['size_mb']} MB")

    out_file = r"D:\c data\Desktop\css\checkpoints_real\cnn_compression_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "uncorrected": {"psnr": mean_uncorr_psnr, "ssim": mean_uncorr_ssim},
            "results": results
        }, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    main()
