"""
Rolling Shutter Correction – Advanced Training Script v3
=========================================================
Improvements over v2
--------------------
1. **4-channel input**: RGB + vertical-coordinate map (Y_norm) → model sees scan-line timing.
2. **Multi-scale flow supervision**: ground-truth warp-consistency target at 8/16/32/64 px.
3. **Charbonnier pixel loss** (robust L1) + **SSIM loss** for perceptual sharpness.
4. **Valid-mask warping**: ignore genuinely out-of-bounds pixels in the photometric loss;
   the in-border region still gets gradient so the model learns from edge cues.
5. **Border-refinement head**: a 2-layer residual CNN that inpaints/refines the warped
   image before the photometric loss, giving the model capacity to fix edge stretching.
6. **Augmentation**: random horizontal flip + tiny brightness/contrast jitter.
"""

import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ── Dataset ──────────────────────────────────────────────────────────────────
class RollingShutterDataset(Dataset):
    def __init__(self, root, flow_size=64, image_size=256, augment=True):
        self.files = sorted(Path(root).rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {root!r}.")
        self.flow_size = int(flow_size)
        self.image_size = int(image_size)
        self.augment = augment
        self.valid_files = []
        for path in self.files:
            try:
                with np.load(path, allow_pickle=False) as d:
                    if not {"condition", "gs"}.issubset(d.files):
                        continue
                    rs, gs = np.asarray(d["condition"]), np.asarray(d["gs"])
                    if rs.ndim != 3 or gs.ndim != 3 or rs.shape != gs.shape:
                        continue
                self.valid_files.append(path)
            except Exception:
                continue
        if not self.valid_files:
            raise ValueError(f"No valid paired NPZ samples found in {root!r}.")
        print(f"Loaded {len(self.valid_files)} valid paired samples from {root}")

    def __len__(self):
        return len(self.valid_files)

    @staticmethod
    def _to_float_chw(array):
        arr = np.asarray(array)
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
        if np.issubdtype(arr.dtype, np.integer) or (tensor.numel() and tensor.max().item() > 1.0):
            tensor = tensor / 255.0
        return tensor

    def __getitem__(self, index):
        path = self.valid_files[index]
        with np.load(path, allow_pickle=False) as data:
            rs = self._to_float_chw(data["condition"])
            gs = self._to_float_chw(data["gs"])

        rs_full = F.interpolate(rs[None], size=(self.image_size, self.image_size),
                                mode="bilinear", align_corners=False)[0]
        gs_full = F.interpolate(gs[None], size=(self.image_size, self.image_size),
                                mode="bilinear", align_corners=False)[0]

        # Augmentation: random horizontal flip
        if self.augment and random.random() < 0.5:
            rs_full = rs_full.flip(-1)
            gs_full = gs_full.flip(-1)

        # Tiny colour jitter (brightness + contrast)
        if self.augment:
            b = 1.0 + random.uniform(-0.08, 0.08)
            c = 1.0 + random.uniform(-0.08, 0.08)
            rs_full = (rs_full * c + b - 1.0).clamp(0.0, 1.0)
            gs_full = (gs_full * c + b - 1.0).clamp(0.0, 1.0)

        # Low-res input for flow network
        rs_low = F.interpolate(rs_full[None], size=(self.flow_size, self.flow_size),
                               mode="bilinear", align_corners=False)[0]

        # ── Y-coordinate channel (scan-line proxy) ────────────────────────────
        # Shape (1, H, W) with values in [0, 1] from top to bottom.
        H_low, W_low = rs_low.shape[-2:]
        y_coords = torch.linspace(0, 1, H_low).view(-1, 1).expand(H_low, W_low).unsqueeze(0)
        rs_low_4ch = torch.cat([rs_low, y_coords], dim=0)   # (4, flow_size, flow_size)

        return rs_low_4ch, rs_full, gs_full


# ── Architecture ─────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FlowHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 2, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class CoarseToFineFlowNet(nn.Module):
    """U-Net style coarse-to-fine optical flow estimator.

    Input: (B, 4, H, W)  – RGB + Y-coord channel
    """
    def __init__(self, channels=(48, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(4, c1, stride=1)       # 64x64
        self.enc2 = ConvBlock(c1, c2, stride=2)      # 32x32
        self.enc3 = ConvBlock(c2, c3, stride=2)      # 16x16
        self.enc4 = ConvBlock(c3, c4, stride=2)      # 8x8

        self.flow8  = FlowHead(c4)
        self.dec3   = ConvBlock(c4 + c3 + 2, c3)
        self.flow16 = FlowHead(c3)
        self.dec2   = ConvBlock(c3 + c2 + 2, c2)
        self.flow32 = FlowHead(c2)
        self.dec1   = ConvBlock(c2 + c1 + 2, c1)
        self.flow64 = FlowHead(c1)

    @staticmethod
    def upsample_flow(flow, size):
        old_h, old_w = flow.shape[-2:]
        new_h, new_w = size
        up = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
        scale = up.new_tensor([new_w / old_w, new_h / old_h]).view(1, 2, 1, 1)
        return up * scale

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        flow = self.flow8(e4)
        flows = [flow]

        flow = self.upsample_flow(flow, e3.shape[-2:])
        d3 = self.dec3(torch.cat([e3,
                                   F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False),
                                   flow], dim=1))
        flow = flow + self.flow16(d3)
        flows.append(flow)

        flow = self.upsample_flow(flow, e2.shape[-2:])
        d2 = self.dec2(torch.cat([e2,
                                   F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False),
                                   flow], dim=1))
        flow = flow + self.flow32(d2)
        flows.append(flow)

        flow = self.upsample_flow(flow, e1.shape[-2:])
        d1 = self.dec1(torch.cat([e1,
                                   F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False),
                                   flow], dim=1))
        flow = flow + self.flow64(d1)
        flows.append(flow)

        # flows[0] → 8x8, flows[1] → 16x16, flows[2] → 32x32, flows[3] → 64x64
        return flows


class BorderRefinementHead(nn.Module):
    """Light 2-layer residual CNN to inpaint / refine border regions after warping."""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, warped):
        return (warped + self.net(warped)).clamp(0.0, 1.0)


# ── Warping + valid mask ──────────────────────────────────────────────────────
def upsample_and_rescale_flow(flow_low, output_size):
    out_h, out_w = output_size
    low_h, low_w = flow_low.shape[-2:]
    flow = F.interpolate(flow_low, size=(out_h, out_w), mode="bilinear", align_corners=False)
    scale = flow.new_tensor([out_w / low_w, out_h / low_h]).view(1, 2, 1, 1)
    return flow * scale


def warp_image_with_mask(image, flow):
    """Returns (warped, valid_mask).

    valid_mask is 1 where sample coords fall inside [0, W-1] x [0, H-1].
    """
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base_x = xx.unsqueeze(0).expand(b, -1, -1)
    base_y = yy.unsqueeze(0).expand(b, -1, -1)
    sample_x = base_x + flow[:, 0]
    sample_y = base_y + flow[:, 1]

    # Valid mask (in-canvas pixels)
    valid = ((sample_x >= 0) & (sample_x <= w - 1) &
             (sample_y >= 0) & (sample_y <= h - 1)).float().unsqueeze(1)

    norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped, valid


# ── Loss functions ────────────────────────────────────────────────────────────
EPS_CHARB = 1e-3

def charbonnier_loss(pred, target, mask=None):
    """Charbonnier (smooth L1) loss, optionally masked."""
    diff = pred - target
    loss = torch.sqrt(diff * diff + EPS_CHARB ** 2)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def ssim_loss(pred, target):
    """1 - SSIM as a differentiable loss."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(pred,   3, 1, padding=1)
    mu2 = F.avg_pool2d(target, 3, 1, padding=1)
    mu1_sq, mu2_sq, mu1mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1  = F.avg_pool2d(pred * pred,   3, 1, padding=1) - mu1_sq
    s2  = F.avg_pool2d(target * target, 3, 1, padding=1) - mu2_sq
    s12 = F.avg_pool2d(pred * target,  3, 1, padding=1) - mu1mu2
    ssim_map = ((2 * mu1mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1.0 - ssim_map.mean()


def multiscale_flow_loss(flows, rs_low_4ch, gs_full, device):
    """
    Self-supervised flow consistency:  warp RS at each scale and compare to
    downsampled GS.  This acts like direct flow supervision without needing
    ground-truth optical flow.
    """
    rs_rgb = rs_low_4ch[:, :3]    # (B, 3, flow_size, flow_size)
    loss = 0.0
    weights = [0.05, 0.1, 0.2, 1.0]   # coarse → fine
    for w, flow in zip(weights, flows):
        h, ww = flow.shape[-2:]
        rs_ds = F.interpolate(rs_rgb, size=(h, ww), mode="bilinear", align_corners=False)
        gs_ds = F.interpolate(gs_full, size=(h, ww), mode="bilinear", align_corners=False)
        # flow at this scale is already in pixel units for this resolution
        warped_ds, _ = warp_image_with_mask(rs_ds, flow)
        loss = loss + w * charbonnier_loss(warped_ds, gs_ds)
    return loss


# ── Perceptual Loss ───────────────────────────────────────────────────────────
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in features.parameters():
            p.requires_grad = False
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction, target):
        pred = (prediction - self.mean) / self.std
        tgt  = (target - self.mean) / self.std
        pf = self.features(pred)
        with torch.no_grad():
            tf = self.features(tgt)
        return F.l1_loss(pf, tf)


# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(data_range ** 2 / mse)


def calculate_ssim_metric(pred, target):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(pred,   3, 1, padding=1)
    mu2 = F.avg_pool2d(target, 3, 1, padding=1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1mu2 = mu1 * mu2
    s1  = F.avg_pool2d(pred * pred,   3, 1, padding=1) - mu1_sq
    s2  = F.avg_pool2d(target * target, 3, 1, padding=1) - mu2_sq
    s12 = F.avg_pool2d(pred * target,  3, 1, padding=1) - mu1mu2
    ssim_map = ((2 * mu1mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("-" * 60)
    print("  Rolling Shutter Correction - Advanced Trainer v3")
    print("-" * 60)
    print(f"  Device : {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Data   : {args.data_dir}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch_size}")
    print(f"  Losses : Charbonnier + SSIM + Perceptual + MultiScale-Flow")
    print("-" * 60)

    # ── Dataset & loaders ───────────────────────────────────────────────────
    full_ds = RollingShutterDataset(args.data_dir, flow_size=args.flow_size,
                                    image_size=args.image_size, augment=True)
    n = len(full_ds)
    n_train = max(1, int(0.70 * n))
    n_val   = max(1, int(0.15 * n))
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_train, n_val, n_test = n - 2, 1, 1

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test], generator=gen)
    print(f"  Split  : Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

    # ── Models ──────────────────────────────────────────────────────────────
    flow_net   = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead(ch=args.refine_channels).to(device)

    params_flow   = sum(p.numel() for p in flow_net.parameters()   if p.requires_grad)
    params_refine = sum(p.numel() for p in refine_net.parameters() if p.requires_grad)
    print(f"  FlowNet params   : {params_flow:,}  ({params_flow/1e6:.3f}M)")
    print(f"  RefineNet params : {params_refine:,}  ({params_refine/1e6:.3f}M)")

    perc_loss_fn = VGGPerceptualLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(flow_net.parameters()) + list(refine_net.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    history  = []

    print(f"\n  Starting training for {args.epochs} epochs …\n")
    t0 = time.time()

    # ── Loss weights ────────────────────────────────────────────────────────
    W_PIX   = args.pixel_weight       # Charbonnier
    W_SSIM  = args.ssim_weight        # 1 – SSIM
    W_PERC  = args.perceptual_weight  # VGG perceptual
    W_FLOW  = args.flow_weight        # multi-scale flow consistency

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        flow_net.train()
        refine_net.train()
        run = {k: 0.0 for k in ("tot", "pix", "ssim_l", "perc", "flow")}
        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d} [Train]", leave=False)

        for rs_4ch, rs_full, gs_full in pbar:
            rs_4ch, rs_full, gs_full = [t.to(device, non_blocking=True)
                                         for t in (rs_4ch, rs_full, gs_full)]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                flows = flow_net(rs_4ch)              # list of 4 multi-scale flows
                flow_full = upsample_and_rescale_flow(flows[-1], rs_full.shape[-2:])
                warped, valid_mask = warp_image_with_mask(rs_full, flow_full)
                refined = refine_net(warped)

                # ── Losses ────────────────────────────────────────────────
                # Photometric: Charbonnier (mask: ignore extreme OOB pixels)
                l_pix  = charbonnier_loss(refined, gs_full, mask=valid_mask.expand_as(refined))
                l_ssim = ssim_loss(refined, gs_full)
                l_perc = perc_loss_fn(refined.float(), gs_full.float())
                l_flow = multiscale_flow_loss(flows, rs_4ch, gs_full, device)

                loss = W_PIX * l_pix + W_SSIM * l_ssim + W_PERC * l_perc + W_FLOW * l_flow

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(flow_net.parameters()) + list(refine_net.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            run["tot"]    += loss.item()
            run["pix"]    += l_pix.item()
            run["ssim_l"] += l_ssim.item()
            run["perc"]   += l_perc.item()
            run["flow"]   += l_flow.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", pix=f"{l_pix.item():.4f}")

        # ── Validation ────────────────────────────────────────────────────
        flow_net.eval()
        refine_net.eval()
        vrun = {k: 0.0 for k in ("tot", "pix", "ssim_l")}

        with torch.no_grad():
            for rs_4ch, rs_full, gs_full in val_loader:
                rs_4ch, rs_full, gs_full = [t.to(device, non_blocking=True)
                                             for t in (rs_4ch, rs_full, gs_full)]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    flows = flow_net(rs_4ch)
                    flow_full = upsample_and_rescale_flow(flows[-1], rs_full.shape[-2:])
                    warped, valid_mask = warp_image_with_mask(rs_full, flow_full)
                    refined = refine_net(warped)
                    l_pix  = charbonnier_loss(refined, gs_full, mask=valid_mask.expand_as(refined))
                    l_ssim = ssim_loss(refined, gs_full)
                    l_perc = perc_loss_fn(refined.float(), gs_full.float())
                    l_flow = multiscale_flow_loss(flows, rs_4ch, gs_full, device)
                    loss   = W_PIX * l_pix + W_SSIM * l_ssim + W_PERC * l_perc + W_FLOW * l_flow
                vrun["tot"]    += loss.item()
                vrun["pix"]    += l_pix.item()
                vrun["ssim_l"] += l_ssim.item()

        scheduler.step()
        n_tr = max(len(train_loader), 1)
        n_vl = max(len(val_loader), 1)

        metrics = {
            "epoch":       epoch,
            "train_loss":  run["tot"]    / n_tr,
            "train_pix":   run["pix"]    / n_tr,
            "train_ssim":  run["ssim_l"] / n_tr,
            "train_perc":  run["perc"]   / n_tr,
            "train_flow":  run["flow"]   / n_tr,
            "val_loss":    vrun["tot"]   / n_vl,
            "val_pix":     vrun["pix"]   / n_vl,
            "val_ssim":    vrun["ssim_l"]/ n_vl,
            "lr":          optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        print(f"Ep {epoch:02d}  Train={metrics['train_loss']:.4f}"
              f"  Val={metrics['val_loss']:.4f}"
              f"  pix={metrics['train_pix']:.4f}"
              f"  ssim={metrics['train_ssim']:.4f}"
              f"  flow={metrics['train_flow']:.4f}"
              f"  lr={metrics['lr']:.2e}")

        # Checkpoint
        ckpt = {
            "epoch":                epoch,
            "flow_net_state":       flow_net.state_dict(),
            "refine_net_state":     refine_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history":              history,
            "metrics":              metrics,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last_model.pt"))
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  + New best checkpoint  (val_loss: {best_val:.4f})")

    # Save history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t0
    print(f"\nTraining finished in {total_time/60:.2f} min.")

    # ── Final Test Evaluation ─────────────────────────────────────────────
    print(f"\n--- Test Set Evaluation ---")
    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pt"),
                            map_location=device, weights_only=False)
    flow_net.load_state_dict(best_ckpt["flow_net_state"])
    refine_net.load_state_dict(best_ckpt["refine_net_state"])
    flow_net.eval()
    refine_net.eval()

    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for rs_4ch, rs_full, gs_full in test_loader:
            rs_4ch, rs_full, gs_full = [t.to(device) for t in (rs_4ch, rs_full, gs_full)]
            flows     = flow_net(rs_4ch)
            flow_full = upsample_and_rescale_flow(flows[-1], rs_full.shape[-2:])
            warped, _ = warp_image_with_mask(rs_full, flow_full)
            refined   = refine_net(warped)
            psnr_list.append(calculate_psnr(refined, gs_full))
            ssim_list.append(calculate_ssim_metric(refined, gs_full))

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    print(f"  Samples : {len(test_loader)}")
    print(f"  PSNR    : {mean_psnr:.3f} dB")
    print(f"  SSIM    : {mean_ssim:.4f}")

    results = {
        "epochs": args.epochs,
        "best_val_loss":        best_val,
        "test_mean_psnr_dB":    mean_psnr,
        "test_mean_ssim":       mean_ssim,
        "training_time_minutes": total_time / 60.0,
        "losses_used": "Charbonnier + SSIM + VGG-Perceptual + MultiScale-Flow",
    }
    with open(os.path.join(args.save_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  All checkpoints & results saved in: {args.save_dir}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Rolling-Shutter Correction Trainer v3")
    parser.add_argument("--data_dir",          type=str,   default=r"D:\RS-Real\train")
    parser.add_argument("--save_dir",          type=str,   default=os.path.join(
                                                             os.path.dirname(os.path.abspath(__file__)),
                                                             "checkpoints"))
    parser.add_argument("--epochs",            type=int,   default=40)
    parser.add_argument("--batch_size",        type=int,   default=6)
    parser.add_argument("--flow_size",         type=int,   default=64)
    parser.add_argument("--image_size",        type=int,   default=256)
    parser.add_argument("--lr",                type=float, default=2e-4)
    parser.add_argument("--weight_decay",      type=float, default=1e-4)
    parser.add_argument("--pixel_weight",      type=float, default=1.0)
    parser.add_argument("--ssim_weight",       type=float, default=0.5)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--flow_weight",       type=float, default=0.15)
    parser.add_argument("--refine_channels",   type=int,   default=32)
    parser.add_argument("--workers",           type=int,   default=0)
    parser.add_argument("--device",            type=str,   default="cuda")
    args = parser.parse_args()
    train(args)
