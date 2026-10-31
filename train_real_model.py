"""
Training on the Real-World RS-Diffusion Dataset (RS-Real)
==========================================================
Key Features:
1. Direct Ground-Truth Optical Flow Supervision using `out_flow`:
   - L_flow = EPE / Charbonnier loss between predicted flow and GT flow.
2. Photometric Charbonnier Loss + Differentiable SSIM Loss (1 - SSIM)
3. Valid-Mask Warping (identifies in-canvas vs disoccluded out-of-bounds regions)
4. BorderRefinementHead (dilated residual CNN) to inpaint and clean up stretched borders
5. Scanline Timing Proxy (Y-coordinate normalization channel)
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

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ── Dataset ──────────────────────────────────────────────────────────────────
class RealRSDataset(Dataset):
    def __init__(self, root, flow_size=64, image_size=256, augment=True):
        self.files = sorted(Path(root).rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {root!r}.")
        self.flow_size = int(flow_size)
        self.image_size = int(image_size)
        self.augment = augment
        self.valid_files = []

        for p in self.files:
            try:
                # Quick size check before loading
                if p.stat().st_size > 10000:
                    self.valid_files.append(p)
            except Exception:
                continue

        if not self.valid_files:
            raise ValueError(f"No valid NPZ samples in {root!r}.")
        print(f"Loaded {len(self.valid_files)} real samples from {root}")

    def __len__(self):
        return len(self.valid_files)

    @staticmethod
    def _to_float_chw(arr):
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
        if t.max() > 1.0:
            t = t / 255.0
        return t

    def __getitem__(self, idx):
        p = self.valid_files[idx]
        with np.load(p, allow_pickle=False) as d:
            rs = self._to_float_chw(d["condition"])
            gs = self._to_float_chw(d["gs"])
            gt_flow = torch.from_numpy(d["out_flow"].copy()).float()   # (2, H, W)

        # Ensure spatial size matches image_size
        if rs.shape[-1] != self.image_size or rs.shape[-2] != self.image_size:
            rs = F.interpolate(rs[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            gs = F.interpolate(gs[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            gt_flow = F.interpolate(gt_flow[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]

        # Horizontal flip augmentation
        if self.augment and random.random() < 0.5:
            rs = rs.flip(-1)
            gs = gs.flip(-1)
            gt_flow = gt_flow.flip(-1)
            gt_flow[0] = -gt_flow[0]   # invert horizontal flow direction

        # Low-res input for the flow network
        rs_low = F.interpolate(rs[None], (self.flow_size, self.flow_size), mode="bilinear", align_corners=False)[0]

        # Scan-line timing proxy (Y-coordinate normalization)
        h_low, w_low = rs_low.shape[-2:]
        y_coord = torch.linspace(0, 1, h_low).view(-1, 1).expand(h_low, w_low).unsqueeze(0)
        rs_low_4ch = torch.cat([rs_low, y_coord], dim=0)

        # Ground truth flow downsampled to flow_size for multi-scale supervision
        gt_flow_low = F.interpolate(gt_flow[None], (self.flow_size, self.flow_size), mode="bilinear", align_corners=False)[0]
        # Rescale displacement magnitude proportionally
        scale = float(self.flow_size) / float(self.image_size)
        gt_flow_low = gt_flow_low * scale

        return rs_low_4ch, rs, gs, gt_flow, gt_flow_low


# ── Model Architecture ───────────────────────────────────────────────────────
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
    def __init__(self, channels=(48, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(4, c1, stride=1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.enc4 = ConvBlock(c3, c4, stride=2)

        self.flow8 = FlowHead(c4)
        self.dec3 = ConvBlock(c4 + c3 + 2, c3)
        self.flow16 = FlowHead(c3)
        self.dec2 = ConvBlock(c3 + c2 + 2, c2)
        self.flow32 = FlowHead(c2)
        self.dec1 = ConvBlock(c2 + c1 + 2, c1)
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
        d3 = self.dec3(torch.cat([e3, F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), flow], dim=1))
        flow = flow + self.flow16(d3)
        flows.append(flow)

        flow = self.upsample_flow(flow, e2.shape[-2:])
        d2 = self.dec2(torch.cat([e2, F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), flow], dim=1))
        flow = flow + self.flow32(d2)
        flows.append(flow)

        flow = self.upsample_flow(flow, e1.shape[-2:])
        d1 = self.dec1(torch.cat([e1, F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), flow], dim=1))
        flow = flow + self.flow64(d1)
        flows.append(flow)

        return flows


class BorderRefinementHead(nn.Module):
    """Residual CNN with dilated convolutions to refine valid pixels only.
    Zero (black) border regions produced by zeros-padding are NOT refined so
    they stay clean black — the residual is gated by a valid-pixel mask."""
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
        # valid mask: pixels that were actually sampled (not the zeros border)
        valid = (warped.sum(dim=1, keepdim=True) > 1e-4).float()
        refined = warped + self.net(warped) * valid
        return refined.clamp(0.0, 1.0)


# ── Warping + Losses ─────────────────────────────────────────────────────────
def upsample_and_rescale_flow(flow_low, output_size):
    out_h, out_w = output_size
    low_h, low_w = flow_low.shape[-2:]
    flow = F.interpolate(flow_low, size=(out_h, out_w), mode="bilinear", align_corners=False)
    scale = flow.new_tensor([out_w / low_w, out_h / low_h]).view(1, 2, 1, 1)
    return flow * scale


def warp_image_with_mask(image, flow):
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    sample_x = xx.unsqueeze(0).expand(b, -1, -1) + flow[:, 0]
    sample_y = yy.unsqueeze(0).expand(b, -1, -1) + flow[:, 1]

    # 1 inside canvas, 0 outside (disocclusion)
    valid = ((sample_x >= 0) & (sample_x <= w - 1) & (sample_y >= 0) & (sample_y <= h - 1)).float().unsqueeze(1)

    norm_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, valid


EPS_CHARB = 1e-3

def charbonnier_loss(pred, target, mask=None):
    diff = pred - target
    loss = torch.sqrt(diff * diff + EPS_CHARB ** 2)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def flow_epe_loss(pred_flow, gt_flow):
    """Endpoint Error (EPE) Charbonnier loss."""
    diff = pred_flow - gt_flow
    epe = torch.sqrt(torch.sum(diff ** 2, dim=1, keepdim=True) + EPS_CHARB ** 2)
    return epe.mean()


def multiscale_flow_supervision(flows, gt_flow_low):
    """Direct multi-scale supervision against ground-truth flow."""
    weights = [0.1, 0.2, 0.4, 1.0]
    loss = 0.0
    for w, pred in zip(weights, flows):
        h, ww = pred.shape[-2:]
        gt_ds = F.interpolate(gt_flow_low, size=(h, ww), mode="bilinear", align_corners=False)
        scale = float(h) / float(gt_flow_low.shape[-2])
        gt_ds = gt_ds * scale
        loss = loss + w * flow_epe_loss(pred, gt_ds)
    return loss


def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(pred, 3, 1, padding=1)
    mu2 = F.avg_pool2d(target, 3, 1, padding=1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1mu2 = mu1 * mu2
    s1 = F.avg_pool2d(pred * pred, 3, 1, padding=1) - mu1_sq
    s2 = F.avg_pool2d(target * target, 3, 1, padding=1) - mu2_sq
    s12 = F.avg_pool2d(pred * target, 3, 1, padding=1) - mu1mu2
    ssim_map = ((2 * mu1mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1.0 - ssim_map.mean()


def border_mask_loss(pred_flow, gt_flow):
    """Binary cross-entropy between predicted and GT out-of-bounds (black border) masks.
    Encourages the model to predict the correct spatial extent of the disocclusion zone."""
    b, _, h, w = pred_flow.shape
    device = pred_flow.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=pred_flow.dtype),
        torch.arange(w, device=device, dtype=pred_flow.dtype),
        indexing="ij",
    )
    def oob_mask(flow):
        sx = xx.unsqueeze(0) + flow[:, 0]
        sy = yy.unsqueeze(0) + flow[:, 1]
        return ((sx < 0) | (sx > w - 1) | (sy < 0) | (sy > h - 1)).float()

    pred_mask = oob_mask(pred_flow)
    with torch.no_grad():
        gt_mask = oob_mask(gt_flow)

    # Clamp to avoid log(0)
    pred_clamped = pred_mask.clamp(1e-4, 1 - 1e-4)
    return F.binary_cross_entropy(pred_clamped, gt_mask)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in features.parameters():
            p.requires_grad = False
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction, target):
        pred = (prediction - self.mean) / self.std
        tgt = (target - self.mean) / self.std
        pf = self.features(pred)
        with torch.no_grad():
            tf = self.features(tgt)
        return F.l1_loss(pf, tf)


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


# ── Training Loop ────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("-" * 65)
    print("  Real-World Rolling Shutter Correction Trainer (Direct GT Flow)")
    print("-" * 65)
    print(f"  Device : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Data   : {args.data_dir}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}")
    print("-" * 65)

    full_ds = RealRSDataset(args.data_dir, flow_size=args.flow_size, image_size=args.image_size, augment=True)
    n = len(full_ds)
    n_train = max(1, int(0.85 * n))
    n_val = n - n_train
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=gen)
    print(f"  Split  : Train={len(train_ds)} | Val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    flow_net = CoarseToFineFlowNet().to(device)
    refine_net = BorderRefinementHead(ch=args.refine_channels).to(device)

    perc_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(flow_net.parameters()) + list(refine_net.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    history = []

    t0 = time.time()
    W_FLOW = args.flow_weight
    W_PIX = args.pixel_weight
    W_SSIM = args.ssim_weight
    W_PERC = args.perceptual_weight
    W_BORDER = args.border_mask_weight

    print(f"\nStarting {args.epochs} training epochs...")

    for epoch in range(1, args.epochs + 1):
        flow_net.train()
        refine_net.train()
        run = {k: 0.0 for k in ("tot", "flow", "pix", "ssim_l", "perc", "border")}
        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d} [Train]", leave=False)

        for rs_4ch, rs_full, gs_full, gt_flow, gt_flow_low in pbar:
            rs_4ch, rs_full, gs_full, gt_flow, gt_flow_low = [
                t.to(device, non_blocking=True) for t in (rs_4ch, rs_full, gs_full, gt_flow, gt_flow_low)
            ]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                flows = flow_net(rs_4ch)
                flow_full = upsample_and_rescale_flow(flows[-1], rs_full.shape[-2:])
                warped, valid_mask = warp_image_with_mask(rs_full, flow_full)
                refined = refine_net(warped)

                # Direct Flow Supervision
                l_flow = multiscale_flow_supervision(flows, gt_flow_low) + 0.5 * flow_epe_loss(flow_full, gt_flow)
                # Photometric & Border Loss
                l_pix = charbonnier_loss(refined, gs_full, mask=valid_mask.expand_as(refined))
                l_ssim = ssim_loss(refined, gs_full)
                l_perc = perc_loss_fn(refined.float(), gs_full.float())
                l_border = border_mask_loss(flow_full, gt_flow)

                loss = W_FLOW * l_flow + W_PIX * l_pix + W_SSIM * l_ssim + W_PERC * l_perc + W_BORDER * l_border

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(flow_net.parameters()) + list(refine_net.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            run["tot"] += loss.item()
            run["flow"] += l_flow.item()
            run["pix"] += l_pix.item()
            run["ssim_l"] += l_ssim.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", flow=f"{l_flow.item():.3f}", pix=f"{l_pix.item():.4f}")

        # Validation
        flow_net.eval()
        refine_net.eval()
        vrun = {k: 0.0 for k in ("tot", "flow", "pix", "ssim_l")}
        val_psnr_list = []
        val_ssim_list = []

        with torch.no_grad():
            for rs_4ch, rs_full, gs_full, gt_flow, gt_flow_low in val_loader:
                rs_4ch, rs_full, gs_full, gt_flow, gt_flow_low = [
                    t.to(device, non_blocking=True) for t in (rs_4ch, rs_full, gs_full, gt_flow, gt_flow_low)
                ]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    flows = flow_net(rs_4ch)
                    flow_full = upsample_and_rescale_flow(flows[-1], rs_full.shape[-2:])
                    warped, valid_mask = warp_image_with_mask(rs_full, flow_full)
                    refined = refine_net(warped)

                    l_flow = multiscale_flow_supervision(flows, gt_flow_low) + 0.5 * flow_epe_loss(flow_full, gt_flow)
                    l_pix = charbonnier_loss(refined, gs_full, mask=valid_mask.expand_as(refined))
                    l_ssim = ssim_loss(refined, gs_full)
                    l_perc = perc_loss_fn(refined.float(), gs_full.float())
                    loss = W_FLOW * l_flow + W_PIX * l_pix + W_SSIM * l_ssim + W_PERC * l_perc

                vrun["tot"] += loss.item()
                vrun["flow"] += l_flow.item()
                vrun["pix"] += l_pix.item()
                vrun["ssim_l"] += l_ssim.item()
                val_psnr_list.append(calc_psnr(refined, gs_full))
                val_ssim_list.append(calc_ssim(refined, gs_full))

        scheduler.step()
        n_tr = max(len(train_loader), 1)
        n_vl = max(len(val_loader), 1)

        metrics = {
            "epoch": epoch,
            "train_loss": run["tot"] / n_tr,
            "train_flow": run["flow"] / n_tr,
            "train_pix": run["pix"] / n_tr,
            "train_ssim": run["ssim_l"] / n_tr,
            "val_loss": vrun["tot"] / n_vl,
            "val_flow": vrun["flow"] / n_vl,
            "val_pix": vrun["pix"] / n_vl,
            "val_psnr": float(np.mean(val_psnr_list)),
            "val_ssim": float(np.mean(val_ssim_list)),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        print(f"Ep {epoch:02d} | Train Loss={metrics['train_loss']:.4f} Flow={metrics['train_flow']:.3f} | "
              f"Val Loss={metrics['val_loss']:.4f} PSNR={metrics['val_psnr']:.2f}dB SSIM={metrics['val_ssim']:.4f} | "
              f"LR={metrics['lr']:.2e}")

        ckpt = {
            "epoch": epoch,
            "flow_net_state": flow_net.state_dict(),
            "refine_net_state": refine_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last_model_real.pt"))
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_real.pt"))
            print(f"  + New best checkpoint (val_loss: {best_val:.4f}, PSNR: {metrics['val_psnr']:.2f} dB)")

    with open(os.path.join(args.save_dir, "history_real.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_time = (time.time() - t0) / 60.0
    print(f"\nTraining completed in {total_time:.2f} minutes!")
    print(f"Best Validation Loss: {best_val:.4f}")
    print(f"All checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"D:\RS-Real\train")
    parser.add_argument("--save_dir", type=str, default=r"D:\c data\Desktop\css\checkpoints_real")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--flow_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--flow_weight", type=float, default=0.5)
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.5)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--refine_channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
