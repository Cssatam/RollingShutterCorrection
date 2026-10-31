"""
Visualize results from the v3 advanced model (with BorderRefinementHead).
Saves visual_comparison_v3.png in the checkpoints folder.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Copy model classes (must match train_model.py) ───────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class FlowHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 2, 3, padding=1),
        )
    def forward(self, x): return self.net(x)

class CoarseToFineFlowNet(nn.Module):
    def __init__(self, channels=(48, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(4, c1, stride=1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.enc4 = ConvBlock(c3, c4, stride=2)
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
        return up * up.new_tensor([new_w / old_w, new_h / old_h]).view(1, 2, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        flow = self.flow8(e4); flows = [flow]
        flow = self.upsample_flow(flow, e3.shape[-2:])
        d3 = self.dec3(torch.cat([e3, F.interpolate(e4, e3.shape[-2:], mode="bilinear", align_corners=False), flow], 1))
        flow = flow + self.flow16(d3); flows.append(flow)
        flow = self.upsample_flow(flow, e2.shape[-2:])
        d2 = self.dec2(torch.cat([e2, F.interpolate(d3, e2.shape[-2:], mode="bilinear", align_corners=False), flow], 1))
        flow = flow + self.flow32(d2); flows.append(flow)
        flow = self.upsample_flow(flow, e1.shape[-2:])
        d1 = self.dec1(torch.cat([e1, F.interpolate(d2, e1.shape[-2:], mode="bilinear", align_corners=False), flow], 1))
        flow = flow + self.flow64(d1); flows.append(flow)
        return flows

class BorderRefinementHead(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
    def forward(self, w): return (w + self.net(w)).clamp(0, 1)

def upsample_and_rescale_flow(flow_low, output_size):
    out_h, out_w = output_size
    low_h, low_w = flow_low.shape[-2:]
    flow = F.interpolate(flow_low, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return flow * flow.new_tensor([out_w / low_w, out_h / low_h]).view(1, 2, 1, 1)

def warp_image(image, flow):
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
    sample_x = xx.unsqueeze(0).expand(b,-1,-1) + flow[:,0]
    sample_y = yy.unsqueeze(0).expand(b,-1,-1) + flow[:,1]
    norm_x = 2.0 * sample_x / max(w-1,1) - 1.0
    norm_y = 2.0 * sample_y / max(h-1,1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)

def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10*math.log10(1.0/mse)

def ssim_val(a, b):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(a, 3, 1, 1); mu2 = F.avg_pool2d(b, 3, 1, 1)
    s1 = F.avg_pool2d(a*a, 3, 1, 1) - mu1**2
    s2 = F.avg_pool2d(b*b, 3, 1, 1) - mu2**2
    s12= F.avg_pool2d(a*b, 3, 1, 1) - mu1*mu2
    return (((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()

# ── Load checkpoint ───────────────────────────────────────────────────────────
CKPT  = r"D:\c data\Desktop\css\checkpoints\best_model.pt"
DATA  = r"D:\c data\Desktop\css\data\RS-Real\train"
OUT   = r"D:\c data\Desktop\css\checkpoints\visual_comparison_v3.png"
FLOW_SIZE  = 64
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
flow_net   = CoarseToFineFlowNet().to(device)
refine_net = BorderRefinementHead(ch=32).to(device)
flow_net.load_state_dict(ckpt["flow_net_state"])
refine_net.load_state_dict(ckpt["refine_net_state"])
flow_net.eval(); refine_net.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

# ── Sample some test images ───────────────────────────────────────────────────
files = sorted(Path(DATA).rglob("*.npz"))
np.random.seed(7)
chosen = np.random.choice(len(files), size=min(4, len(files)), replace=False)

N = len(chosen)
fig, axes = plt.subplots(N, 5, figsize=(22, 6*N+1))
fig.patch.set_facecolor("#1a1a2e")
if N == 1:
    axes = [axes]

col_titles = ["Rolling Shutter (Input)", "Warped Only", "Refined Output", "Ground Truth", "Error Map"]
for col_i, title in enumerate(col_titles):
    axes[0][col_i].set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)

for row, idx in enumerate(chosen):
    path = files[idx]
    with np.load(path, allow_pickle=False) as d:
        rs_np = np.asarray(d["condition"])
        gs_np = np.asarray(d["gs"])

    def to_tensor(arr):
        t = torch.from_numpy(arr.copy()).permute(2,0,1).float()
        if t.max() > 1: t /= 255.0
        return t

    rs = F.interpolate(to_tensor(rs_np)[None], (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    gs = F.interpolate(to_tensor(gs_np)[None], (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    rs_low = F.interpolate(rs, (FLOW_SIZE, FLOW_SIZE), mode="bilinear", align_corners=False)
    H, W = rs_low.shape[-2:]
    y_coord = torch.linspace(0,1,H).view(-1,1).expand(H,W).unsqueeze(0).unsqueeze(0)
    rs_4ch = torch.cat([rs_low, y_coord], dim=1).to(device)

    with torch.no_grad():
        flows     = flow_net(rs_4ch)
        flow_full = upsample_and_rescale_flow(flows[-1], rs.shape[-2:])
        warped    = warp_image(rs.to(device), flow_full)
        refined   = refine_net(warped)

    rs_show      = rs[0].permute(1,2,0).numpy()
    warped_show  = warped[0].cpu().permute(1,2,0).numpy()
    refined_show = refined[0].cpu().permute(1,2,0).numpy()
    gs_show      = gs[0].permute(1,2,0).numpy()
    err          = np.abs(refined_show - gs_show).mean(axis=2)

    p_in  = psnr(rs.to(device),     gs.to(device))
    p_warp= psnr(warped,             gs.to(device))
    p_ref = psnr(refined,            gs.to(device))
    s_ref = ssim_val(refined,        gs.to(device))

    images = [rs_show, warped_show, refined_show, gs_show, err]
    cmaps  = [None, None, None, None, "hot"]
    labels = [
        f"Input PSNR: {p_in:.1f} dB",
        f"Warped PSNR: {p_warp:.1f} dB",
        f"Refined PSNR: {p_ref:.1f} dB\nSSIM: {s_ref:.4f}",
        "Ground Truth",
        f"Max err: {err.max():.3f}",
    ]
    for col_i, (img, cmap, label) in enumerate(zip(images, cmaps, labels)):
        ax = axes[row][col_i]
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=0.3)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_xlabel(label, color="white", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

fig.suptitle(
    f"Rolling Shutter Correction v3  |  40 Epochs  |  1000 Samples\n"
    f"Charbonnier + SSIM + Perceptual + MultiScale-Flow + BorderRefinement",
    color="white", fontsize=15, fontweight="bold", y=1.01
)
plt.savefig(OUT, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUT}")
