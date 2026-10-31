import os
from pathlib import Path
import numpy as np

def generate_synthetic_rsc_dataset(output_dir, num_samples=12, height=256, width=256):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    print(f"Generating {num_samples} synthetic rolling-shutter samples in {output_dir}...")

    for i in range(num_samples):
        # Create a textured synthetic image (e.g. geometric patterns / color gradients)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        
        freq_x = np.random.uniform(2, 6)
        freq_y = np.random.uniform(2, 6)
        r = 0.5 + 0.4 * np.sin(2 * np.pi * xx / width * freq_x)
        g = 0.5 + 0.4 * np.cos(2 * np.pi * yy / height * freq_y)
        b = 0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy) / (width + height) * 4)
        gs_img = np.stack([r, g, b], axis=-1)
        gs_uint8 = np.clip(gs_img * 255.0, 0, 255).astype(np.uint8)

        # Generate a realistic rolling-shutter scanline displacement
        # RS distortion is scanline (row y) dependent due to sequential readout
        max_shift = np.random.uniform(6.0, 15.0) * np.random.choice([-1.0, 1.0])
        flow_x = (yy / height) * max_shift
        flow_y = 0.2 * np.sin(2 * np.pi * yy / height) * max_shift
        flow = np.stack([flow_x, flow_y], axis=-1)  # (H, W, 2)

        # Distort GS to get RS condition: RS(x, y) = GS(x - flow_x, y - flow_y)
        sample_x = np.clip(xx - flow_x, 0, width - 1).astype(np.int32)
        sample_y = np.clip(yy - flow_y, 0, height - 1).astype(np.int32)
        rs_uint8 = gs_uint8[sample_y, sample_x]

        file_path = Path(output_dir) / f"sample_{i:04d}.npz"
        np.savez_compressed(file_path, condition=rs_uint8, gs=gs_uint8, out_flow=flow)

    print(f"Successfully generated {num_samples} samples in {output_dir}.")

if __name__ == "__main__":
    out_dir = r"C:\Users\udayb\.gemini\antigravity\scratch\rolling_shutter_correction\data\synthetic_sample"
    generate_synthetic_rsc_dataset(out_dir, num_samples=10)
