import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HF_DATASET_API = "https://huggingface.co/api/datasets/Yzl-code/RS-Diffusion"
BASE_RESOLVE_URL = "https://huggingface.co/datasets/Yzl-code/RS-Diffusion/resolve/main/"

def get_train_file_list():
    req = urllib.request.Request(HF_DATASET_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    files = [s["rfilename"] for s in data.get("siblings", []) if s["rfilename"].startswith("train/") and s["rfilename"].endswith(".npz")]
    return sorted(files)

def download_single_file(rel_path, target_root):
    out_path = Path(target_root) / rel_path
    if out_path.exists() and out_path.stat().st_size > 10000:
        return True, "already exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_RESOLVE_URL + rel_path
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            # Validate npz
            with np.load(out_path) as npz:
                if not {"condition", "gs"}.issubset(npz.files):
                    raise ValueError("corrupted keys")
            return True, "downloaded"
        except Exception as e:
            if attempt == 2:
                if out_path.exists():
                    out_path.unlink()
                return False, str(e)
            time.sleep(1)

def download_dataset(target_dir=r"D:\RS-Real", num_samples=300, max_workers=8):
    target_path = Path(target_dir)
    train_dir = target_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {train_dir}")
    print("Fetching file list from Hugging Face...")
    all_files = get_train_file_list()
    print(f"Total available training files: {len(all_files)}")

    # Select representative samples evenly spread across multiple scenes
    # e.g., scene01, scene02, scene03... to get diversity of motion
    selected_files = []
    # group by scene
    scenes = {}
    for f in all_files:
        scene = f.split("/")[1]
        scenes.setdefault(scene, []).append(f)

    samples_per_scene = max(1, num_samples // len(scenes))
    for scene, s_files in scenes.items():
        step = max(1, len(s_files) // samples_per_scene)
        selected_files.extend(s_files[::step][:samples_per_scene])
        if len(selected_files) >= num_samples:
            break
    selected_files = selected_files[:num_samples]

    print(f"Downloading {len(selected_files)} samples across {len(scenes)} scenes with {max_workers} threads...")
    t0 = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_file, f, target_path): f for f in selected_files}
        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()
            if success:
                success_count += 1
            if i % 25 == 0 or i == len(selected_files):
                pct = (i / len(selected_files)) * 100
                elapsed = time.time() - t0
                print(f"Progress: [{i}/{len(selected_files)}] ({pct:.1f}%) - {success_count} valid files - {elapsed:.1f}s")

    print(f"Finished! Successfully downloaded {success_count} samples into {train_dir}")
    return train_dir

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    download_dataset(num_samples=n)
