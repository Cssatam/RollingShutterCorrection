"""
Download real RS-Diffusion dataset from Hugging Face:
- 1,000 Real Test Samples (test/npz)
- 3,000+ Real Training Samples across all 27 scenes (train/npz)
- Official Pretrained Weights: RS_Real.pt (AAAI 2025)
Using multi-threaded downloading with huggingface_hub / urllib.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
import numpy as np

REPO_ID = "Yzl-code/RS-Diffusion"
TARGET_DIR = Path(r"D:\RS-Real")

def verify_npz(path):
    try:
        with np.load(path) as d:
            return {"condition", "gs", "out_flow"}.issubset(d.files)
    except Exception:
        return False

def download_file(filename, target_root):
    out_file = target_root / filename
    if out_file.exists() and out_file.stat().st_size > 50000:
        if verify_npz(out_file):
            return "exists"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(target_root),
            local_dir_use_symlinks=False
        )
        if verify_npz(out_file):
            return "downloaded"
        else:
            if out_file.exists():
                out_file.unlink()
            return "corrupt"
    except Exception as e:
        return f"error: {e}"

def main():
    print("=" * 60)
    print("Downloading Official RS-Diffusion Real-World Dataset & Weights")
    print(f"Hugging Face Repo: {REPO_ID}")
    print(f"Target Root      : {TARGET_DIR}")
    print("=" * 60)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download pretrained checkpoint RS_Real.pt
    ckpt_file = TARGET_DIR / "checkpoint" / "RS_Real.pt"
    if not ckpt_file.exists() or ckpt_file.stat().st_size < 10000000:
        print("\n[1/3] Downloading official pretrained checkpoint RS_Real.pt...")
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename="RS_Real.pt",
            local_dir=str(TARGET_DIR / "checkpoint"),
            local_dir_use_symlinks=False
        )
        print("Pretrained checkpoint downloaded successfully!")
    else:
        print("\n[1/3] Pretrained checkpoint RS_Real.pt already exists.")

    api = HfApi()
    print("\n[2/3] Fetching file index from Hugging Face...")
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    test_files = sorted([f for f in all_files if f.startswith("test/npz/") and f.endswith(".npz")])
    train_files = sorted([f for f in all_files if f.startswith("train/") and f.endswith(".npz")])

    print(f"Found {len(test_files)} real test samples.")
    print(f"Found {len(train_files)} real training samples across scenes.")

    # 2. Select balanced training files across all 27 scenes
    # Group train files by scene
    scenes = {}
    for f in train_files:
        # e.g. train/scene01/sample_0000.npz
        parts = f.split("/")
        if len(parts) >= 2:
            sc = parts[1]
            scenes.setdefault(sc, []).append(f)

    print(f"Total scenes: {len(scenes)}")

    # Pick up to 120 samples per scene -> ~3,200 real training pairs
    SAMPLES_PER_SCENE = 120
    selected_train = []
    for sc, sc_files in scenes.items():
        step = max(1, len(sc_files) // SAMPLES_PER_SCENE)
        picked = sc_files[::step][:SAMPLES_PER_SCENE]
        selected_train.extend(picked)

    print(f"Selected {len(selected_train)} diverse real training samples across all {len(scenes)} scenes.")
    print(f"Selected {len(test_files)} full real test benchmark samples.")

    files_to_download = test_files + selected_train
    print(f"\nTotal files to download: {len(files_to_download)}")

    # 3. Download with thread pool
    MAX_WORKERS = 16
    print(f"\n[3/3] Starting multi-threaded download ({MAX_WORKERS} workers)...")
    t0 = time.time()
    downloaded = 0
    existed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, f, TARGET_DIR): f for f in files_to_download}
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res == "downloaded":
                downloaded += 1
            elif res == "exists":
                existed += 1
            else:
                failed += 1

            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                speed = i / max(elapsed, 0.001)
                print(f"Progress: [{i}/{total}] ({(i/total)*100:.1f}%) | "
                      f"New: {downloaded} | Existed: {existed} | Failed: {failed} | "
                      f"{speed:.1f} files/s ({elapsed:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print(f"Download complete in {(time.time() - t0) / 60:.2f} minutes!")
    print(f"  Downloaded : {downloaded}")
    print(f"  Pre-existed: {existed}")
    print(f"  Failed     : {failed}")
    print(f"  Target dir : {TARGET_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
