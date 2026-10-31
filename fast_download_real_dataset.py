"""
Fast, robust multi-threaded downloader for RS-Diffusion Real Dataset:
- 1,000 Official Real Test Samples
- 2,500 Official Real Training Samples (distributed across all 27 scenes)
- Official Pretrained Weights (RS_Real.pt ~1.28 GB)
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE_URL = "https://huggingface.co/datasets/Yzl-code/RS-Diffusion/resolve/main/"
ROOT_DIR = Path(r"D:\RS-Real")
INDEX_PATH = ROOT_DIR / "index.json"

def verify_npz(file_path):
    try:
        if file_path.stat().st_size < 10000:
            return False
        with np.load(file_path, allow_pickle=False) as d:
            return {"condition", "gs", "out_flow"}.issubset(d.files)
    except Exception:
        return False

def download_file(rel_path, target_root):
    out_path = target_root / rel_path
    if out_path.exists() and verify_npz(out_path):
        return True, "exists"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + rel_path
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            # Write to temp file then rename for atomic completion
            temp_path = out_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(data)
            temp_path.replace(out_path)

            if verify_npz(out_path):
                return True, "downloaded"
            else:
                if out_path.exists():
                    out_path.unlink()
                return False, "corrupt"
        except Exception as e:
            if attempt == 3:
                temp_path = out_path.with_suffix(".tmp")
                if temp_path.exists():
                    temp_path.unlink()
                if out_path.exists():
                    out_path.unlink()
                return False, str(e)
            time.sleep(1 + attempt)

def download_checkpoint():
    ckpt_path = ROOT_DIR / "checkpoint" / "RS_Real.pt"
    if ckpt_path.exists() and ckpt_path.stat().st_size > 1200000000:
        print(f"[Weights] RS_Real.pt already exists ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        return

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ckpt_path.with_suffix(".tmp")
    url = BASE_URL + "RS_Real.pt"
    print(f"\n[Weights] Downloading official RS_Real.pt (~1.28 GB)...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 4 * 1024 * 1024  # 4 MB chunks
        t0 = time.time()
        with open(temp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - t0, 0.001)
                speed = (downloaded / 1e6) / elapsed
                pct = (downloaded / total_size) * 100 if total_size else 0
                print(f"\r  RS_Real.pt: {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB ({pct:.1f}%) @ {speed:.1f} MB/s", end="", flush=True)

    temp_path.replace(ckpt_path)
    print(f"\n[Weights] RS_Real.pt download complete! ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

def main():
    print("=" * 65)
    print("  Official RS-Diffusion Real-World Dataset & Model Downloader")
    print("=" * 65)

    with open(INDEX_PATH, "r") as f:
        idx = json.load(f)

    test_files = idx["test"]
    train_files = idx["train"]
    print(f"Index loaded: {len(train_files)} total train files, {len(test_files)} test files.")

    # Group train files by scene
    scenes = {}
    for f in train_files:
        sc = f.split("/")[1]
        scenes.setdefault(sc, []).append(f)

    # Sample ~95 samples per scene across all 27 scenes -> ~2,565 real train samples
    SAMPLES_PER_SCENE = 95
    selected_train = []
    for sc, sc_files in scenes.items():
        step = max(1, len(sc_files) // SAMPLES_PER_SCENE)
        selected_train.extend(sc_files[::step][:SAMPLES_PER_SCENE])

    print(f"Selected {len(selected_train)} train samples from {len(scenes)} scenes.")
    print(f"Selected {len(test_files)} test samples.")

    target_files = test_files + selected_train
    print(f"Total dataset files to download/verify: {len(target_files)}")

    MAX_WORKERS = 16
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    exist_count = 0

    print(f"\nStarting parallel download with {MAX_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, rel_f, ROOT_DIR): rel_f for rel_f in target_files}
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()
            if success:
                if msg == "exists":
                    exist_count += 1
                else:
                    ok_count += 1
            else:
                fail_count += 1

            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                speed = i / max(elapsed, 0.001)
                pct = (i / total) * 100
                print(f"Progress: [{i}/{total}] ({pct:.1f}%) | Downloaded: {ok_count} | Cached: {exist_count} | Failed: {fail_count} | {speed:.1f} files/s", flush=True)

    print(f"\nDataset download finished in {(time.time() - t0)/60:.2f} min!")
    print(f"  Newly downloaded : {ok_count}")
    print(f"  Already existed  : {exist_count}")
    print(f"  Failures         : {fail_count}")

    # Now download the 1.28 GB pretrained weights
    download_checkpoint()
    print("\nAll downloads finished successfully!")

if __name__ == "__main__":
    main()
