# Git Repository Setup & Manifest Guide
## Rolling Shutter Correction (RSC) Project

This document outlines **what to include**, **what to exclude**, and **how to structure and publish** this research project to a clean, professional GitHub repository.

---

## 1. Summary: What Goes into Git vs. What Stays Local

```
📂 css/ (Repository Root)
│
├── 🟢 INCLUDED IN GIT (Code, Notebooks, Configs, Benchmarks, Reports)
│   ├── .gitignore                                 <- Specifies ignored files
│   ├── requirements.txt                           <- Python environment dependencies
│   ├── README.md                                  <- Main GitHub landing page & reproduction instructions
│   ├── MASTER_RESEARCH_REPORT_AND_FINDINGS.md     <- All-in-one comprehensive paper report
│   ├── PROJECT_RESULTS_AND_FINDINGS_REPORT.md     <- Complete architecture & methodology specs
│   ├── CNN_COMPRESSION_BENCHMARK_REPORT.md        <- Quantization & pruning benchmark report
│   ├── GIT_REPOSITORY_STRUCTURE_AND_GUIDE.md      <- This guide
│   │
│   ├── 📁 Core Model & Training Scripts:
│   │   ├── train_real_model.py                    <- Real-world trainer with direct GT flow supervision
│   │   ├── train_model.py                         <- Synthetic/v3 multi-loss trainer
│   │   ├── evaluate_real_test.py                  <- 999-sample benchmark evaluation script
│   │   ├── run_compression_benchmark.py           <- Quantization (W8/W8A8/W4) & pruning benchmark
│   │   ├── test_pipeline.py                       <- Pipeline verification and latency sanity check
│   │   ├── update_nb.py                           <- Notebook config update helper
│   │
│   ├── 📁 Dataset Utilities (Code Only):
│   │   ├── download_rs_real.py                    <- Multi-threaded downloader from Hugging Face
│   │   ├── fast_download_real_dataset.py          <- Fast resilient chunked downloader
│   │   ├── download_full_real_dataset.py          <- Full Hugging Face downloader
│   │   ├── generate_synthetic_sample.py           <- Procedural synthetic RS/GS generator
│   │   ├── generate_1000_samples.py               <- Scaled synthetic dataset generator
│   │
│   ├── 📁 Visualization & Analysis Tools:
│   │   ├── visualize_real_results.py              <- 6-column real camera visualizer
│   │   ├── generate_detached_border_viz.py        <- Border stretching vs detached zeros comparison
│   │   ├── visualize_v3.py                        <- 5-column synthetic visualizer
│   │   ├── visualize_predictions.py               <- Baseline 4-column visualizer
│   │
│   ├── 📁 Jupyter Research Notebooks:
│   │   ├── Rolling_Shutter_Correction_Complete_Project_REVISED.ipynb
│   │   ├── Step3_RS_Diffusion_Ablation_Study.ipynb
│   │
│   └── 📁 Metric Logs & Results (Small JSON files, < 20 KB):
│       ├── checkpoints/history.json               <- 40-epoch loss history (synthetic)
│       ├── checkpoints/test_results.json          <- Synthetic test metrics
│       ├── checkpoints_real/history_real.json     <- 30-epoch loss history (real-world)
│       ├── checkpoints_real/benchmark_test_results.json <- 999-sample official benchmark metrics
│       └── checkpoints_real/cnn_compression_benchmark_results.json <- Compression benchmark outputs
│
└── 🔴 EXCLUDED FROM GIT (Ignore via .gitignore)
    ├── data/                                      <- Raw dataset (1,000+ .npz files, several GBs)
    ├── data/RS-Real/                              <- Real camera training images (.npz, .jpg)
    ├── data/synthetic_sample/                     <- Synthetic .npz pairs
    ├── RS-Diffusion/                              <- Upstream cloned submodule repo (has its own .git)
    ├── __pycache__/                               <- Compiled Python bytecode files (*.pyc)
    ├── .ipynb_checkpoints/                        <- Jupyter automatic checkpoint caches
    ├── checkpoints/*.pt                           <- Model weights (> 10 MB each, share via Releases)
    ├── checkpoints_real/*.pt                      <- Model weights (best_model_real.pt, etc.)
    └── *.tmp, *.npy, *.log                        <- Temporary download buffers and traces
```

---

## 2. Detailed File Manifest

### 2.1 Essential Source Files (Track in Git)

| File / Folder | Role in Project | Size | Reason to Include |
| :--- | :--- | :---: | :--- |
| [`.gitignore`](file:///D:/c%20data/Desktop/css/.gitignore) | Git exclusion rules | < 2 KB | Prevents accidental upload of datasets and multi-gigabyte files. |
| [`requirements.txt`](file:///D:/c%20data/Desktop/css/requirements.txt) | Dependency list | < 1 KB | Allows users to `pip install -r requirements.txt`. |
| [`README.md`](file:///D:/c%20data/Desktop/css/README.md) | Project Overview | ~2 KB | Landing page for GitHub visitors and reviewers. |
| [`MASTER_RESEARCH_REPORT_AND_FINDINGS.md`](file:///D:/c%20data/Desktop/css/MASTER_RESEARCH_REPORT_AND_FINDINGS.md) | Master Paper Reference | ~31 KB | Full findings, literature gap, mathematical formulas, and comparative tables. |
| [`PROJECT_RESULTS_AND_FINDINGS_REPORT.md`](file:///D:/c%20data/Desktop/css/PROJECT_RESULTS_AND_FINDINGS_REPORT.md) | Architecture Specs | ~28 KB | Input/output specifications, parameter counts, and training logs. |
| [`CNN_COMPRESSION_BENCHMARK_REPORT.md`](file:///D:/c%20data/Desktop/css/CNN_COMPRESSION_BENCHMARK_REPORT.md) | Compression Paper Study | ~10 KB | Quantization (W8, W8A8, W4) and pruning study matching RS-Diffusion. |
| [`train_real_model.py`](file:///D:/c%20data/Desktop/css/train_real_model.py) | Main Real Trainer | ~22 KB | Standalone training script with direct ground-truth flow supervision. |
| [`train_model.py`](file:///D:/c%20data/Desktop/css/train_model.py) | Synthetic Trainer | ~27 KB | Multi-loss synthetic trainer with 4-channel scanline coordinate input. |
| [`evaluate_real_test.py`](file:///D:/c%20data/Desktop/css/evaluate_real_test.py) | Benchmark Evaluator | ~7 KB | Computes 999-sample PSNR/SSIM/EPE and multi-crop border analysis. |
| [`run_compression_benchmark.py`](file:///D:/c%20data/Desktop/css/run_compression_benchmark.py) | Compression Tester | ~15 KB | Automated W8, W8A8, W4, pruning and latency benchmarking script. |
| [`test_pipeline.py`](file:///D:/c%20data/Desktop/css/test_pipeline.py) | Sanity Verification | ~9 KB | Quick end-to-end forward/backward sanity check and latency probe. |
| [`download_rs_real.py`](file:///D:/c%20data/Desktop/css/download_rs_real.py) | Dataset Downloader | ~4 KB | Reproducible fetch script to download RS-Real from Hugging Face. |
| [`fast_download_real_dataset.py`](file:///D:/c%20data/Desktop/css/fast_download_real_dataset.py) | Parallel Downloader | ~6 KB | Resilient multi-threaded downloader with atomic write operations. |
| [`visualize_real_results.py`](file:///D:/c%20data/Desktop/css/visualize_real_results.py) | Visualization Tool | ~5 KB | Generates side-by-side 6-column qualitative comparison plots. |
| [`generate_detached_border_viz.py`](file:///D:/c%20data/Desktop/css/generate_detached_border_viz.py) | Border Analysis Tool | ~5 KB | Produces detached zero-padded vs stretched border comparisons. |
| [`checkpoints_real/*.json`](file:///D:/c%20data/Desktop/css/checkpoints_real/benchmark_test_results.json) | Benchmark Results | < 20 KB | Verifiable empirical numerical test results. |

---

### 2.2 Excluded Folders & Files (Do NOT Commit)

| Path / Pattern | Why Exclude | Alternative Delivery Method |
| :--- | :--- | :--- |
| `data/` | Huge dataset folder (> 1,000 files, multiple GBs). Git is not designed to store binary training data. | Provided via `download_rs_real.py` script automatically. |
| `RS-Diffusion/` | Cloned third-party repository with its own `.git` history. Committing it causes nested submodule conflicts. | Clone separately via `git clone https://github.com/lhaippp/RS-Diffusion`. |
| `checkpoints/*.pt` | Trained model weights (~12.9 MB each). Exceeds GitHub recommended limits and inflates git repo size. | Upload to **GitHub Releases** (e.g. `v1.0.0-weights`) or Hugging Face Models. |
| `__pycache__/` | Ephemeral compiled Python bytecode. Specific to machine and Python version. | Ignored via `.gitignore`. |
| `*.tmp`, `*.npy` | Intermediate download caches and raw flow numpy buffers. | Regenerated on demand by evaluation scripts. |

---

## 3. Step-by-Step Commands to Initialize and Push Git Repo

Open PowerShell or terminal in `D:\c data\Desktop\css`:

```powershell
# 1. Initialize Git in the project root
git init

# 2. Check that .gitignore is working (ensure data/ and .pt files are ignored)
git status

# 3. Add all valid source code, reports, configs, and notebooks
git add .gitignore requirements.txt README.md
git add *.py *.ipynb *.md
git add checkpoints/*.json checkpoints_real/*.json

# 4. Commit the project
git commit -m "Initial commit: Lightweight CNN for Real-Time Rolling Shutter Correction (Code, Benchmarks, Reports)"

# 5. Create a new repository on GitHub (e.g., username/lightweight-rsc)
# Link the remote repository and push to main branch:
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

---

## 4. How to Distribute Pretrained Model Weights

Because binary checkpoint files (`best_model_real.pt`, 12.94 MB) should not clutter git commit history, use one of the two standard practices:

### Option A: GitHub Releases (Recommended & Free)
1. Push your code to GitHub.
2. In your GitHub repository, click **Releases** $\to$ **Create a new release** (tag: `v1.0.0`).
3. Attach `best_model_real.pt` and `best_model.pt` as binary release assets.
4. Add the direct download link in `README.md`.

### Option B: Git LFS (Large File Storage)
If you specifically want checkpoint tracking directly in git:
```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add checkpoints_real/best_model_real.pt
git commit -m "Add pretrained real model weights via Git LFS"
git push origin main
```

---

## 5. Quick Reproduction Guide for Repo Cloners

Anyone cloning your GitHub repository can replicate your results in 3 commands:

```bash
# 1. Clone the repository
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the 50 test samples and evaluate benchmark
python download_rs_real.py 50
python run_compression_benchmark.py
```
