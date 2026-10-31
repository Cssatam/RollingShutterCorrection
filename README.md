# Real-Time Single-Image Rolling Shutter Correction via Lightweight Coarse-to-Fine Optical Flow

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Device](https://img.shields.io/badge/Hardware-Edge%20%2F%20Embedded%20Ready-success.svg)]()

> **Research Paper:** *A Lightweight CNN-Based Approach to Real-Time Rolling Shutter Correction on Resource-Constrained Devices*  
> **Author:** Chinmay Satam  
> **Advisors:** Dr. Vidyullata Devmane and Dr. Bhavesh Patel  
> **Affiliation:** Honours with Research, Foundation of Research & Innovation  

---

## 🚀 Key Highlights

- **Ultra-Lightweight Architecture**: Only **1,122,019 parameters (~1.12M)**, achieving a **71.2× parameter reduction** and **74.6× storage reduction** (4.28 MB vs. 319.5 MB) compared to heavy generative baselines (RS-Diffusion, AAAI 2025).
- **True Real-Time Performance**: Evaluates at **284.9 FPS (3.51 ms per frame)** on an RTX 3080 Ti Laptop GPU — **53.0× faster** than 8-step diffusion models and **10.5× faster** than 1-step diffusion.
- **Superior Real-World Quality**: Delivers **17.55 dB PSNR** and **0.5786 SSIM** on 999 official real-world camera test captures (+1.66 dB gain over distorted input). Interior field-of-view reaches **19.02 dB**.
- **Lossless INT8 Quantization**: Symmetric 8-bit weight quantization (W8) compresses the model to **1.07 MB (4.0× reduction)** with **zero loss in PSNR (+0.00 ± 0.03 dB)**.
- **Microcontroller-Ready Full Quantization (W8A8)**: 8-bit weights and activations run with only a **−0.01 dB penalty**, ready for edge integer NPUs.

---

## 📊 Benchmark Summary

Evaluated on 50 real-world dual-sensor camera samples:

| Model Architecture | Parameters | Storage | Latency | Throughput | PSNR | SSIM | EPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Correction (RS Input)** | — | — | — | — | 15.53 dB | 0.4930 | — |
| **RS-Diffusion (8-step)** | 79.9M | 319.5 MB | 186.0 ms | 5.38 FPS | 26.16 dB | 0.8600 | 0.662 px |
| **RS-Diffusion (1-step)** | 79.9M | 319.5 MB | 37.0 ms | 27.06 FPS | 25.77 dB | 0.8560 | 0.713 px |
| **Proposed CNN (fp32)** | **1.12M** | **4.28 MB** | **3.51 ms** | **284.9 FPS** | **18.74 dB** | **0.6060** | **4.000 px** |
| **Proposed CNN (W8 int8)** | **1.12M** | **1.07 MB** | **3.95 ms** | **253.2 FPS** | **18.75 dB** | **0.6060** | **3.996 px** |
| **Proposed CNN (W8A8)** | **1.12M** | **1.07 MB** | **7.62 ms** | **131.2 FPS** | **18.74 dB** | **0.6050** | **3.996 px** |
| **Proposed CNN (W4 4-bit)** | **1.12M** | **0.54 MB** | **4.09 ms** | **244.5 FPS** | **18.15 dB** | **0.5980** | **4.422 px** |

---

## 📂 Repository Structure

```
css/
├── .gitignore                                 # Production Git ignore rules (excludes data/ and .pt files)
├── requirements.txt                           # Minimal pip dependencies
├── README.md                                  # Project overview and reproduction guide
│
├── 📖 In-Depth Research Documentation:
│   ├── MASTER_RESEARCH_REPORT_AND_FINDINGS.md # Complete research paper findings & mathematical proofs
│   ├── PROJECT_RESULTS_AND_FINDINGS_REPORT.md # Full architecture specifications & training logs
│   ├── CNN_COMPRESSION_BENCHMARK_REPORT.md    # Dedicated quantization & pruning study
│   └── GIT_REPOSITORY_STRUCTURE_AND_GUIDE.md  # Detailed Git repository manifest guide
│
├── 🧠 Core Pipeline & Training:
│   ├── train_real_model.py                    # Real-world training with direct GT optical flow
│   ├── train_model.py                         # Synthetic v3 trainer with 4-ch scanline proxy
│   ├── evaluate_real_test.py                  # 999-sample official benchmark evaluation script
│   ├── run_compression_benchmark.py           # Automated W8/W8A8/W4 quantization & pruning benchmark
│   └── test_pipeline.py                       # Pipeline sanity check and latency probe
│
├── 📥 Dataset & Synthetic Generation:
│   ├── download_rs_real.py                    # Multi-threaded Hugging Face dataset downloader
│   ├── fast_download_real_dataset.py          # Parallel chunked downloader with validation
│   ├── generate_synthetic_sample.py           # Procedural synthetic rolling shutter generator
│   └── generate_1000_samples.py               # Scaled procedural dataset synthesizer
│
├── 🎨 Visualization & Edge Rectification:
│   ├── visualize_real_results.py              # 6-column qualitative dewarping visualizer
│   ├── generate_detached_border_viz.py        # Detached black zeros vs border stretching visualizer
│   └── visualize_v3.py                        # 5-column synthetic visualizer
│
├── 📓 Interactive Research Notebooks:
│   ├── Rolling_Shutter_Correction_Complete_Project_REVISED.ipynb
│   └── Step3_RS_Diffusion_Ablation_Study.ipynb
│
└── 📊 Numerical Results & Metric Logs:
    ├── checkpoints_real/benchmark_test_results.json
    ├── checkpoints_real/cnn_compression_benchmark_results.json
    └── checkpoints_real/history_real.json
```

---

## 🛠️ Quick Start & Reproduction

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/Cssatam/RollingShutterCorrection.git
cd RollingShutterCorrection
pip install -r requirements.txt
```

### 2. Instant Demo (1-Click Test)
Run the self-contained demo on sample rolling shutter distortion:
```bash
python demo.py
```
*(Outputs `demo_result.png` showing the input vs corrected comparison).*

### 3. Download Pretrained Checkpoint & Test Subset
Download sample test pairs from the official RS-Real dataset:
```bash
# Download 50 real test samples for quick benchmark
python download_rs_real.py 50
```

### 3. Run Compression Benchmark (W8, W8A8, W4, Pruning)
```bash
python run_compression_benchmark.py
```

### 4. Train from Scratch on Real Camera Captures
```bash
python train_real_model.py --epochs 30 --batch_size 8 --lr 2.5e-4 --device cuda
```

### 5. Evaluate Full Official 999-Sample Test Benchmark
```bash
python evaluate_real_test.py
```

---

## 📑 Citation & Contact
If you use this codebase or model architecture in your research, please cite:
```bibtex
@article{satam2026lightweight,
  title={A Lightweight CNN-Based Approach to Real-Time Rolling Shutter Correction on Resource-Constrained Devices},
  author={Satam, Chinmay and Devmane, Vidyullata and Patel, Bhavesh},
  year={2026}
}
```
