# What Can IRT Person-Fit Statistics Reveal About SFT-Stage Benchmark Contamination?

Code for the paper: "What Can IRT Person-Fit Statistics Reveal About SFT-Stage Benchmark Contamination?" (EMNLP 2026 submission).

Anonymous review: [https://anonymous.4open.science/r/irt_personfit_contam-6004/](https://anonymous.4open.science/r/irt_personfit_contam-6004/)

## Overview

Benchmark contamination in LLMs undermines evaluation validity, yet most detection methods require privileged access to training data, model internals, or output probabilities. We present the first controlled characterization of IRT person-fit statistics for detecting SFT-stage benchmark contamination, requiring only binary response patterns and calibrated item parameters.

Using lz* in a matched-difference design (Δlz*) across three 7–8B model families (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B; n=10 clean seeds per family) on MMLU (14,042 items) and ARC-Challenge (295 items), we identify three boundaries: (1) θ-anchoring amplifies detection by 2.4× while reducing coefficient of variation from 60% to 19%; (2) a detection–diagnosis tradeoff where simpler statistics yield stronger detection but lack lz*'s diagnostic capabilities; and (3) ability-dependent non-monotonic dose-response under free-θ estimation, driven by θ-shift absorption.

## Repository Structure

```
├── src/                    # Core IRT / person-fit module
│   └── person_fit.py       # 4PL IRT model, lz, lz*, theta MLE, outfit/infit
├── qwen/                   # Qwen-2.5-7B-Instruct experiments
│   ├── prepare_data.py     # Generate training data (clean + contaminated)
│   ├── train.py            # QLoRA SFT training
│   ├── eval_lz.py          # MMLU/ARC evaluation + lz* computation
│   ├── compute_delta.py    # Delta lz* and summary statistics
│   ├── compute_cohens_d.py # Cohen's d with bootstrap CIs
│   ├── compute_fixed_theta.py          # Fixed-theta analysis
│   ├── compute_contam5_cohens_d.py     # 5% contamination Cohen's d
│   ├── gen_contam_5.py / gen_contam_5_s2.py / gen_contam_5_s3.py  # Contamination data gen
│   ├── gen_contam_25.py                # 25% contamination data gen
│   ├── theta_fixed_lzstar.py           # Theta-anchored lz*
│   ├── theta_fixed_recompute.py        # Theta-fixed recomputation
│   ├── unified_lz_recompute.py         # Unified lz recomputation
│   └── recompute_theta_fixed_n10.py    # N=10 theta-fixed recomputation
├── llama/                  # Llama-3.1-8B-Instruct experiments
│   ├── prepare_data.py / prepare_data_5pct.py
│   ├── train.py / eval_lz.py
│   ├── compute_cohens_d.py
│   └── theta_fixed_recompute.py
├── mistral/                # Mistral-7B-Instruct-v0.3 experiments
│   ├── prepare_data.py / train.py / eval_lz.py
│   └── theta_fixed_n10_recompute.py
├── scripts/                # Figure generation and analysis
│   ├── gen_appendix_4pl_ablation.py    # 3PL vs 4PL power analysis
│   ├── gen_fig_heatmap.py              # Cohen's d heatmap
│   └── gen_fig_theta_scatter.py        # Theta scatter plot
├── figures/                # Paper figure generation
│   ├── gen_fig1.py                     # Response pattern visualization
│   └── paper/gen_fig_theta_scatter.py  # Theta scatter (paper version)
└── requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Generate training data with controlled contamination levels:

```bash
cd qwen
python prepare_data.py
```

Creates training sets for conditions: `clean`, `contam_5`, `contam_15`, `contam_25`, `contam_50` (varying percentages of MMLU items mixed into Alpaca SFT data).

### 2. Training (QLoRA SFT)

```bash
python train.py --condition contam_15 --gpu 0
```

**Conditions:** `clean_s1` through `clean_s10`, `contam_5`, `contam_15`, `contam_25`, `contam_50`

### 3. Evaluation (lz* computation)

```bash
python eval_lz.py --condition contam_15 --gpu 0
```

Evaluates the fine-tuned adapter on MMLU and ARC-Challenge using 3-shot prompting and computes:
- θ MLE (ability estimate)
- lz* (Snijders 2001 corrected person-fit statistic)
- Accuracy (overall, seen items, unseen items)

### 4. Statistical Analysis

```bash
python compute_delta.py          # Delta lz* and summary statistics
python unified_lz_recompute.py   # Unified recomputation with theta-fixed analysis
python compute_cohens_d.py       # Cohen's d with bootstrap CIs
```

### 5. Cross-Model Replication

The `llama/` and `mistral/` directories contain analogous scripts for Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3.

## Configuration

Update the `BASE_DIR` and model path constants in each script to point to your local directories. Models used:
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Data

- MMLU items from `cais/mmlu` (HuggingFace datasets)
- ARC-Challenge items from `allenai/ai2_arc` (HuggingFace datasets)
- 4PL IRT parameters from PSN-IRT item bank
- Alpaca SFT data from `tatsu-lab/alpaca`

## License

[To be added after review]
