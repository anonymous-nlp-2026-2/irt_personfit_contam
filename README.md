# What Can IRT Person-Fit Statistics Reveal About SFT-Stage Benchmark Contamination?

Code for the paper: "What Can IRT Person-Fit Statistics Reveal About SFT-Stage Benchmark Contamination?" (EMNLP 2026 submission).

## Overview

This repository provides the first controlled characterization of IRT person-fit statistics for detecting supervised fine-tuning (SFT) benchmark contamination in Large Language Models. Benchmark contamination undermines evaluation validity, yet most detection methods require privileged access to training data, model internals, or output probabilities. We systematically characterize the behavioral boundaries of IRT person-fit statistics in LLM benchmark contamination detection, requiring only binary response patterns and calibrated four-parameter logistic item parameters; diagnostic theta-anchoring requires a clean baseline. Using the lz* statistic in a matched-difference design across three 7-8B model families (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B; n=10 clean seeds per family) on MMLU (14,042 items) and ARC-Challenge (295 items), we identify three boundaries: (1) model-specific theta-shift dynamics, where contamination-induced ability inflation absorbs the person-fit signal and theta-anchoring amplifies detection by 2.4x; (2) a detection-diagnosis tradeoff where simpler statistics yield stronger detection but lack lz*'s diagnostic capabilities (seen/unseen decomposition, mechanistic theta-shift analysis); and (3, exploratory) ability-dependent non-monotonic dose-response under free-theta estimation, driven by theta-shift absorption, which theta-anchored analysis resolves into stable positive dose-response. These findings establish when and why binary response patterns carry contamination information beyond aggregate accuracy.

## Repository Structure

```
.
├── src/                        # Core IRT / person-fit module
│   └── person_fit.py           # 4PL IRT model, lz, lz*, theta MLE, outfit/infit
├── qwen/                       # Qwen-2.5-7B-Instruct experiments
│   ├── prepare_data.py         # Generate training data (clean + contaminated)
│   ├── train.py                # QLoRA SFT training
│   ├── eval_lz.py              # MMLU evaluation + lz* computation
│   ├── compute_cohens_d.py     # Cohen's d with bootstrap CIs
│   ├── compute_delta.py        # Delta lz* summary statistics
│   ├── compute_fixed_theta.py  # Fixed-theta analysis
│   ├── unified_lz_recompute.py # Unified recomputation with theta-fixed
│   ├── theta_fixed_lzstar.py   # Theta-fixed lz* computation
│   ├── theta_fixed_recompute.py
│   ├── recompute_theta_fixed_n10.py
│   ├── compute_contam5_cohens_d.py
│   ├── gen_contam_5.py         # Generate 5% contamination data
│   ├── gen_contam_5_s2.py
│   ├── gen_contam_5_s3.py
│   └── gen_contam_25.py        # Generate 25% contamination data
├── llama/                      # Llama-3.1-8B-Instruct experiments
│   ├── prepare_data.py
│   ├── prepare_data_5pct.py
│   ├── train.py
│   ├── eval_lz.py
│   ├── theta_fixed_recompute.py
│   └── compute_cohens_d.py
├── mistral/                    # Mistral-7B-Instruct-v0.3 experiments
│   ├── prepare_data.py
│   ├── train.py
│   ├── eval_lz.py
│   └── theta_fixed_n10_recompute.py
├── scripts/                    # Figure generation and analysis scripts
│   ├── gen_appendix_4pl_ablation.py  # 3PL vs 4PL power analysis
│   ├── gen_fig_heatmap.py            # Cohen's d heatmap
│   └── gen_fig_theta_scatter.py      # Theta scatter plot
├── figures/                    # Paper figure assets and generation
│   ├── gen_fig1.py             # Response pattern visualization (Fig. 1)
│   └── paper/                  # Generated figures (PDF + PNG)
│       ├── fig_1_composite.pdf
│       ├── fig_1_response_pattern.pdf
│       ├── fig_2_power_function.pdf
│       ├── fig_3_ctrl_contam.pdf
│       ├── fig_4_model_heatmap.pdf
│       ├── fig_dose_response.pdf
│       ├── fig_framework.pdf
│       ├── fig_grpo.pdf
│       ├── fig_heatmap.pdf
│       ├── fig_theta_scatter.pdf
│       ├── fig_method_pipeline.pdf
│       └── gen_fig_theta_scatter.py
├── paper/                      # LaTeX source
│   ├── main.tex
│   ├── main.bbl
│   ├── main.pdf
│   ├── references.bib
│   ├── acl.sty
│   ├── acl_natbib.bst
│   └── sections/               # Paper sections
│       ├── abstract.tex
│       ├── introduction.tex
│       ├── background.tex
│       ├── related_work.tex
│       ├── method.tex
│       ├── results.tex
│       ├── discussion.tex
│       ├── conclusion.tex
│       ├── limitations.tex
│       ├── appendix.tex
│       ├── experiments.tex
│       ├── sec_setup.tex
│       ├── sec_simulation.tex
│       ├── sec_ctrl_contam.tex
│       └── sec_empirical.tex
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

This creates training sets for conditions: `clean`, `contam_5`, `contam_15`, `contam_25`, `contam_50` (varying percentages of MMLU items mixed into Alpaca SFT data).

### 2. Training (QLoRA SFT)

```bash
python train.py --condition contam_15 --gpu 0
```

**Conditions:** `clean_s1`, `clean_s2`, ..., `contam_5`, `contam_15`, `contam_25`, `contam_50`

### 3. Evaluation (lz* computation)

```bash
python eval_lz.py --condition contam_15 --gpu 0
```

Evaluates the fine-tuned adapter on full MMLU using 3-shot prompting and computes:
- Theta MLE (ability estimate)
- lz* (Snijders 2001 corrected person-fit statistic)
- Accuracy (overall, seen items, unseen items)

### 4. Statistical Analysis

```bash
python compute_delta.py          # Delta lz* and summary statistics
python unified_lz_recompute.py   # Unified recomputation with theta-fixed analysis
python compute_cohens_d.py       # Cohen's d with bootstrap CIs
```

### 5. Cross-Model Replication

The `llama/` and `mistral/` directories contain analogous scripts for Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3. Usage is identical.

## Configuration

Before running, update the `BASE_DIR` and model path constants in each script to point to your local directories. Models used:
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Data

- MMLU items from `cais/mmlu` (HuggingFace datasets)
- ARC-Challenge items from `allenai/ai2_arc` (HuggingFace datasets)
- 4PL IRT parameters from PSN-IRT item bank (place `mmlu_params.csv` / `arc_c_params.csv` in each model directory)
- Alpaca SFT data from `tatsu-lab/alpaca`

## License

[To be added after review]
