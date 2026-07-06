# FLUX-Net Brain Tumor Classification

FLUX-Net is a from-scratch deep learning project for four-class brain MRI classification. It classifies images as `glioma`, `meningioma`, `notumor`, or `pituitary` using a lightweight hybrid architecture built around convolutional feature extraction, FFT-based spectral fusion, a compact S6-style state-space bottleneck, and a multi-scale attention pooling head.

This repository contains the completed model implementation, training and evaluation pipeline, ablation experiments, baseline comparisons, profiling scripts, reproducibility utilities, and generated result artifacts.

## Project Highlights

- Proposed model: **FLUX-Net**, short for Frequency Lightweight Unified X-attention Net.
- Task: four-class brain tumor MRI classification.
- Dataset layout: Kaggle-style `Training/` and `Testing/` folders.
- Architecture: Conv blocks, spectral attention blocks, S6-Lite bottleneck, and MSAP classification head.
- Training: AdamW, warmup cosine restarts, label smoothing, EMA, mixed precision, gradient accumulation, and early stopping.
- Evaluation: held-out test metrics, classification reports, confusion matrices, ROC/PR curves, confidence analysis, embeddings, and Grad-CAM visualizations.
- Reproducibility: seed/fold sweeps, ablations, baseline runs, profiling, bootstrap confidence intervals, and leakage checking.

## Repository Structure

```text
.
├── models/                         # FLUX-Net, baselines, data loading, training, losses, metrics
├── data/kaggle-7023/               # Expected local dataset directory
├── Training_results/               # Final training/evaluation artifacts
├── outputs/                        # Sweep outputs, manifests, aggregate tables, profiling outputs
├── train_ablation.py               # Train FLUX-Net and ablated variants
├── train_baseline.py               # Train baseline CNN/ViT-style models
├── run_experiments.py              # Run multi-seed, multi-fold experiments
├── aggregate_results.py            # Aggregate seed/fold results
├── profile_model.py                # Estimate parameters, FLOPs, latency, and memory
├── bootstrap_ci.py                 # Bootstrap confidence intervals
├── check_dataset_leakage.py        # Exact/near duplicate screening
├── make_paper_table.py             # Merge metrics and profiling into paper tables
├── architecture.md                 # Detailed architecture specification
└── REPRODUCIBILITY.md              # Reproducibility workflow
```

## Dataset

The project expects the dataset to be organized as:

```text
data/kaggle-7023/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

If the dataset is stored elsewhere, pass `--data-root /path/to/dataset` to the training and utility scripts.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include PyTorch, torchvision, albumentations, scikit-learn, pandas, NumPy, SciPy, OpenCV, Pillow, matplotlib, and tqdm.

## Model Overview

FLUX-Net processes `224 x 224` RGB MRI images through four spatial stages:

```text
Input 224x224x3
  -> Stem convolution
  -> Stage 1: ConvBlock + SpectralAttnBlock, 56x56, dim 48
  -> Stage 2: ConvBlock + SpectralAttnBlock, 28x28, dim 96
  -> Stage 3: ConvBlock + SpectralAttnBlock, 14x14, dim 192
  -> Stage 4: ConvBlock + SpectralAttnBlock, 7x7, dim 320
  -> S6-Lite bidirectional state-space bottleneck
  -> Multi-Scale Attention Pooling head
  -> Four-class logits
```

The core architectural components are:

- **SpectralFusion**: applies FFT-domain feature modulation with a learnable radial frequency gate.
- **S6-Lite bottleneck**: models long-range spatial dependencies at the final low-resolution feature map.
- **MSAP head**: combines pooled representations from all four stages with input-dependent scale attention.
- **Stochastic depth**: regularizes deeper layers with increasing drop-path rates.

See [architecture.md](/Users/manpreet/Documents/GitHub/BrainTumor/architecture.md) for the full architecture specification.

## Training

Train the complete FLUX-Net model:

```bash
python train_ablation.py --ablation full
```

Train an ablated variant:

```bash
python train_ablation.py --ablation no_spectral_fusion
python train_ablation.py --ablation no_ssm
python train_ablation.py --ablation no_msap
```

Useful training options:

```bash
python train_ablation.py \
  --ablation full \
  --data-root data/kaggle-7023 \
  --output-dir outputs \
  --experiment-name fluxnet_final \
  --epochs 200 \
  --batch-size 16 \
  --grad-accumulation-steps 8 \
  --fold 0 \
  --n-folds 5 \
  --seed 42
```

By default, training uses mixed precision when available. Add `--no-amp` to disable AMP.

## Baselines

Train a baseline model from scratch:

```bash
python train_baseline.py --model resnet50
python train_baseline.py --model efficientnet_b0
python train_baseline.py --model mobilenet_v3_large
python train_baseline.py --model convnext_tiny
python train_baseline.py --model swin_tiny
```

For a separate ImageNet-pretrained comparison, add `--pretrained`:

```bash
python train_baseline.py --model resnet50 --pretrained
```

## Reproducibility Workflow

Run FLUX-Net over multiple seeds and folds:

```bash
python run_experiments.py \
  --experiments fluxnet \
  --seeds 42,123,2025 \
  --folds 0,1,2,3,4
```

Run FLUX-Net with ablations:

```bash
python run_experiments.py \
  --experiments fluxnet,no_spectral_fusion,no_ssm,no_msap \
  --seeds 42,123,2025 \
  --folds 0,1,2,3,4
```

Run selected baselines:

```bash
python run_experiments.py \
  --experiments resnet50,efficientnet_b0,mobilenet_v3_large,convnext_tiny,swin_tiny \
  --seeds 42,123,2025 \
  --folds 0,1,2,3,4
```

Preview commands without launching training:

```bash
python run_experiments.py --experiments fluxnet,no_ssm --seeds 42,123 --folds 0,1 --dry-run
```

## Evaluation Artifacts

Completed evaluation artifacts are stored under:

```text
Training_results/evaluation/
Training_results/evaluation_tta/
Training_results/cross_dataset/
```

Typical outputs include:

- `metrics.json` and `metrics.csv`
- `predictions.csv`
- `classification_report.txt`
- `error_analysis.json`
- confusion matrix, ROC, PR, confidence, and per-class metric plots
- Grad-CAM sample heatmaps
- t-SNE embedding visualizations

## Aggregation and Reporting

Aggregate seed/fold runs:

```bash
python aggregate_results.py --output-dir outputs
```

This writes:

```text
outputs/aggregate_runs.csv
outputs/aggregate_summary.csv
outputs/aggregate_summary.md
```

Profile models:

```bash
python profile_model.py --model fluxnet --csv outputs/profiling_results.csv
python profile_model.py --all --csv outputs/profiling_results.csv
```

Build a final paper-ready table:

```bash
python make_paper_table.py \
  --aggregate outputs/aggregate_summary.csv \
  --profile outputs/profiling_results.csv \
  --out-csv outputs/paper_table.csv \
  --out-md outputs/paper_table.md
```

Compute bootstrap confidence intervals:

```bash
python bootstrap_ci.py \
  --predictions Training_results/evaluation/predictions.csv \
  --out-csv outputs/bootstrap_ci.csv
```

Check for train/test leakage:

```bash
python check_dataset_leakage.py \
  --data-root data/kaggle-7023 \
  --out-csv outputs/leakage_matches.csv
```

## Results

Final values should be filled from the generated evaluation and aggregation files. Placeholders are kept intentionally so the README can be updated with the exact reported numbers.

### Held-Out Test Set

| Metric | Value |
|:---|---:|
| Accuracy | `<RESULT_ACCURACY>` |
| Balanced Accuracy | `<RESULT_BALANCED_ACCURACY>` |
| Macro Precision | `<RESULT_MACRO_PRECISION>` |
| Macro Recall | `<RESULT_MACRO_RECALL>` |
| Macro F1 | `<RESULT_MACRO_F1>` |
| Weighted F1 | `<RESULT_WEIGHTED_F1>` |
| AUC-ROC | `<RESULT_AUC_ROC>` |
| AUC-PR | `<RESULT_AUC_PR>` |
| Log Loss | `<RESULT_LOG_LOSS>` |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|:---|---:|---:|---:|---:|
| Glioma | `<GLIOMA_PRECISION>` | `<GLIOMA_RECALL>` | `<GLIOMA_F1>` | `<GLIOMA_SUPPORT>` |
| Meningioma | `<MENINGIOMA_PRECISION>` | `<MENINGIOMA_RECALL>` | `<MENINGIOMA_F1>` | `<MENINGIOMA_SUPPORT>` |
| No Tumor | `<NOTUMOR_PRECISION>` | `<NOTUMOR_RECALL>` | `<NOTUMOR_F1>` | `<NOTUMOR_SUPPORT>` |
| Pituitary | `<PITUITARY_PRECISION>` | `<PITUITARY_RECALL>` | `<PITUITARY_F1>` | `<PITUITARY_SUPPORT>` |

### Seed/Fold Summary

| Experiment | Runs | Seeds | Folds | Accuracy | Macro F1 | AUC-ROC |
|:---|---:|:---|:---|---:|---:|---:|
| FLUX-Net | `<FLUXNET_RUNS>` | `<FLUXNET_SEEDS>` | `<FLUXNET_FOLDS>` | `<FLUXNET_ACC_MEAN_STD>` | `<FLUXNET_F1_MEAN_STD>` | `<FLUXNET_AUC_MEAN_STD>` |
| No Spectral Fusion | `<NO_SF_RUNS>` | `<NO_SF_SEEDS>` | `<NO_SF_FOLDS>` | `<NO_SF_ACC_MEAN_STD>` | `<NO_SF_F1_MEAN_STD>` | `<NO_SF_AUC_MEAN_STD>` |
| No SSM | `<NO_SSM_RUNS>` | `<NO_SSM_SEEDS>` | `<NO_SSM_FOLDS>` | `<NO_SSM_ACC_MEAN_STD>` | `<NO_SSM_F1_MEAN_STD>` | `<NO_SSM_AUC_MEAN_STD>` |
| No MSAP | `<NO_MSAP_RUNS>` | `<NO_MSAP_SEEDS>` | `<NO_MSAP_FOLDS>` | `<NO_MSAP_ACC_MEAN_STD>` | `<NO_MSAP_F1_MEAN_STD>` | `<NO_MSAP_AUC_MEAN_STD>` |

### Baseline Comparison

| Model | Pretrained | Params | FLOPs | Latency | Memory | Accuracy | Macro F1 | AUC-ROC |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| FLUX-Net | No | `<FLUXNET_PARAMS>` | `<FLUXNET_FLOPS>` | `<FLUXNET_LATENCY>` | `<FLUXNET_MEMORY>` | `<FLUXNET_ACC>` | `<FLUXNET_F1>` | `<FLUXNET_AUC>` |
| ResNet-50 | `<RESNET_PRETRAINED>` | `<RESNET_PARAMS>` | `<RESNET_FLOPS>` | `<RESNET_LATENCY>` | `<RESNET_MEMORY>` | `<RESNET_ACC>` | `<RESNET_F1>` | `<RESNET_AUC>` |
| EfficientNet-B0 | `<EFFNET_PRETRAINED>` | `<EFFNET_PARAMS>` | `<EFFNET_FLOPS>` | `<EFFNET_LATENCY>` | `<EFFNET_MEMORY>` | `<EFFNET_ACC>` | `<EFFNET_F1>` | `<EFFNET_AUC>` |
| MobileNetV3-Large | `<MOBILENET_PRETRAINED>` | `<MOBILENET_PARAMS>` | `<MOBILENET_FLOPS>` | `<MOBILENET_LATENCY>` | `<MOBILENET_MEMORY>` | `<MOBILENET_ACC>` | `<MOBILENET_F1>` | `<MOBILENET_AUC>` |
| ConvNeXt-Tiny | `<CONVNEXT_PRETRAINED>` | `<CONVNEXT_PARAMS>` | `<CONVNEXT_FLOPS>` | `<CONVNEXT_LATENCY>` | `<CONVNEXT_MEMORY>` | `<CONVNEXT_ACC>` | `<CONVNEXT_F1>` | `<CONVNEXT_AUC>` |
| Swin-Tiny | `<SWIN_PRETRAINED>` | `<SWIN_PARAMS>` | `<SWIN_FLOPS>` | `<SWIN_LATENCY>` | `<SWIN_MEMORY>` | `<SWIN_ACC>` | `<SWIN_F1>` | `<SWIN_AUC>` |

### Confidence Intervals

| Metric | Estimate | 95% CI Lower | 95% CI Upper |
|:---|---:|---:|---:|
| Accuracy | `<CI_ACCURACY>` | `<CI_ACCURACY_LOW>` | `<CI_ACCURACY_HIGH>` |
| Balanced Accuracy | `<CI_BALANCED_ACCURACY>` | `<CI_BALANCED_ACCURACY_LOW>` | `<CI_BALANCED_ACCURACY_HIGH>` |
| Macro F1 | `<CI_MACRO_F1>` | `<CI_MACRO_F1_LOW>` | `<CI_MACRO_F1_HIGH>` |
| Weighted F1 | `<CI_WEIGHTED_F1>` | `<CI_WEIGHTED_F1_LOW>` | `<CI_WEIGHTED_F1_HIGH>` |

## Cross-Dataset Evaluation

Cross-dataset artifacts are stored in:

```text
Training_results/cross_dataset/
```

Reported cross-dataset results:

| Dataset | Accuracy | Macro F1 | AUC-ROC | Notes |
|:---|---:|---:|---:|:---|
| BRISC2025 | `<BRISC2025_ACC>` | `<BRISC2025_MACRO_F1>` | `<BRISC2025_AUC>` | `<BRISC2025_NOTES>` |
| MRI Bounding Boxes | `<MRI_BBOX_ACC>` | `<MRI_BBOX_MACRO_F1>` | `<MRI_BBOX_AUC>` | `<MRI_BBOX_NOTES>` |

## Reproducibility Checklist

- Train/test split follows the dataset's `Training/` and `Testing/` directories.
- Validation uses stratified folds from the training set.
- Seeds and folds are tracked in run manifests.
- Final metrics are saved as JSON/CSV files.
- Classification reports and prediction CSVs are retained.
- Profiling outputs are generated separately from accuracy evaluation.
- Bootstrap confidence intervals are computed from prediction files.
- Exact and near-duplicate leakage checks can be run with `check_dataset_leakage.py`.

## Notes

- The project is intended for research and engineering evaluation, not standalone clinical diagnosis.
- Baseline models are trained from scratch by default unless `--pretrained` is explicitly passed.
- FLOP estimates from `profile_model.py` are approximate and consistently hook-based; FFT, softmax, pooling, activations, and the Python SSM scan are not fully counted.
- For detailed reproduction steps, see [REPRODUCIBILITY.md](/Users/manpreet/Documents/GitHub/BrainTumor/REPRODUCIBILITY.md).
