# Reproducibility Workflow

This project supports single runs, 3-seed/5-fold sweeps, ablations, profiling,
bootstrap confidence intervals, and train/test leakage checks.

## 1. Environment

```bash
pip install -r requirements.txt
```

The expected dataset layout is:

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

Use `--data-root` if your dataset is stored elsewhere.

## 2. Single Runs

Train the proposed model:

```bash
python train_ablation.py --ablation full
```

Train one ablation:

```bash
python train_ablation.py --ablation no_spectral_fusion
python train_ablation.py --ablation no_ssm
python train_ablation.py --ablation no_msap
```

Train one baseline:

```bash
python train_baseline.py --model resnet50
python train_baseline.py --model efficientnet_b0
python train_baseline.py --model mobilenet_v3_large
python train_baseline.py --model convnext_tiny
python train_baseline.py --model swin_tiny
```

By default, baseline models are trained from scratch. Add `--pretrained` only
for a separate ImageNet-pretrained comparison table.

## 3. Three Seeds and Five Folds

Run FLUX-Net with 3 independent seeds and 5 folds:

```bash
python run_experiments.py \
  --experiments fluxnet \
  --seeds 42,123,2025 \
  --folds 0,1,2,3,4
```

Run FLUX-Net plus ablations:

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

For a quick command check without launching training:

```bash
python run_experiments.py --experiments fluxnet,no_ssm --seeds 42,123 --folds 0,1 --dry-run
```

## 4. Aggregate Mean +/- Std

After sweep runs finish:

```bash
python aggregate_results.py --output-dir outputs
```

This writes:

```text
outputs/aggregate_runs.csv
outputs/aggregate_summary.csv
outputs/aggregate_summary.md
```

## 5. FLOPs, Parameters, Latency, and Memory

Profile one model:

```bash
python profile_model.py --model fluxnet --csv outputs/profiling_results.csv
```

Profile all supported models:

```bash
python profile_model.py --all --csv outputs/profiling_results.csv
```

The FLOP estimate is hook-based and consistently counts convolution, linear,
and normalization layers. FFT, softmax, pooling, activations, and the Python SSM
scan are not fully counted, so report it as an approximate FLOP estimate.

## 6. Final Paper Table

Merge aggregate metrics and profiling into one table:

```bash
python make_paper_table.py \
  --aggregate outputs/aggregate_summary.csv \
  --profile outputs/profiling_results.csv \
  --out-csv outputs/paper_table.csv \
  --out-md outputs/paper_table.md
```

## 7. Bootstrap 95% Confidence Intervals

For a completed evaluation prediction file:

```bash
python bootstrap_ci.py \
  --predictions Training_results/evaluation/predictions.csv \
  --out-csv outputs/bootstrap_ci.csv
```

This reports accuracy, balanced accuracy, macro F1, and weighted F1 with 95%
bootstrap confidence intervals.

## 8. Dataset Leakage Check

Check exact and near duplicates between `Training/` and `Testing/`:

```bash
python check_dataset_leakage.py \
  --data-root data/kaggle-7023 \
  --out-csv outputs/leakage_matches.csv
```

Use the generated CSV to inspect any flagged exact or near-duplicate images.

## 9. Reporting Recommendation

For ESWA, report:

- Mean +/- std over folds/seeds for FLUX-Net.
- Main baseline table with parameter count, FLOPs, latency, memory, accuracy,
  macro F1, and AUC.
- Ablation table for no SpectralFusion, no SSM, and no MSAP.
- Bootstrap confidence interval for the final held-out test result.
- A short leakage-check statement describing exact/near-duplicate screening.
