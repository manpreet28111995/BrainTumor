# FLUX-Net Architecture Specification

**F**requency **L**ightweight **U**nified **X**-attention **Net**

A hybrid grid architecture combining interleaved Conv + Spectral Attention blocks, FFT-based spectral fusion, and an S6-Lite SSM bottleneck — trained entirely from scratch.

---

## 1. Architecture Flow

```
Input (224×224×3)
    │
  [Stem: 2× Conv3×3 stride 2]         → 56×56×48
    │
  ┌── Stage 1 (56×56, dim=48, ×2 cells, SD=5%) ───────────────┐
  │  StochasticDepth → ConvBlock → SpectralAttnBlock  × 2     │
  │  ConvBlock = LN → DW 7×7 → PW 1×1(×3) → GELU → PW 1×1   │
  │  SpectralAttnBlock = LN → SpectralFusion → FFN(×2)        │
  └──────────────────────┬──────────────────────────────────────┘
                         │ Downsample (strided conv dim×2)
  ┌── Stage 2 (28×28, dim=96, ×3 cells, SD=10%) ──────────────┐
  │  StochasticDepth → ConvBlock → SpectralAttnBlock  × 3     │
  └──────────────────────┬──────────────────────────────────────┘
                         │ Downsample
  ┌── Stage 3 (14×14, dim=192, ×6 cells, SD=15%) ─────────────┐
  │  StochasticDepth → ConvBlock → SpectralAttnBlock  × 6     │
  └──────────────────────┬──────────────────────────────────────┘
                         │ Downsample
  ┌── Stage 4 (7×7, dim=320, ×2 cells, SD=20%) ───────────────┐
  │  StochasticDepth → ConvBlock → SpectralAttnBlock  × 2     │
  │  → S6-Lite SSM Bottleneck (bidirectional, d_state=8)      │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌── MSAP Head ──────────────────────────────────────────────┐
  │  Pool Stage 1-4 → Scale Attention → MLP → 4-class logits  │
  └────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1. StochasticDepth
```
Input [B, C, H, W]
  → During training: drop entire block with probability sd_drop_rate
  → Scale surviving paths by 1/(1 - sd_drop_rate) to maintain expected value
  → During inference: identity (no drop, no scaling)
Output [B, C, H, W]
```

Drop rates increase linearly across stages (5% → 10% → 15% → 20%) so deeper layers regularize more. Prevents co-adaptation between blocks (ConvNeXt / Swin proven technique).

### 2.2. ConvBlock
```
Input [B, C, H, W]
  → LayerNorm (channel-wise)
  → Depthwise Conv 7×7 (groups=C)
  → Pointwise Conv 1×1 (C → C×3)
  → GELU
  → Pointwise Conv 1×1 (C×3 → C)
  → Dropout 0.1
  + Residual connection
Output [B, C, H, W]
```

### 2.3. SpectralAttnBlock (Core Novelty)
```
Input [B, C, H, W]
  → LayerNorm
  → SpectralFusion:
      1. 2D FFT → frequency domain
      2. Learnable radial frequency gate (sigmoid-logit radius)
      3. Channel-wise modulation (1×1 conv in spectral domain)
      4. iFFT → spatial domain
  → FFN: Linear(C→2C) → GELU → Linear(2C→C)
  + Residual connection
Output [B, C, H, W]
```

The learnable radius controls the low-pass / high-pass balance:
- **radius → 1**: emphasizes low frequencies (tumor core, homogeneous regions)
- **radius → 0**: emphasizes high frequencies (tumor edges, texture boundaries)

### 2.4. S6-Lite Bottleneck
```
Input [B, 320, 7, 7]
  → Flatten to 49 tokens [B, 49, 320]
  → Positional embedding
  → Bidirectional S6 SSM (forward + reverse)
  → Reshape back [B, 320, 7, 7]
Output [B, 320, 7, 7]
```

Pure PyTorch implementation of the Selective State-Space Model (Mamba-style) without custom CUDA kernels.

### 2.5. MSAP Classification Head
```
Input: Feature maps from all 4 stages
  Stage 1: [B, 48, 56, 56]
  Stage 2: [B, 96, 28, 28]
  Stage 3: [B, 192, 14, 14]
  Stage 4: [B, 320, 7, 7]
  → Per-scale: AvgPool + MaxPool → Linear(2C_i → 128)
  → Scale Attention (input-dependent softmax weights)
  → Weighted sum → MLP → 4-class logits
```

---

## 3. Parameter Count

| Module | Params |
|:---|:---:|
| Stem | 0.01 M |
| Stage 1 (dim=48, ×2 cells) | 0.06 M |
| Stage 2 (dim=96, ×3 cells) | 0.32 M |
| Stage 3 (dim=192, ×6 cells) | 2.50 M |
| Stage 4 (dim=320, ×2 cells) | 2.29 M |
| Downsample layers | 0.76 M |
| S6-Lite SSM Bottleneck (2 layers) | 2.31 M |
| MSAP Head | 0.24 M |
| StochasticDepth modules | 0 (no trainable params) |
| **Total** | **8.51 M** |

---

## 4. Training Configuration

| Setting | Value |
|:---|:---|
| Initialization | Custom initialization: Kaiming normal convs, unit norm layers, truncated-normal linear layers |
| Input range | [0, 1] → Normalize to [-1, 1] |
| Optimizer | AdamW (lr=5e-4, wd=0.05, betas=(0.9, 0.999)) |
| Scheduler | Warmup cosine schedule with warm restarts (T_0=30, T_mult=2) |
| Warmup epochs | 10 |
| Batch size | 16 (effective 128 via 8× grad accumulation) |
| Epochs | 200 + early stopping (patience=25) |
| Augmentation | HorizontalFlip, ShiftScaleRotate, CLAHE, ColorJitter, GaussNoise, CoarseDropout |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Stochastic depth (sd_prob) | 0.2 (per-stage: 0.05, 0.10, 0.15, 0.20) |
| EMA decay (ema_decay) | 0.99 |
| Mixed precision | FP16 (automatic) |
| Cross-validation | Stratified 5-fold (fold 0 used for final) |

---

## 5. Overfitting Prevention Summary

| Technique | Where | Effect |
|:---|:---|:---|
| **StochasticDepth** | Every block in every stage | Randomly skips entire blocks during training; prevents co-adaptation |
| **Dropout (0.3)** | ConvBlock, SpectralAttnBlock FFN, and MSAPHead classifier | Drops random features |
| **Label smoothing (0.1)** | ClassificationLoss | Softens targets; prevents overconfidence |
| **Weight decay (0.05)** | AdamW optimizer | Decoupled L2 regularization on non-bias/non-normalization parameters |
| **EMA (0.99 decay)** | Validation + test inference | Smooths weight trajectory |
| **Early stopping (patience=25)** | Per-epoch monitoring | Stops when val accuracy/loss plateaus |
| **Stratified 5-fold CV** | Data splitting | Ensures class balance across folds |

**Overfitting detection (logged every epoch):**
```
Gap — Acc: +0.0603 Loss: +0.0909
```
- `Acc gap > +0.15` → model is memorizing training set, not generalizing
- `Loss gap > +0.20` → same signal from loss perspective
- Both printed every epoch for real-time monitoring

---

## 6. Key Design Decisions

| Decision | Rationale |
|:---|:---|
| **From scratch** | Avoids ImageNet bias; proves architecture novelty |
| **Grid layout** | Single path with interleaved blocks is parameter-efficient |
| **FFT spectral fusion** | Frequency-domain processing offers unique inductive bias for medical images |
| **No self-attention** | Avoids O(N²) cost — spectral fusion is O(N log N) |
| **SSM at bottleneck** | Linear-complexity long-range modeling at 7×7 |
| **Multi-scale head** | Tumors vary 10× in size — single-scale GAP is insufficient |
| **StochasticDepth** | Prevents co-adaptation between blocks without adding parameters |
| **EMA inference** | Validation/test uses smoothed weights for better generalization |

---

## 7. Results

### 7.1. Held-Out Test Set Performance

Evaluated on **5,600 samples** (balanced: 1,400 per class) from the Kaggle-7023 test set using EMA weights.

| Metric | Value |
|:---|---:|
| Accuracy | **97.66%** |
| Balanced Accuracy | 97.66% |
| Macro F1 | 97.66% |
| Macro Precision | 97.75% |
| Macro Recall | 97.66% |
| Cohen's Kappa | 96.88% |
| Matthews Correlation | 96.91% |
| AUC-ROC | **0.9983** |
| AUC-PR (macro) | 0.9945 |
| Log Loss | 0.157 |

### 7.2. Per-Class Breakdown

| Class | Precision | Recall | F1-Score | IoU / Dice |
|:---|---:|---:|---:|---:|
| Glioma | 94.11% | 99.29% | 96.63% | 93.48% / 96.63% |
| Meningioma | 97.33% | 98.93% | 98.12% | 96.31% / 98.12% |
| No Tumor | 99.78% | 99.07% | 99.43% | 98.86% / 99.43% |
| Pituitary | 99.77% | 93.36% | 96.46% | 93.16% / 96.46% |

### 7.3. Confusion Matrix

| Actual \ Predicted | Glioma | Meningioma | No Tumor | Pituitary |
|:---|---:|---:|---:|---:|
| Glioma | **1,390** | 10 | 0 | 0 |
| Meningioma | 10 | **1,385** | 2 | 3 |
| No Tumor | 10 | 3 | **1,387** | 0 |
| Pituitary | 67 | 25 | 1 | **1,307** |

### 7.4. Training Progress

- **Best epoch:** 183 (val accuracy 97.05%, val loss 0.503)
- **Total epochs trained:** 200 (maximum configured epoch budget)
- **Final train-val gap:** Acc +0.06, Loss +0.09 (no overfitting)

### 7.5. Test-Time Augmentation (TTA)

TTA evaluation on an additional **1,600 samples** using multi-augmentation voting. This run is reported as an auxiliary robustness check, not as the primary result, because it was evaluated on a different subset from the 5,600-sample held-out test set:

| Metric | Value |
|:---|---:|
| Accuracy | 92.75% |
| Macro F1 | 92.68% |
| AUC-ROC | 0.9799 |

### 7.6. Cross-Dataset Evaluation

Cross-dataset results are available in `Training_results/cross_dataset/`, demonstrating generalization beyond the Kaggle distribution.

---

*FLUX-Net achieves 97.66% test accuracy and 0.998 AUC-ROC with only **8.51M parameters**, trained entirely from scratch on a single Tesla T4 GPU.*
