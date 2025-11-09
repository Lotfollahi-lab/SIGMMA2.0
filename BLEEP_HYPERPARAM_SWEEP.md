# BLEEP Hyperparameter Sweep Recommendations

## Current Issues Diagnosed

Based on your training output:
- **pos_gap is NEGATIVE** (-0.06 to -0.12): Model assigns **higher similarity to wrong pairs** than correct ones
- **top1_acc near 0%**: Model cannot identify correct image-spot pairs
- **spot_std and img_std ~0.10-0.17**: Embeddings may have reasonable variance but aren't aligned

## Root Cause Hypotheses

1. **Temperature too high** → similarities too flat, no discrimination
2. **Learning rate issues** → not converging or overshooting
3. **Soft label targets** → model fighting against proper alignment
4. **Batch size too small** → noisy gradients, unstable contrastive learning
5. **Projection head collapse** → embeddings not separating properly

---

## Priority 1: Critical Hyperparameters

### 1. Temperature (`self.temperature`)
**Current unknown - likely 0.07 (BLEEP default)**

The temperature controls how "peaked" the similarity distribution is:
- Too high → all similarities look similar → no learning signal
- Too low → overly confident, unstable gradients

**Sweep:**
```python
temperature: [0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
```

**Expected behavior:**
- Lower temp (0.01-0.03): Sharper peaks, faster initial learning, risk of collapse
- Higher temp (0.1-0.2): Smoother learning, may be too flat

**Look for:** Positive pos_gap emerging earliest, stable training


### 2. Learning Rate
**Current unknown - check your training script**

**Sweep:**
```python
learning_rate: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
```

**Test with different optimizers:**
- AdamW (recommended, with weight_decay=0.01)
- Adam
- SGD with momentum=0.9

**Expected:** LR too high → divergence; too low → no progress


### 3. Batch Size
**Current unknown - appears small based on variance**

Contrastive learning needs larger batches for better negative sampling:

**Sweep:**
```python
batch_size: [32, 64, 128, 256]
```

**Note:** If GPU memory limited, use gradient accumulation:
```python
effective_batch_size = batch_size * accumulation_steps
accumulation_steps: [2, 4, 8]
```

**Expected:** Larger batch → more stable pos_gap, better top1_acc

---

## Priority 2: Architecture & Regularization

### 4. Projection Dimension
**Current: 256**

Higher dim → more capacity, but harder to align

**Sweep:**
```python
projection_dim: [128, 256, 512]
```

**Expected:** 128 may learn faster, 512 more expressive but slower


### 5. Dropout in Projection Head
**Current: 0.1**

**Sweep:**
```python
dropout: [0.0, 0.1, 0.2, 0.3]
```

**Expected:** Higher dropout prevents overfitting but may slow learning


### 6. Layer Normalization
**Currently COMMENTED OUT in BleepProjectionHead!**

```python
# x = self.layer_norm(x)  # <-- THIS IS DISABLED
```

**Test:**
- Enable LayerNorm (may stabilize training)
- Add LayerNorm BEFORE projection (alternative placement)

**Sweep:**
```python
use_layer_norm: [True, False]
layer_norm_position: ["after_residual", "before_projection"]
```

---

## Priority 3: Loss & Training Strategy

### 7. Soft Label Smoothing
**Current: Using average of image and spot similarities as targets**

This may be fighting the alignment! Consider:

**Alternative loss strategies:**
```python
# A) Hard labels (standard contrastive)
targets = torch.eye(batch_size, device=logits.device)

# B) Reduced smoothing
alpha = 0.5  # or 0.3, 0.7
targets = alpha * F.eye(...) + (1-alpha) * soft_targets

# C) Asymmetric - only smooth spot side (spots noisier)
spot_targets = soft_targets
image_targets = hard_targets
```

**Sweep:**
```python
label_smoothing_alpha: [0.0, 0.3, 0.5, 0.7, 1.0]  # 0=hard, 1=full soft
```


### 8. Quality Filtering Threshold
**Current: min_expressed_genes = 30**

**Sweep:**
```python
min_expressed_genes: [20, 30, 50, 100]
```

**Monitor:** % of valid spots used (currently shown in output)


### 9. Weight Decay
**Current unknown**

**Sweep:**
```python
weight_decay: [0.0, 0.01, 0.05, 0.1]
```

---

## Priority 4: Training Schedule

### 10. Warmup Steps
Add learning rate warmup to stabilize early training:

```python
warmup_steps: [100, 500, 1000]
warmup_schedule: ["linear", "cosine"]
```


### 11. LR Scheduler
**Sweep:**
```python
scheduler: ["cosine", "step", "reduce_on_plateau", "constant"]
# For cosine:
min_lr: [1e-6, 1e-7]
# For step:
step_size: [5, 10]
gamma: [0.1, 0.5]
```

---

## Suggested Experiment Plan

### Phase 1: Emergency Fix (Run First)
**Goal: Get ANY positive learning signal**

```yaml
experiment_1_baseline_fix:
  temperature: 0.03  # Much lower
  learning_rate: 1e-4
  batch_size: 64
  label_smoothing_alpha: 0.0  # Hard labels
  use_layer_norm: True
  epochs: 10
```

```yaml
experiment_2_temp_sweep:
  temperature: [0.01, 0.03, 0.05, 0.1]
  learning_rate: 1e-4
  batch_size: 64
  label_smoothing_alpha: 0.0
  epochs: 10
```

### Phase 2: Optimization (If Phase 1 works)
```yaml
experiment_3_lr_batch:
  temperature: <best_from_phase1>
  learning_rate: [3e-5, 1e-4, 3e-4]
  batch_size: [32, 64, 128]
  label_smoothing_alpha: [0.0, 0.3, 0.5]
  epochs: 20
```

### Phase 3: Architecture Tuning
```yaml
experiment_4_architecture:
  # Use best config from Phase 2
  projection_dim: [128, 256, 512]
  dropout: [0.0, 0.1, 0.2]
  layer_norm_position: ["before", "after"]
  epochs: 20
```

---

## Key Metrics to Track

### Primary Success Metrics
1. **pos_gap** → Should become POSITIVE and increase (target: 0.2-0.5)
2. **top1_acc** → Should increase from 0% to >50% by epoch 5-10
3. **diag vs off_diag** → Diagonal should be clearly higher

### Secondary Health Metrics
4. **img_self_sim & spot_self_sim** → Should both be ~1.0 (self-similarity)
5. **img_off_diag_sim & spot_off_diag_sim** → Should be positive but < diagonal
6. **embedding_std** → Should stay >0.1 (not collapsed)
7. **cross_diag_std vs cross_off_diag_std** → Diagonal should have lower std (more consistent)

### Warning Signs
- **pos_gap decreasing** → Wrong direction, stop and adjust
- **embedding_std < 0.05** → Embedding collapse
- **loss not decreasing** → Learning rate or batch size issue
- **All similarities → 1.0** → Temperature too high or normalization issue

---

## Quick Diagnostic Commands

Check current training config:
```bash
# Find training script
grep -r "temperature" img_spatial_alignment/scripts/
grep -r "learning_rate\|lr" img_spatial_alignment/scripts/
grep -r "batch_size" img_spatial_alignment/scripts/
```

Monitor training live:
```bash
tail -f runs_bleep/*_train.out | grep -E "pos_gap|Loss|Alignment"
```

---

## Expected Healthy Training Progression

**Epoch 1-2:**
- pos_gap: -0.1 → 0.0 → +0.05 ✅
- top1_acc: 0% → 1-5% ✅

**Epoch 3-5:**
- pos_gap: +0.05 → +0.15 ✅
- top1_acc: 5% → 20% ✅

**Epoch 10+:**
- pos_gap: +0.2 → +0.4 ✅
- top1_acc: 30% → 60% ✅

**Your current training (epoch 3):**
- pos_gap: -0.06 to -0.12 ❌ (WRONG DIRECTION)
- top1_acc: 0-0.1% ❌ (NO PROGRESS)

→ **Action required: Run Phase 1 experiments immediately**

---

## Code Modifications Needed

I've already added detailed metrics. You'll also need to modify:

1. **Training script** - add hyperparameters as arguments
2. **Config file** - create sweep configurations
3. **Loss function** - add label smoothing parameter
4. **BleepProjectionHead** - make LayerNorm configurable

Would you like me to help implement any of these changes?
