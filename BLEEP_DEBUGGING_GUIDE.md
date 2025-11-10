# BLEEP Model Debugging Guide

This guide explains the debugging statements added to the BLEEP model and what to look for to diagnose training issues.

## Debugging Output Sections

### 1. Input Debugging
```
[BLEEP Forward Pass]
Input shapes, ranges, mean, std
```
**What to check:**
- **Shapes:** Ensure batch sizes match and dimensions are correct
- **Ranges:** Image features should be in a reasonable range (typically 0-1 or -1 to 1 for normalized images)
- **Mean/Std:** Check if inputs are properly normalized
  - If std is very small (~0), data might be constant or poorly preprocessed
  - If mean is very far from 0 and std is large, consider normalization

**Red flags:**
- NaN or Inf values
- Zero std (indicates constant features)
- Spot features with very different scale than images

---

### 2. Image Encoder Output
```
[After Image Encoder]
Embeddings shape, range, mean, std, norm
```
**What to check:**
- **Norm (per sample):** Should be reasonably bounded (e.g., 10-100 for ResNet50)
  - If norm is exploding (>1000), you may have gradient explosion
  - If norm is vanishing (<0.1), features might be collapsing
- **Mean/Std:** Check if outputs are reasonable
  - All zeros or very small values = dead network
  - Very large values = potential overflow issues

---

### 3. Projection Head Outputs
```
[After Image Projection]
[After Spot Projection]
```
**What to check:**
- **Norm consistency:** Image and spot embeddings should have similar norms
  - Large imbalance (e.g., one is 10x the other) will hurt contrastive learning
- **Distribution:** Both should have similar mean/std characteristics
  - If one modality has much larger variance, it will dominate the similarity computation

**Red flags:**
- Norm collapsing to zero
- One modality significantly larger than the other
- NaN values appearing after projection

---

### 4. Cross-Modal Similarity (Logits)
```
[Logits - Cross-Modal Similarity]
Diagonal vs off-diagonal comparison
```
**What to check:**
- **Diagonal vs off-diagonal gap:** This is THE KEY metric!
  - **Positive gap (diagonal > off-diagonal):** Model is learning! ✅
  - **Gap ~0:** Model not distinguishing correct from incorrect pairs ❌
  - **Negative gap:** Model is actively wrong ❌
  
**Expected progression:**
- Early training: Gap ~0 or slightly negative (random)
- During training: Gap should steadily increase
- Good training: Gap should be >0.5 to 2.0

**Example:**
```
Diagonal mean: 2.5, Off-diagonal mean: 0.3
Gap: 2.2  <- Great! Model is learning
```

**Red flags:**
- Gap not increasing over epochs
- Very large gap early (>5) might indicate overfitting or data issues
- Logits saturating (all very large or very small)

---

### 5. Self-Similarity Matrices
```
[Self-Similarity Matrices]
```
**What to check:**
- **Diagonal values:** Should be highest (each sample most similar to itself)
  - If diagonal isn't largest, embeddings are problematic
- **Off-diagonal values:** Should be lower than diagonal
  - High off-diagonal = embeddings are too similar (collapse)

**Healthy pattern:**
```
images_similarity diagonal: 50-200 (depends on embedding norm)
images_similarity off-diagonal: lower than diagonal
```

---

### 6. Targets Distribution
```
[Targets]
```
**What to check:**
- **Row sums:** Must be ~1.0 (it's a softmax)
- **Diagonal values:** Should be dominant (>0.5 ideally)
  - If diagonal is low (<0.3), the soft labels are too diffuse
- **Entropy:** 
  - Low entropy (~0-1): Sharp distribution (confident labels) ✅
  - High entropy (>3): Very uniform (uncertain labels) ❌

**Red flags:**
- Targets are uniform (all ~1/batch_size)
- Diagonal not highest
- Row sums not equal to 1.0

---

### 7. Loss Components
```
[Loss Components]
spots_loss and images_loss breakdown
```
**What to check:**
- **Loss values:** Should be positive and bounded
  - Typical range: 0.5 to 5.0 for cross-entropy
  - If >10, something is likely wrong
- **Symmetry:** spots_loss and images_loss should be similar
  - Large asymmetry indicates one modality is easier to learn
- **Variance (std):** 
  - High variance = some samples much harder than others
  - Zero variance = might indicate a bug

**Expected trajectory:**
- Should decrease over training epochs
- If stuck at a high value (>3) after many epochs, model isn't learning

---

### 8. Gradient Flow Check
```
[Gradient Flow Check]
requires_grad, temperature
```
**What to check:**
- **requires_grad:** Must be True for both embeddings
  - If False, gradients won't flow back!
- **Temperature:** Check if it's reasonable (typically 0.01 to 1.0)
  - Too high (>1): Similarities too soft
  - Too low (<0.01): Similarities too sharp, unstable

---

## Common Issues and Diagnoses

### Issue 1: Loss Not Decreasing
**Check:**
1. Diagonal vs off-diagonal gap - is it increasing?
2. Embedding norms - are they stable?
3. Gradient flow - are gradients enabled?
4. Learning rate might be too low

### Issue 2: Loss Exploding
**Check:**
1. Input ranges - are they normalized?
2. Embedding norms - are they exploding?
3. Temperature value - is it too small?
4. Gradient clipping might be needed

### Issue 3: Model Collapsing (All Embeddings Same)
**Check:**
1. Embedding norms - are they all the same?
2. Self-similarity off-diagonal - is it too high?
3. Diagonal vs off-diagonal gap - is it zero?
4. Might need stronger regularization or different initialization

### Issue 4: One Modality Dominates
**Check:**
1. Projection output norms - are they balanced?
2. Loss components - is one much larger?
3. Input statistics - are they on similar scales?
4. May need to normalize embeddings or adjust architecture

---

## Quick Diagnostic Checklist

Run your model for 1-2 batches and check:

- [ ] No NaN or Inf values anywhere
- [ ] Input features have reasonable ranges and non-zero std
- [ ] Embedding norms are stable (not exploding or vanishing)
- [ ] Diagonal > off-diagonal in logits (the gap should be positive)
- [ ] Target diagonal values are high (>0.5)
- [ ] Loss is in a reasonable range (0.5-5.0)
- [ ] Both modalities have similar loss contributions
- [ ] Gradients are flowing (requires_grad=True)

---

## Monitoring During Training

Track these metrics across epochs:

1. **Primary metric:** Diagonal vs off-diagonal gap (should increase)
2. **Loss trend:** Should decrease steadily
3. **Embedding norms:** Should remain stable
4. **Target entropy:** Should remain low (sharp distributions)

If after 10-20 epochs you don't see improvement in the diagonal/off-diagonal gap, there's likely a fundamental issue with the model or data.
