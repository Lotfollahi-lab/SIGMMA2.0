# Critical Bugs Fixed in BLEEP Model

## 🚨 Summary of Issues Found

Your model was **NOT learning** due to fundamental bugs in the similarity computation. **DO NOT train for 150 epochs without these fixes!**

---

## Issues Identified from Training Logs

### Issue 1: NEGATIVE Diagonal vs Off-Diagonal Gap ❌
```
Diagonal vs off-diagonal gap: -0.2461 to -0.8418
```
**Problem:** The model was scoring INCORRECT pairs HIGHER than correct pairs - learning backwards!

### Issue 2: Zero Entropy Targets ❌
```
targets entropy: 0.0000
targets diagonal: mean=1.0000
```
**Problem:** Soft labels completely degenerate (probability 1.0 on diagonal, 0.0 elsewhere). This defeats the purpose of BLEEP's soft distillation approach.

### Issue 3: Extremely High Loss Values ❌
```
Loss: ~34.0 (should be 0.5-5.0)
```
**Problem:** Loss values >30 indicate something fundamentally broken.

### Issue 4: Saturated Self-Similarities ❌
```
images_similarity diagonal: mean=255.87
images_similarity off-diagonal: mean=125.81
```
**Problem:** Unnormalized similarities cause softmax saturation, making gradients vanish.

---

## Root Causes

### Cause 1: Missing Normalization
The embeddings were **NOT normalized** before computing similarities. This caused:
- Self-similarities with huge values (255.87 instead of 1.0)
- Softmax saturation (all probability on diagonal)
- Vanishing gradients
- Zero entropy targets

### Cause 2: Wrong Temperature Application
```python
# WRONG (original):
targets = F.softmax((images_similarity + spots_similarity) / 2 * self.temperature, dim=-1)
```

You were **multiplying** by temperature instead of **dividing**. This made the problem worse by amplifying already huge values before softmax.

---

## Fixes Applied

### Fix 1: Normalize Embeddings ✅
```python
# Normalize to unit vectors before computing similarities
image_embeddings_norm = F.normalize(image_embeddings, dim=-1)
spot_embeddings_norm = F.normalize(spot_embeddings, dim=-1)

# Now similarities are cosine similarities in range [-1, 1]
logits = (spot_embeddings_norm @ image_embeddings_norm.T) / self.temperature
images_similarity = image_embeddings_norm @ image_embeddings_norm.T
spots_similarity = spot_embeddings_norm @ spot_embeddings_norm.T
```

**Effect:**
- Self-similarity diagonal will now be ~1.0 (not 255.87)
- Similarities are now in a reasonable range [-1, 1]
- Softmax won't saturate
- Targets will have proper entropy

### Fix 2: Correct Temperature Application ✅
```python
# CORRECT:
targets = F.softmax((images_similarity + spots_similarity) / 2 / self.temperature, dim=-1)
```

**Effect:**
- Temperature now properly controls the sharpness of soft labels
- Lower temperature = sharper distribution
- Higher temperature = softer distribution

---

## Expected Changes After Fix

### What You Should See Now:

1. **Self-Similarities:**
   ```
   images_similarity diagonal: mean ~1.0000 (was 255.87)
   spots_similarity diagonal: mean ~1.0000 (was 255.75)
   images_similarity off-diagonal: mean ~0.0 to 0.5 (was 125.81)
   ```

2. **Targets Entropy:**
   ```
   targets entropy: 1.0 to 3.0 (was 0.0000)
   targets diagonal: 0.3 to 0.8 (was 1.0000)
   ```

3. **Loss Values:**
   ```
   Loss: 1.0 to 5.0 (was ~34.0)
   ```

4. **Diagonal vs Off-Diagonal Gap:**
   ```
   Gap: Starting near 0, should INCREASE over epochs (was negative)
   After training: Gap should be > 0.5
   ```

---

## What to Monitor During Training

### Critical Metrics (Check Every Few Epochs):

1. **Diagonal vs Off-Diagonal Gap** - Should be POSITIVE and INCREASING
   - Epoch 1-5: ~0.0 to 0.2
   - Epoch 10-20: ~0.3 to 0.7
   - Epoch 50+: ~0.7 to 2.0

2. **Targets Entropy** - Should be POSITIVE (1-3 range)
   - Not 0.0 (too sharp)
   - Not >4.0 (too uniform)

3. **Loss Trajectory** - Should DECREASE
   - Should start at 2-5
   - Should decrease steadily
   - Should converge below 1.0 for good training

4. **Self-Similarity Diagonals** - Should be ~1.0
   - If not 1.0, normalization isn't working

---

## Verification Before 150-Epoch Training

**Run a SHORT test (5-10 epochs) and verify:**

✅ Diagonal vs off-diagonal gap is POSITIVE  
✅ Gap is INCREASING over epochs  
✅ Targets entropy is > 0 (ideally 1-3)  
✅ Loss is in range 1.0-5.0 and decreasing  
✅ Self-similarity diagonals are ~1.0  
✅ No NaN or Inf values  

**If any of these fail, STOP and debug before full training!**

---

## Why This Matters for BLEEP

BLEEP's key innovation is using **soft labels** from self-similarities to provide richer training signal than hard contrastive learning. The original bugs:

1. Made soft labels degenerate (entropy=0) → No advantage over standard contrastive learning
2. Caused learning to go backwards (negative gap) → Model getting worse, not better
3. Led to exploding loss values → Unstable training, poor convergence

With these fixes, the model can now:
- Learn proper soft label distributions
- Leverage cross-modal and within-modal similarities
- Train stably with reasonable loss values
- Actually improve over epochs

---

## Temperature Recommendations

With proper normalization, temperature controls sharpness:

- **temperature=0.07**: Very sharp (like CLIP) - hard contrastive
- **temperature=0.1**: Moderately sharp - good default
- **temperature=0.5**: Softer labels - more regularization
- **temperature=1.0**: Current setting - quite soft

For BLEEP, **temperature=0.1 to 0.5** is recommended. Your current value of 1.0 might be too soft.

Consider testing:
```python
self.temperature = 0.1  # or 0.2
```

---

## Next Steps

1. ✅ **DONE:** Applied normalization fix
2. ✅ **DONE:** Fixed temperature application
3. **TODO:** Run 5-10 epoch test to verify fixes work
4. **TODO:** Check that gap is positive and increasing
5. **TODO:** Consider adjusting temperature to 0.1-0.5
6. **TODO:** Once verified, proceed with 150-epoch training

**Do NOT start 150-epoch training without verifying the fixes work!**
