# BLEEP Training Issue: Quick Start Guide

## 🚨 Problem Summary
Your BLEEP model has **negative pos_gap** (-0.06 to -0.12), meaning it assigns **higher similarity to wrong image-spot pairs than correct ones**. The model is learning the opposite of what it should!

## ✅ What I've Done

### 1. Enhanced Metrics (Already Applied)
I've updated `models_multimodal_image_encoders.py` to show detailed similarity metrics:

**New output format:**
```
[Alignment] pos_gap=-0.0961 | diag=-0.1234 | off_diag=-0.0273 | top1=0.0%
[Img_Sim] self=1.0000 | off_diag=0.8234 | std=0.0523
[Spot_Sim] self=1.0000 | off_diag=0.7891 | std=0.0612
[Cross_Sim] diag_std=0.0234 | off_diag_std=0.0156
[Embeddings] spot_std=0.1667 | img_std=0.1075 | mean_nz_genes=59.5
```

**What to look for:**
- ✅ `pos_gap` should be **positive** and increase over epochs (target: 0.2-0.5)
- ✅ `diag` should be **higher** than `off_diag` in [Alignment]
- ✅ `self` should be ~1.0 (embeddings normalized correctly)
- ✅ `off_diag` in Img_Sim and Spot_Sim should be **lower** than self
- ⚠️ Watch for embedding collapse: `spot_std` and `img_std` should stay > 0.1

### 2. Created Sweep Configurations
I've created organized sweep configs in `configs/sweeps/`:

```
configs/sweeps/
├── temp/
│   ├── hest1k-LUAD_BLEEP_temp_0.01.yaml
│   ├── hest1k-LUAD_BLEEP_temp_0.03.yaml
│   ├── hest1k-LUAD_BLEEP_temp_0.05.yaml
│   └── hest1k-LUAD_BLEEP_temp_0.07.yaml
├── lr/  (create similar for LR sweep)
├── batch/  (create similar for batch sweep)
└── best/  (for final combo)
```

### 3. Created Analysis Tools
- `analyze_sweep_results.py` - Monitor sweep progress
- `BLEEP_HYPERPARAM_SWEEP.md` - Complete guide

## 🏃 Quick Start: Run Emergency Fix

### Step 1: Test Lower Temperature (Most Critical)
```bash
cd /lustre/scratch126/cellgen/lotfollahi/mm58/spatial_alignment/img_spatial_alignment

# Test temperature 0.01 (much lower than current 0.07)
bsub -J bleep_temp01 \
     -q gpu-normal \
     -gpu "num=1:mode=exclusive_process:gmem=40000" \
     -R "select[mem>64000] rusage[mem=64000]" \
     -M 64000 \
     -n 8 \
     -o runs_bleep/temp01_%J.out \
     -e runs_bleep/temp01_%J.err \
     "python scripts/train_model.py -config_file configs/sweeps/temp/hest1k-LUAD_BLEEP_temp_0.01.yaml"
```

### Step 2: Monitor Training
```bash
# Watch in real-time
tail -f runs_bleep/temp01_*.out | grep -E "Alignment|pos_gap"

# After 5 epochs, check if pos_gap is improving:
grep "pos_gap" runs_bleep/temp01_*.out | tail -20
```

### Step 3: If Temperature 0.01 Works
Run full temperature sweep:
```bash
# Sweep all 4 temperature values
for temp in 0.01 0.03 0.05 0.07; do
    bsub -J bleep_temp${temp} \
         -q gpu-normal \
         -gpu "num=1:mode=exclusive_process:gmem=40000" \
         -R "select[mem>64000] rusage[mem=64000]" \
         -M 64000 \
         -n 8 \
         -o runs_bleep/temp${temp}_%J.out \
         -e runs_bleep/temp${temp}_%J.err \
         "python scripts/train_model.py -config_file configs/sweeps/temp/hest1k-LUAD_BLEEP_temp_${temp}.yaml"
done
```

### Step 4: Analyze Results
```bash
python analyze_sweep_results.py runs_bleep
```

## 🔍 What Success Looks Like

**Epoch 1-2 (Healthy Training):**
```
[Alignment] pos_gap=0.0234 | diag=0.5123 | off_diag=0.4889 | top1=2.3%
[Alignment] pos_gap=0.0567 | diag=0.5678 | off_diag=0.5111 | top1=5.8%
```
✅ pos_gap is **positive** and **increasing**

**Epoch 5-10 (Good Progress):**
```
[Alignment] pos_gap=0.1523 | diag=0.6891 | off_diag=0.5368 | top1=18.7%
[Alignment] pos_gap=0.2341 | diag=0.7456 | off_diag=0.5115 | top1=31.2%
```
✅ pos_gap > 0.15, top1_acc > 15%

**Your Current Training (❌ WRONG):**
```
[Alignment] pos_gap=-0.0961 | diag=-0.1234 | off_diag=-0.0273 | top1=0.0%
[Alignment] pos_gap=-0.0779 | diag=-0.1567 | off_diag=-0.0788 | top1=0.0%
```
❌ pos_gap is **negative** (diag < off_diag)

## 📋 Full Sweep Plan

### Phase 1: Temperature (Days 1-2)
Run all 4 temperature configs, pick best by:
1. First positive pos_gap
2. Highest max_pos_gap in first 10 epochs
3. Stable/increasing trend

### Phase 2: Learning Rate (Days 3-4)
Using best temperature, test:
- lr=0.0001
- lr=0.0003
- lr=0.001 (current)

### Phase 3: Batch Size (Days 5-6)
Using best temp + lr, test:
- batch=256
- batch=512
- batch=1024 (current)
- batch=2048

### Phase 4: Final Training (Days 7+)
With best hyperparameters, train for 150 epochs

## 🐛 Debugging Tips

**If pos_gap stays negative:**
- Lower temperature more (try 0.005, 0.001)
- Check if data is corrupted (look at mean_nz_genes)
- Try hard labels instead of soft (requires code change)

**If training crashes:**
- Reduce batch size
- Lower learning rate
- Check GPU memory

**If metrics show NaN:**
- Add gradient clipping
- Check for division by zero in loss

**If embeddings collapse (std < 0.05):**
- Increase dropout
- Add weight decay
- Use layer normalization

## 📞 Next Steps

1. **Immediate**: Run temperature=0.01 config and monitor for 5 epochs
2. **If working**: Launch full temperature sweep
3. **If not working**: I can help modify the loss function to use hard labels
4. **After sweep**: Analyze results with the analysis script

## 📚 Key Files Created

1. `configs/sweeps/temp/hest1k-LUAD_BLEEP_temp_*.yaml` - Temperature sweep configs
2. `BLEEP_HYPERPARAM_SWEEP.md` - Detailed hyperparameter guide
3. `analyze_sweep_results.py` - Results analysis script
4. Enhanced metrics in `model/model/models_multimodal_image_encoders.py`

---

**Need help?** Let me know if you want me to:
- Create the remaining sweep configs (LR, batch size)
- Modify the loss function to try hard labels
- Add gradient clipping or other debugging features
- Create visualization scripts for the metrics
