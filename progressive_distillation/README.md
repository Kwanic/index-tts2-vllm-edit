# Progressive Distillation for IndexTTS-2 s2mel

This directory contains scripts for performing **Progressive Distillation** on the s2mel (semantic-to-mel) model to reduce inference steps from **25 → 1**.

## 📊 Overview

**Progressive Distillation** is a technique to accelerate diffusion models by training them to use fewer steps:
- **Teacher**: Current model (25 steps)
- **Student**: Learns to match teacher output with fewer steps (12 → 6 → 3 → 1)

### Expected Results

| Steps | Inference Time | Quality | RTF (est.) |
|-------|---------------|---------|-----------|
| 25 (original) | 100% | ⭐⭐⭐⭐⭐ | 0.30 |
| 12 | ~50% | ⭐⭐⭐⭐⭐ | 0.18 |
| 6 | ~25% | ⭐⭐⭐⭐ | 0.12 |
| 3 | ~15% | ⭐⭐⭐ | 0.08 |
| 1 | ~5% | ⭐⭐ | 0.05 |

---

## 🚀 Quick Start

### Step 1: Prepare Your Audio Dataset

You need **audio files only** (no text/transcriptions required):

```bash
# Download a free dataset (example: LibriTTS)
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Or use your own audio files
mkdir -p data/raw_audio
# Copy your .wav files to data/raw_audio/
```

**Recommended datasets:**
- ✅ [LibriTTS](https://www.openslr.org/60/) (English, 24K hours, free)
- ✅ [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) (English multi-speaker, free)
- ✅ [Common Voice](https://commonvoice.mozilla.org/) (Multi-language, free)
- ✅ Your own audio data (any language)

---

### Step 2: Preprocess Audio

Extract features needed for training:

```bash
cd progressive_distillation

python prepare_data.py \
    --audio_dir ../data/raw_audio \
    --output_dir ../data/preprocessed \
    --model_dir ../checkpoints/IndexTTS-2-vLLM \
    --device cuda
```

**What this does:**
- Extracts mel-spectrograms
- Computes semantic codes (via w2v-bert)
- Extracts speaker style embeddings (via campplus)
- Saves everything as `.pt` files

**Processing time:** ~1-2 seconds per audio file

**Test with limited data first:**
```bash
python prepare_data.py \
    --audio_dir ../data/raw_audio \
    --output_dir ../data/preprocessed_test \
    --model_dir ../checkpoints/IndexTTS-2-vLLM \
    --max_files 100  # Process only 100 files for testing
```

---

### Step 3: Test Dataset Loading

Verify your preprocessed data loads correctly:

```bash
python dataset.py \
    --data_dir ../data/preprocessed \
    --batch_size 2
```

You should see output like:
```
✓ Loaded 10000 valid samples
  (Filtered from 10500 total files)

Batch 0:
  mel: torch.Size([2, 80, 1234])
  S_infer: torch.Size([2, 567, 1024])
  style: torch.Size([2, 192])
  ...
```

---

### Step 4: Train Progressive Distillation

Run the full training pipeline:

```bash
python train.py \
    --data_dir ../data/preprocessed \
    --model_dir ../checkpoints/IndexTTS-2-vLLM \
    --output_dir ../distilled_models \
    --step_schedule 25,12,6,3,1 \
    --epochs_per_stage 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_workers 4
```

**Training time (rough estimates):**
- Per epoch: ~10-30 minutes (depends on dataset size and GPU)
- Per stage (50 epochs): ~8-24 hours
- Full pipeline (4 stages): ~1-4 days

**Monitor training:**
```bash
tensorboard --logdir distilled_models/logs
```

---

### Step 5: Use Distilled Model for Inference

Replace the original s2mel checkpoint with your distilled model:

```python
# In your inference code
# Before:
diffusion_steps = 25  # Original

# After:
diffusion_steps = 1  # Or 3, 6, 12 depending on your distillation target
```

Or modify `infer_vllm_v2.py`:

```python
# Line ~420
# diffusion_steps = 25
diffusion_steps = 1  # Use 1-step distilled model
```

And update the checkpoint path:

```python
# Line ~139
s2mel_path = os.path.join(self.model_dir, "s2mel_distilled_1step.pth")
```

---

## 📁 File Structure

```
progressive_distillation/
├── README.md              # This file
├── prepare_data.py        # Preprocess audio files
├── dataset.py             # PyTorch Dataset & DataLoader
├── trainer.py             # Training logic
└── train.py              # Main training script

# Generated during training:
distilled_models/
├── logs/                  # TensorBoard logs
├── student_12steps_best.pth
├── student_6steps_best.pth
├── student_3steps_best.pth
└── student_1steps_best.pth
```

---

## ⚙️ Training Configuration

### Recommended Settings

**For small datasets (< 10K samples):**
```bash
python train.py \
    --step_schedule 25,12,6 \
    --epochs_per_stage 100 \
    --batch_size 8 \
    --learning_rate 1e-4
```

**For large datasets (> 100K samples):**
```bash
python train.py \
    --step_schedule 25,12,6,3,1 \
    --epochs_per_stage 30 \
    --batch_size 16 \
    --learning_rate 2e-4
```

**For fast experimentation:**
```bash
python train.py \
    --step_schedule 25,6,1 \
    --epochs_per_stage 20 \
    --batch_size 4
```

---

## 💡 Tips & Troubleshooting

### GPU Memory Issues

If you get OOM errors:

1. **Reduce batch size:**
   ```bash
   --batch_size 2
   ```

2. **Filter shorter sequences:**
   ```bash
   --max_length 2048  # Instead of 4096
   ```

3. **Use gradient accumulation** (modify `trainer.py`):
   ```python
   # Accumulate gradients over 4 steps
   if step % 4 == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

### Training Instability

If loss diverges or quality degrades:

1. **Lower learning rate:**
   ```bash
   --learning_rate 5e-5
   ```

2. **Train longer per stage:**
   ```bash
   --epochs_per_stage 100
   ```

3. **Use conservative schedule:**
   ```bash
   --step_schedule 25,18,12,8,6,4,2,1
   ```

### Poor Quality at 1-step

This is expected. Consider:

1. **Stop at 3 or 6 steps** for better quality
2. **Use Consistency Distillation** instead (more advanced)
3. **Increase training data** (quality improves with more data)

---

## 📊 Dataset Requirements

### Minimum Requirements
- **Size:** 1K+ audio files (10K+ recommended)
- **Duration:** 3-30 seconds per file
- **Quality:** Clean speech (minimal background noise)
- **Language:** Any language works

### Optimal Dataset
- **Size:** 100K+ audio files
- **Diverse speakers:** Multiple speakers improves generalization
- **Diverse content:** Various speaking styles, emotions
- **Clean recording:** Studio quality preferred

### You DON'T Need
- ❌ Transcriptions / text labels
- ❌ Aligned phonemes
- ❌ Speaker labels (though helpful)

---

## 🔬 Advanced: Consistency Distillation

For even better 1-step quality, consider **Consistency Distillation**:

```python
# TODO: Implement consistency distillation
# See: https://arxiv.org/abs/2303.01469
```

---

## 📚 References

- [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512)
- [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142)
- [Consistency Models](https://arxiv.org/abs/2303.01469)

---

## 🆘 Support

If you encounter issues:

1. Check the logs in `distilled_models/logs/`
2. Verify your preprocessed data with `dataset.py`
3. Test with a small dataset first (`--max_files 100`)
4. Open an issue with error logs

---

## 📝 Citation

If you use this code, please cite the original IndexTTS paper and the Progressive Distillation paper.
