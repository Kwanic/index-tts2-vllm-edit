# Dataset Preparation Guide

## Core Concept

**Progressive Distillation does NOT require text labels!** This is pure audio-to-audio distillation.

You only need:
- ✅ Audio files (any language)
- ❌ No text transcriptions required
- ❌ No phoneme alignment required
- ❌ No speaker labels required (optional)

---

## 🎯 Dataset Selection

### Recommended Public Datasets

#### 1. **LibriTTS** (Highly Recommended)
- **Language**: English
- **Size**: ~200GB (train-clean-100: ~24GB)
- **Speakers**: 2,456 speakers
- **Duration**: 585 hours
- **Download**: https://www.openslr.org/60/

```bash
# Download training set (100 hours, clean speech)
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Directory structure:
# LibriTTS/train-clean-100/
#   ├── 19/
#   │   ├── 198/
#   │   │   ├── 19_198_000001_000000.wav
#   │   │   ├── 19_198_000001_000001.wav
#   │   │   └── ...
```

#### 2. **VCTK** (Multi-speaker)
- **Language**: English (with accents)
- **Size**: ~11GB
- **Speakers**: 110 speakers
- **Download**: https://datashare.ed.ac.uk/handle/10283/3443

```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip
```

#### 3. **LJSpeech** (Single speaker)
- **Language**: English
- **Size**: ~2.6GB
- **Speakers**: 1 speaker (female)
- **Download**: https://keithito.com/LJ-Speech-Dataset/

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
```

#### 4. **Common Voice** (Multi-language)
- **Language**: 100+ languages (including Chinese)
- **Size**: Variable
- **Download**: https://commonvoice.mozilla.org/

```bash
# Requires account registration to download
# For Chinese dataset:
wget https://mozilla-common-voice-datasets.s3.amazonaws.com/cv-corpus-14.0-2023-06-23/zh-CN.tar.gz
```

#### 5. **CN-Celeb** (Chinese)
- **Language**: Chinese
- **Speakers**: 1,000+ speakers
- **Download**: http://openslr.org/82/

---

## 📋 Dataset Requirements

### Minimum Requirements
- **File Count**: 1,000+ audio files
- **Duration**: 3-30 seconds per file
- **Sample Rate**: Any (will be resampled to 22.05kHz and 16kHz)
- **Format**: WAV, MP3, FLAC, M4A, etc.
- **Quality**: Clear speech, minimal background noise

### Recommended Configuration
- **File Count**: 10,000+ audio files (more is better)
- **Duration**: 5-15 seconds (too short or too long is not ideal)
- **Speakers**: Multiple speakers (better generalization)
- **Recording Environment**: Clean studio quality

### Data Distribution
```
Ideal dataset distribution:

Duration distribution:
  3-5s:    20%
  5-10s:   50%  ← Optimal range
  10-20s:  25%
  20-30s:  5%

Speaker distribution:
  Single speaker: Acceptable, but less generalization
  Multi-speaker:  Recommended (10+ speakers)
```

---

## 🛠️ Preparing Your Dataset

### Option 1: Using Public Datasets

```bash
# 1. Download LibriTTS (train-clean-100)
cd data
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# 2. Organize file structure (optional, script will recursively search)
mkdir -p raw_audio
find LibriTTS/train-clean-100 -name "*.wav" -exec cp {} raw_audio/ \;

# 3. Preprocess
cd ../progressive_distillation
python prepare_data.py \
    --audio_dir ../data/raw_audio \
    --output_dir ../data/preprocessed \
    --model_dir ../checkpoints/IndexTTS-2-vLLM
```

### Option 2: Using Your Own Audio

```bash
# 1. Create directory
mkdir -p data/my_audio

# 2. Copy your audio files
cp /path/to/your/audio/*.wav data/my_audio/

# 3. Check files
ls data/my_audio/ | head -10

# 4. Preprocess
cd progressive_distillation
python prepare_data.py \
    --audio_dir ../data/my_audio \
    --output_dir ../data/preprocessed \
    --model_dir ../checkpoints/IndexTTS-2-vLLM
```

### Option 3: Extract Audio from YouTube/Videos

```bash
# Use yt-dlp to download audio
pip install yt-dlp

# Download and convert to WAV
yt-dlp -x --audio-format wav \
    -o "data/youtube/%(title)s.%(ext)s" \
    "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Split into short segments (using pydub)
python split_audio.py --input data/youtube --output data/my_audio --duration 10
```

---

## 🧪 Data Preprocessing Pipeline

### 1. Basic Preprocessing

```bash
python prepare_data.py \
    --audio_dir data/raw_audio \
    --output_dir data/preprocessed \
    --model_dir checkpoints/IndexTTS-2-vLLM \
    --device cuda
```

**Output**:
```
Loading IndexTTS2 models from checkpoints/IndexTTS-2-vLLM...
✓ All models loaded successfully
Found 13100 audio files
Output directory: data/preprocessed
Processing audio: 100%|██████████| 13100/13100 [2:15:30<00:00,  1.61it/s]

✓ Processing complete!
  Successful: 13050
  Failed: 50
  Failed files logged to: data/preprocessed/failed_files.txt
```

### 2. Verify Preprocessing Results

```bash
python dataset.py --data_dir data/preprocessed --batch_size 4
```

**Expected output**:
```
✓ Loaded 13050 valid samples
  (Filtered from 13100 total files)

Batch 0:
  mel: torch.Size([4, 80, 1234])
  S_infer: torch.Size([4, 567, 1024])
  style: torch.Size([4, 192])
  prompt_mel: 4 items
  target_lengths: tensor([1234, 1105, 998, 1456])
```

### 3. Preprocessed Content

Each audio file is converted to:

```python
{
    'mel': Tensor([1, 80, T]),           # Mel-spectrogram (ground truth)
    'semantic_codes': Tensor([1, T']),    # Semantic codes (discrete)
    'S_infer': Tensor([1, T', 1024]),    # Semantic embeddings (continuous)
    'style': Tensor([1, 192]),            # Speaker style embedding
    'prompt_mel': Tensor([1, 80, P]),    # First ~3 seconds as prompt
    'prompt_len': int,                    # Prompt length
    'target_length': int,                 # Target mel length
}
```

---

## 📊 Dataset Statistics

### View Dataset Information

Create script `dataset_stats.py`:

```python
import torch
from pathlib import Path
import numpy as np

data_dir = Path("data/preprocessed")
files = list(data_dir.glob("*.pt"))

lengths = []
for f in files:
    data = torch.load(f)
    lengths.append(data['target_length'])

lengths = np.array(lengths)

print(f"Dataset Statistics:")
print(f"  Total samples: {len(lengths)}")
print(f"  Mean length: {lengths.mean():.0f} frames")
print(f"  Min length: {lengths.min()} frames")
print(f"  Max length: {lengths.max()} frames")
print(f"  Median: {np.median(lengths):.0f} frames")
print(f"  Std dev: {lengths.std():.0f} frames")

# Duration distribution (assuming hop_size=256, sr=22050)
durations = lengths * 256 / 22050
print(f"\nDuration Distribution:")
print(f"  < 5s: {(durations < 5).sum()} ({(durations < 5).mean()*100:.1f}%)")
print(f"  5-10s: {((durations >= 5) & (durations < 10)).sum()} ({((durations >= 5) & (durations < 10)).mean()*100:.1f}%)")
print(f"  10-20s: {((durations >= 10) & (durations < 20)).sum()} ({((durations >= 10) & (durations < 20)).mean()*100:.1f}%)")
print(f"  > 20s: {(durations >= 20).sum()} ({(durations >= 20).mean()*100:.1f}%)")
```

Run:
```bash
python dataset_stats.py
```

---

## ❓ FAQ

### Q1: How much data do I need?

**Minimum**: 1,000 audio files (can train, but quality may be limited)
**Recommended**: 10,000+ audio files
**Optimal**: 100,000+ audio files (approaching original model quality)

### Q2: Can I mix different languages?

✅ **Yes!** Your dataset can contain multiple languages, which may even improve generalization.

### Q3: Audio quality requirements?

- **Clean recording**: ⭐⭐⭐⭐⭐ (Best)
- **Slight background noise**: ⭐⭐⭐⭐ (Acceptable)
- **Noisy environment**: ⭐⭐ (Not recommended)

### Q4: Do I need paired text?

❌ **No!** Progressive Distillation is self-supervised and only requires audio.

### Q5: Can I use music?

❌ **Not recommended.** The model is designed for speech, not music.

### Q6: Data augmentation?

Optional data augmentation methods:
- ✅ Speed perturbation (0.9x - 1.1x)
- ✅ Volume normalization
- ⚠️ Pitch shift (use with caution)
- ❌ Adding noise not recommended (degrades quality)

---

## 📦 Preprocessed Data Format

Preprocessed directory structure:

```
data/preprocessed/
├── audio_001.pt
├── audio_002.pt
├── audio_003.pt
├── ...
└── failed_files.txt  (if any files failed)
```

Each `.pt` file contains a dictionary that can be loaded directly:

```python
import torch

data = torch.load("data/preprocessed/audio_001.pt")

print(data.keys())
# dict_keys(['mel', 'semantic_codes', 'S_infer', 'style',
#            'prompt_mel', 'prompt_len', 'target_length', 'audio_path'])

print(data['mel'].shape)          # torch.Size([1, 80, 1234])
print(data['S_infer'].shape)      # torch.Size([1, 567, 1024])
print(data['style'].shape)        # torch.Size([1, 192])
```

---

## 🚀 Next Steps

After data preparation is complete, start training:

```bash
cd progressive_distillation
bash run_example.sh
```

Or:

```bash
python train.py \
    --data_dir ../data/preprocessed \
    --model_dir ../checkpoints/IndexTTS-2-vLLM \
    --output_dir ../distilled_models
```
