#!/bin/bash

# Example script for running progressive distillation
# Modify paths according to your setup

set -e  # Exit on error

# ========================================
# Configuration
# ========================================

# Paths
AUDIO_DIR="data/raw_audio"           # Your raw audio files
PREPROCESSED_DIR="data/preprocessed"  # Preprocessed features
MODEL_DIR="checkpoints/IndexTTS-2-vLLM"  # IndexTTS-2 checkpoint
OUTPUT_DIR="distilled_models"         # Output directory

# Training parameters
STEP_SCHEDULE="25,12,6,3,1"  # Distillation schedule
EPOCHS_PER_STAGE=50           # Epochs per stage
BATCH_SIZE=4                  # Batch size
LEARNING_RATE=1e-4            # Learning rate
NUM_WORKERS=4                 # DataLoader workers

# Data filtering
MAX_LENGTH=4096               # Max mel-spectrogram length
MIN_LENGTH=100                # Min mel-spectrogram length

# Device
DEVICE="cuda"                 # cuda or cpu

# ========================================
# Step 1: Preprocess Audio Data
# ========================================

echo "================================================"
echo "Step 1: Preprocessing audio data..."
echo "================================================"

if [ ! -d "$PREPROCESSED_DIR" ]; then
    python prepare_data.py \
        --audio_dir "$AUDIO_DIR" \
        --output_dir "$PREPROCESSED_DIR" \
        --model_dir "$MODEL_DIR" \
        --device "$DEVICE"

    echo "✓ Preprocessing complete"
else
    echo "✓ Preprocessed data already exists at $PREPROCESSED_DIR"
    echo "  (Delete this directory to reprocess)"
fi

# ========================================
# Step 2: Test Dataset Loading
# ========================================

echo ""
echo "================================================"
echo "Step 2: Testing dataset loading..."
echo "================================================"

python dataset.py \
    --data_dir "$PREPROCESSED_DIR" \
    --batch_size 2

echo "✓ Dataset test passed"

# ========================================
# Step 3: Run Progressive Distillation
# ========================================

echo ""
echo "================================================"
echo "Step 3: Training progressive distillation..."
echo "================================================"

python train.py \
    --data_dir "$PREPROCESSED_DIR" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --step_schedule "$STEP_SCHEDULE" \
    --epochs_per_stage "$EPOCHS_PER_STAGE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    --max_length "$MAX_LENGTH" \
    --min_length "$MIN_LENGTH" \
    --device "$DEVICE"

echo ""
echo "================================================"
echo "✓ Progressive distillation complete!"
echo "================================================"
echo "Models saved to: $OUTPUT_DIR"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "To use the distilled model:"
echo "  1. Copy the checkpoint to your model directory"
echo "  2. Modify infer_vllm_v2.py to use fewer diffusion steps"
echo "================================================"
