#!/bin/bash

# Example script for running progressive distillation
# Implements Salimans & Ho (2022) 2-step distillation
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

# Training parameters (NEW DEFAULTS)
STEP_SCHEDULE="25,13,7,4,2,1"  # Strict halving schedule (recommended)
EPOCHS_PER_STAGE=200           # Increased from 50 to 200 for better quality
BATCH_SIZE=4                   # Batch size
LEARNING_RATE=1e-4             # Learning rate
NUM_WORKERS=4                  # DataLoader workers
SAVE_INTERVAL=10               # Save checkpoint every N epochs

# Distillation method
USE_TWO_STEP=true              # Use 2-step distillation (recommended)
# Set to false for old direct distillation: USE_TWO_STEP=false

# Optional: Custom epochs for specific stages
# Format: "stage_index:epochs,stage_index:epochs"
# Example: "0:300,4:400" means stage 0 uses 300 epochs, stage 4 uses 400
EPOCHS_OVERRIDE=""             # Leave empty to use EPOCHS_PER_STAGE for all

# Data filtering
MAX_LENGTH=4096                # Max mel-spectrogram length
MIN_LENGTH=100                 # Min mel-spectrogram length

# Device
DEVICE="cuda"                  # cuda or cpu

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

# Build command with optional parameters
CMD="python train.py \
    --data_dir \"$PREPROCESSED_DIR\" \
    --model_dir \"$MODEL_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --step_schedule \"$STEP_SCHEDULE\" \
    --epochs_per_stage \"$EPOCHS_PER_STAGE\" \
    --batch_size \"$BATCH_SIZE\" \
    --learning_rate \"$LEARNING_RATE\" \
    --num_workers \"$NUM_WORKERS\" \
    --save_interval \"$SAVE_INTERVAL\" \
    --max_length \"$MAX_LENGTH\" \
    --min_length \"$MIN_LENGTH\" \
    --device \"$DEVICE\""

# Add 2-step distillation flag
if [ "$USE_TWO_STEP" = false ]; then
    CMD="$CMD --no_two_step_distillation"
fi

# Add epochs override if specified
if [ -n "$EPOCHS_OVERRIDE" ]; then
    CMD="$CMD --epochs_override \"$EPOCHS_OVERRIDE\""
fi

# Run training
eval $CMD

echo ""
echo "================================================"
echo "✓ Progressive distillation complete!"
echo "================================================"
echo "Models saved to: $OUTPUT_DIR"
echo ""
echo "📊 Training Summary:"
echo "   Schedule: $STEP_SCHEDULE"
echo "   Epochs per stage: $EPOCHS_PER_STAGE"
echo "   Method: $([ "$USE_TWO_STEP" = true ] && echo '2-step distillation (Salimans & Ho 2022)' || echo 'direct distillation')"
echo ""
echo "📈 Monitor training:"
echo "   tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "🚀 Use the distilled model:"
echo "   1. Best checkpoint: $OUTPUT_DIR/student_1steps_best.pth"
echo "   2. Modify inference to use 1 step instead of 25"
echo "   3. Expected speedup: 25x faster"
echo "   4. Expected quality: ~90-95% (with 2-step distillation)"
echo "================================================"
