"""
Main training script for Progressive Distillation

Implements Salimans & Ho (2022) Progressive Distillation for TTS

Usage:
    # Basic usage with default 2-step distillation
    python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM

    # Custom schedule (strict halving recommended)
    python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM \
        --step_schedule 25,13,7,4,2,1 --epochs_per_stage 200

    # Different epochs for different stages
    python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM \
        --epochs_per_stage 200 --epochs_override "0:300,4:400"

    # Use old direct distillation (not recommended)
    python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM \
        --no_two_step_distillation
"""

import argparse
import torch
from pathlib import Path

from trainer import ProgressiveDistillationTrainer
from dataset import get_dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Progressive Distillation Training (Salimans & Ho 2022)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 2-step distillation (recommended)
  python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM

  # Custom epochs per stage
  python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM \\
      --epochs_per_stage 300

  # Override epochs for specific stages
  python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM \\
      --epochs_override "0:300,4:400"
        """
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with preprocessed training data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to IndexTTS-2 model directory")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="distilled_models",
                        help="Directory to save distilled models")
    parser.add_argument("--step_schedule", type=str, default="25,13,7,4,2,1",
                        help="Distillation schedule (default: strict halving 25→13→7→4→2→1)")
    parser.add_argument("--epochs_per_stage", type=int, default=200,
                        help="Number of epochs per distillation stage (default: 200)")
    parser.add_argument("--epochs_override", type=str, default=None,
                        help="Override epochs for specific stages, format: '0:300,4:400'")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Distillation method
    parser.add_argument("--no_two_step_distillation", action="store_true",
                        help="Disable 2-step distillation (use direct distillation instead)")

    # Data filtering
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum mel-spectrogram length")
    parser.add_argument("--min_length", type=int, default=100,
                        help="Minimum mel-spectrogram length")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")

    args = parser.parse_args()

    # Parse step schedule
    step_schedule = [int(x) for x in args.step_schedule.split(',')]
    print(f"Step schedule: {' → '.join(map(str, step_schedule))}")

    # Validate schedule for 2-step distillation
    use_two_step = not args.no_two_step_distillation
    if use_two_step:
        print("✓ Using 2-step distillation (Salimans & Ho 2022)")
        # Check if schedule is approximately halving
        for i in range(len(step_schedule) - 1):
            ratio = step_schedule[i] / step_schedule[i+1]
            if ratio < 1.8 or ratio > 2.2:
                print(f"⚠️  Warning: Step {step_schedule[i]}→{step_schedule[i+1]} is not ~2x halving (ratio: {ratio:.2f})")
                print("   For best results with 2-step distillation, use strict halving like [25,13,7,4,2,1]")
    else:
        print("⚠️  Using direct distillation (quality may be lower)")

    # Parse epochs override
    epochs_override = None
    if args.epochs_override:
        epochs_override = {}
        for pair in args.epochs_override.split(','):
            stage, epochs = pair.split(':')
            epochs_override[int(stage)] = int(epochs)
        print(f"Custom epochs: {epochs_override}")

    # Create dataloader
    print(f"\nLoading dataset from {args.data_dir}...")
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_length=args.max_length,
        min_length=args.min_length
    )

    # Initialize trainer
    print(f"\nInitializing trainer...")
    config_path = Path(args.model_dir) / "config.yaml"
    teacher_ckpt_path = Path(args.model_dir) / "s2mel.pth"

    trainer = ProgressiveDistillationTrainer(
        config_path=str(config_path),
        teacher_ckpt_path=str(teacher_ckpt_path),
        output_dir=args.output_dir,
        device=args.device
    )

    # Run progressive distillation
    print(f"\nStarting training...\n")
    final_model = trainer.run_progressive_distillation(
        dataloader=train_loader,
        step_schedule=step_schedule,
        epochs_per_stage=args.epochs_per_stage,
        learning_rate=args.learning_rate,
        use_two_step_distillation=use_two_step,
        save_interval=args.save_interval,
        epochs_override=epochs_override
    )

    print("\n✓ Training complete!")
    print(f"✓ Models saved to: {args.output_dir}")
    print(f"\n📊 Summary:")
    print(f"   Initial steps: {step_schedule[0]}")
    print(f"   Final steps: {step_schedule[-1]}")
    print(f"   Speedup: {step_schedule[0]}x")
    print(f"   Method: {'2-step distillation' if use_two_step else 'direct distillation'}")


if __name__ == "__main__":
    main()
