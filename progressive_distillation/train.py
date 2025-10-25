"""
Main training script for Progressive Distillation

Usage:
    python train.py --data_dir data/preprocessed --model_dir checkpoints/IndexTTS-2-vLLM
"""

import argparse
import torch
from pathlib import Path

from trainer import ProgressiveDistillationTrainer
from dataset import get_dataloader


def main():
    parser = argparse.ArgumentParser(description="Progressive Distillation Training")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with preprocessed training data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to IndexTTS-2 model directory")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="distilled_models",
                        help="Directory to save distilled models")
    parser.add_argument("--step_schedule", type=str, default="25,12,6,3,1",
                        help="Distillation schedule (comma-separated)")
    parser.add_argument("--epochs_per_stage", type=int, default=50,
                        help="Number of epochs per distillation stage")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")

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
        learning_rate=args.learning_rate
    )

    print("\n✓ Training complete!")
    print(f"✓ Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
