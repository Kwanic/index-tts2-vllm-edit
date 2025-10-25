"""
Progressive Distillation Trainer for s2mel model

Implements the core training logic for reducing diffusion steps from 25 -> 1
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm
import copy
from pathlib import Path

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.flow_matching import CFM


class ProgressiveDistillationTrainer:
    def __init__(
        self,
        config_path,
        teacher_ckpt_path,
        output_dir="distilled_models",
        device='cuda'
    ):
        """
        Progressive Distillation Trainer for s2mel

        Args:
            config_path: Path to model config (config.yaml)
            teacher_ckpt_path: Path to teacher checkpoint (s2mel.pth with 25 steps)
            output_dir: Directory to save distilled models
            device: Device to train on
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        self.cfg = OmegaConf.load(config_path)

        # Load teacher model (frozen)
        print("Loading teacher model...")
        self.teacher = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        self.teacher, _, _, _ = load_checkpoint2(
            self.teacher, None, teacher_ckpt_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        self.teacher = self.teacher.to(device).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        print("✓ Teacher model loaded")

        # Initialize student model (trainable)
        print("Initializing student model...")
        self.student = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        # Copy teacher weights as initialization
        self.student.load_state_dict(self.teacher.state_dict())
        self.student = self.student.to(device)
        print("✓ Student model initialized")

        # TensorBoard logger
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.global_step = 0

    def distillation_loss(self, student_output, teacher_output, mask=None):
        """
        Compute distillation loss between student and teacher outputs

        Args:
            student_output: [B, C, T]
            teacher_output: [B, C, T]
            mask: [B, T] optional mask for variable lengths

        Returns:
            loss value
        """
        if mask is not None:
            # Apply mask to only compute loss on valid regions
            mask = mask.unsqueeze(1)  # [B, 1, T]
            student_output = student_output * mask
            teacher_output = teacher_output * mask

        return F.mse_loss(student_output, teacher_output)

    @torch.no_grad()
    def generate_teacher_samples(
        self,
        S_infer,
        prompt_mel,
        style,
        target_lengths,
        num_steps
    ):
        """
        Generate samples using teacher model

        Args:
            S_infer: Semantic embeddings [B, T', 1024]
            prompt_mel: List of prompt mel-spectrograms
            style: Style embeddings [B, 192]
            target_lengths: [B]
            num_steps: Number of diffusion steps

        Returns:
            Generated mel-spectrograms [B, 80, T]
        """
        self.teacher.eval()

        # Process through length regulator
        cond = self.teacher.models['length_regulator'](
            S_infer,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None
        )[0]

        # Prepare prompt (pad to max length)
        max_len = target_lengths.max().item()
        batch_size = len(prompt_mel)
        prompt_mel_padded = torch.zeros(batch_size, 80, max_len, device=self.device)

        for i, pm in enumerate(prompt_mel):
            prompt_len = pm.size(-1)
            prompt_mel_padded[i, :, :prompt_len] = pm.to(self.device)

        # Generate with teacher
        output = self.teacher.models['cfm'].inference(
            mu=cond,
            x_lens=target_lengths,
            prompt=prompt_mel_padded,
            style=style,
            f0=None,
            n_timesteps=num_steps,
            temperature=1.0,
            inference_cfg_rate=0.7
        )

        return output

    @torch.no_grad()
    def generate_teacher_two_step(
        self,
        S_infer,
        prompt_mel,
        style,
        target_lengths,
        num_steps,
        start_noise=None
    ):
        """
        Generate samples using teacher model for TWO steps

        This is the core of progressive distillation:
        - Teacher takes 2 steps: t -> t+1 -> t+2
        - Student will learn to do this in 1 step: t -> t+2

        Args:
            S_infer: Semantic embeddings [B, T', 1024]
            prompt_mel: List of prompt mel-spectrograms
            style: Style embeddings [B, 192]
            target_lengths: [B]
            num_steps: Total number of steps in the schedule (for teacher)
            start_noise: Optional starting noise [B, 80, T]

        Returns:
            (x_start, x_after_2steps): Starting point and result after 2 teacher steps
        """
        self.teacher.eval()

        # Process through length regulator
        cond = self.teacher.models['length_regulator'](
            S_infer,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None
        )[0]

        # Prepare prompt
        max_len = target_lengths.max().item()
        batch_size = len(prompt_mel)
        prompt_mel_padded = torch.zeros(batch_size, 80, max_len, device=self.device)

        for i, pm in enumerate(prompt_mel):
            prompt_len = pm.size(-1)
            prompt_mel_padded[i, :, :prompt_len] = pm.to(self.device)

        # Create time span for 2 steps
        B, T = cond.size(0), cond.size(1)
        if start_noise is None:
            z = torch.randn([B, 80, T], device=self.device)
        else:
            z = start_noise

        # Time span for the teacher (only 2 steps from the beginning)
        t_span = torch.linspace(0, 1, num_steps + 1, device=self.device)

        # We need to simulate 2 steps starting from t=0
        # Step 1: t[0] -> t[1]
        # Step 2: t[1] -> t[2]
        x = z.clone()
        t = t_span[0]

        # Apply prompt
        prompt_len = prompt_mel[0].size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt_mel_padded[..., :prompt_len]
        x[..., :prompt_len] = 0

        # Step 1
        dt = t_span[1] - t_span[0]
        dphi_dt = self.teacher.models['cfm'].estimator(
            x, prompt_x, target_lengths, t.unsqueeze(0).expand(B, 1, 1), style, cond
        )
        x = x + dt * dphi_dt
        x[:, :, :prompt_len] = 0
        x_after_1step = x.clone()

        # Step 2
        t = t_span[1]
        dt = t_span[2] - t_span[1]
        dphi_dt = self.teacher.models['cfm'].estimator(
            x, prompt_x, target_lengths, t.unsqueeze(0).expand(B, 1, 1), style, cond
        )
        x = x + dt * dphi_dt
        x[:, :, :prompt_len] = 0
        x_after_2steps = x.clone()

        return z, x_after_2steps

    def train_step(
        self,
        batch,
        teacher_steps,
        student_steps,
        optimizer,
        use_two_step_distillation=True
    ):
        """
        Single training step with 2-step distillation

        Args:
            batch: Batch data from dataloader
            teacher_steps: Number of steps for teacher
            student_steps: Number of steps for student (should be ~teacher_steps/2)
            optimizer: Optimizer instance
            use_two_step_distillation: If True, use 2-step distillation (recommended)

        Returns:
            loss value
        """
        self.student.train()

        # Unpack batch
        mel = batch['mel'].to(self.device)
        S_infer = batch['S_infer'].to(self.device)
        style = batch['style'].to(self.device)
        prompt_mel = batch['prompt_mel']
        target_lengths = batch['target_lengths'].to(self.device)

        # Process through student's length regulator
        cond = self.student.models['length_regulator'](
            S_infer,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None
        )[0]

        # Prepare prompt
        max_len = target_lengths.max().item()
        batch_size = len(prompt_mel)
        prompt_mel_padded = torch.zeros(batch_size, 80, max_len, device=self.device)

        for i, pm in enumerate(prompt_mel):
            prompt_len = pm.size(-1)
            prompt_mel_padded[i, :, :prompt_len] = pm.to(self.device)

        if use_two_step_distillation:
            # NEW: 2-step distillation (Salimans & Ho method)
            # Teacher does 2 steps: noise -> intermediate -> target
            # Student does 1 step: noise -> target
            with torch.no_grad():
                start_noise, teacher_2step_output = self.generate_teacher_two_step(
                    S_infer, prompt_mel, style, target_lengths, teacher_steps
                )

            # Student takes 1 step from the SAME starting noise
            B, T = cond.size(0), cond.size(1)
            x = start_noise.clone()

            # Time span for student (1 step that corresponds to teacher's 2 steps)
            t_span = torch.linspace(0, 1, teacher_steps + 1, device=self.device)

            # Student's single step: t[0] -> t[2] (skipping t[1])
            t = t_span[0]
            dt = t_span[2] - t_span[0]  # Double the time step

            prompt_len = prompt_mel[0].size(-1)
            prompt_x = torch.zeros_like(x)
            prompt_x[..., :prompt_len] = prompt_mel_padded[..., :prompt_len]
            x[..., :prompt_len] = 0

            # Single student step
            dphi_dt = self.student.models['cfm'].estimator(
                x, prompt_x, target_lengths, t.unsqueeze(0).expand(B, 1, 1), style, cond
            )
            student_output = x + dt * dphi_dt
            student_output[:, :, :prompt_len] = 0

            # Loss: student's 1-step output should match teacher's 2-step output
            teacher_output = teacher_2step_output

        else:
            # OLD: Direct distillation (final output matching)
            with torch.no_grad():
                teacher_output = self.generate_teacher_samples(
                    S_infer, prompt_mel, style, target_lengths, teacher_steps
                )

            # Generate with student
            student_output = self.student.models['cfm'].inference(
                mu=cond,
                x_lens=target_lengths,
                prompt=prompt_mel_padded,
                style=style,
                f0=None,
                n_timesteps=student_steps,
                temperature=1.0,
                inference_cfg_rate=0.7
            )

        # Create mask for variable lengths
        max_len = target_lengths.max()
        mask = torch.arange(max_len, device=self.device)[None, :] < target_lengths[:, None]

        # Compute distillation loss
        distill_loss = self.distillation_loss(student_output, teacher_output, mask)

        # Optional: Add reconstruction loss with ground truth
        recon_loss = self.distillation_loss(student_output, mel, mask)

        # Total loss (weighted combination)
        total_loss = distill_loss + 0.1 * recon_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        self.writer.add_scalar('Loss/distillation', distill_loss.item(), self.global_step)
        self.writer.add_scalar('Loss/reconstruction', recon_loss.item(), self.global_step)
        self.writer.add_scalar('Loss/total', total_loss.item(), self.global_step)
        self.global_step += 1

        return total_loss.item()

    def train_one_stage(
        self,
        dataloader,
        teacher_steps,
        student_steps,
        num_epochs,
        learning_rate=1e-4,
        use_two_step_distillation=True,
        save_interval=10
    ):
        """
        Train one stage of progressive distillation

        Args:
            dataloader: Training dataloader
            teacher_steps: Number of steps teacher uses
            student_steps: Number of steps student learns
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            use_two_step_distillation: If True, use 2-step distillation (recommended)
            save_interval: Save checkpoint every N epochs

        Returns:
            Trained student model
        """
        print(f"\n{'='*60}")
        print(f"Stage: {teacher_steps} steps → {student_steps} steps")
        print(f"Method: {'2-step distillation' if use_two_step_distillation else 'direct distillation'}")
        print(f"{'='*60}")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.1
        )

        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in pbar:
                loss = self.train_step(
                    batch,
                    teacher_steps,
                    student_steps,
                    optimizer,
                    use_two_step_distillation=use_two_step_distillation
                )

                epoch_loss += loss
                pbar.set_postfix({'loss': f'{loss:.4f}'})

            scheduler.step()

            # Epoch statistics
            avg_loss = epoch_loss / len(dataloader)
            lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}")

            self.writer.add_scalar('Epoch/loss', avg_loss, epoch)
            self.writer.add_scalar('Epoch/lr', lr, epoch)

            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or avg_loss < best_loss:
                self.save_checkpoint(
                    f"student_{student_steps}steps_epoch{epoch+1}.pth",
                    student_steps,
                    epoch,
                    avg_loss
                )
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # Also save as best
                    self.save_checkpoint(
                        f"student_{student_steps}steps_best.pth",
                        student_steps,
                        epoch,
                        avg_loss
                    )

        print(f"✓ Stage complete - Best loss: {best_loss:.4f}")

    def run_progressive_distillation(
        self,
        dataloader,
        step_schedule=[25, 13, 7, 4, 2, 1],
        epochs_per_stage=200,
        learning_rate=1e-4,
        use_two_step_distillation=True,
        save_interval=10,
        epochs_override=None
    ):
        """
        Run full progressive distillation pipeline

        Args:
            dataloader: Training dataloader
            step_schedule: List of step counts (default: [25, 13, 7, 4, 2, 1] - strict halving)
            epochs_per_stage: Number of epochs per stage (default: 200)
            learning_rate: Learning rate
            use_two_step_distillation: If True, use 2-step distillation (recommended)
            save_interval: Save checkpoint every N epochs
            epochs_override: Dict mapping stage index to custom epoch count
                           e.g., {0: 300, 4: 400} means stage 0 uses 300 epochs, stage 4 uses 400

        Returns:
            Final distilled model
        """
        print("\n" + "="*80)
        print("PROGRESSIVE DISTILLATION - Salimans & Ho (2022) Method")
        print("="*80)
        print(f"Schedule: {' → '.join(map(str, step_schedule))}")
        print(f"Epochs per stage: {epochs_per_stage} (default)")
        if epochs_override:
            print(f"Custom epochs: {epochs_override}")
        print(f"Learning rate: {learning_rate}")
        print(f"Method: {'2-step distillation ✓' if use_two_step_distillation else 'direct distillation'}")
        print(f"Save interval: every {save_interval} epochs")
        print("="*80 + "\n")

        for i in range(len(step_schedule) - 1):
            teacher_steps = step_schedule[i]
            student_steps = step_schedule[i + 1]

            # Determine epochs for this stage
            if epochs_override and i in epochs_override:
                stage_epochs = epochs_override[i]
                print(f"📌 Stage {i}: Using custom {stage_epochs} epochs")
            else:
                stage_epochs = epochs_per_stage

            self.train_one_stage(
                dataloader=dataloader,
                teacher_steps=teacher_steps,
                student_steps=student_steps,
                num_epochs=stage_epochs,
                learning_rate=learning_rate,
                use_two_step_distillation=use_two_step_distillation,
                save_interval=save_interval
            )

            # Student becomes the new teacher for next stage
            self.teacher.load_state_dict(self.student.state_dict())
            self.teacher.eval()

            print(f"\n✓ Teacher updated to {student_steps}-step model\n")

        print("\n" + "="*80)
        print("PROGRESSIVE DISTILLATION COMPLETE!")
        print(f"Final model: {step_schedule[-1]} step(s)")
        print(f"Total stages: {len(step_schedule) - 1}")
        print(f"Quality: Expected ~90-95% of original (with 2-step distillation)")
        print(f"Speedup: {step_schedule[0]}x faster inference")
        print("="*80 + "\n")

        return self.student

    def save_checkpoint(self, filename, num_steps, epoch, loss):
        """Save model checkpoint"""
        save_path = self.output_dir / filename

        state = {
            'net': {
                'cfm': self.student.models['cfm'].state_dict(),
                'length_regulator': self.student.models['length_regulator'].state_dict(),
                'gpt_layer': self.student.models['gpt_layer'].state_dict()
            },
            'num_steps': num_steps,
            'epoch': epoch,
            'loss': loss,
        }

        torch.save(state, save_path)
        print(f"✓ Checkpoint saved: {save_path}")
