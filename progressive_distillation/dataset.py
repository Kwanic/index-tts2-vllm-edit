"""
PyTorch Dataset for Progressive Distillation Training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class S2MelDistillationDataset(Dataset):
    """
    Dataset for s2mel progressive distillation training

    Loads preprocessed features created by prepare_data.py
    """
    def __init__(self, data_dir, max_length=4096, min_length=100):
        """
        Args:
            data_dir: Directory containing preprocessed .pt files
            max_length: Maximum mel-spectrogram length (for memory)
            min_length: Minimum mel-spectrogram length
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.min_length = min_length

        # Find all preprocessed files
        self.data_files = list(self.data_dir.glob("*.pt"))

        # Filter by length
        print(f"Loading dataset from {data_dir}...")
        self.valid_files = []
        for f in self.data_files:
            try:
                data = torch.load(f)
                if min_length <= data['target_length'] <= max_length:
                    self.valid_files.append(f)
            except Exception as e:
                print(f"Skipping corrupted file {f}: {e}")

        print(f"✓ Loaded {len(self.valid_files)} valid samples")
        print(f"  (Filtered from {len(self.data_files)} total files)")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        # Load preprocessed data
        data = torch.load(self.valid_files[idx])

        return {
            'mel': data['mel'].squeeze(0),                  # [80, T]
            'S_infer': data['S_infer'].squeeze(0),          # [T', 1024]
            'style': data['style'].squeeze(0),              # [192]
            'prompt_mel': data['prompt_mel'].squeeze(0),    # [80, prompt_len]
            'target_length': data['target_length'],         # scalar
            'prompt_len': data['prompt_len'],               # scalar
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    Handles variable-length sequences
    """
    # Find max lengths in batch
    max_mel_len = max(item['target_length'] for item in batch)
    max_sem_len = max(item['S_infer'].size(0) for item in batch)

    batch_size = len(batch)

    # Initialize padded tensors
    mel_padded = torch.zeros(batch_size, 80, max_mel_len)
    S_infer_padded = torch.zeros(batch_size, max_sem_len, 1024)
    style_batch = torch.zeros(batch_size, 192)
    prompt_mel_batch = []
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        mel_len = item['target_length']
        sem_len = item['S_infer'].size(0)

        mel_padded[i, :, :mel_len] = item['mel']
        S_infer_padded[i, :sem_len, :] = item['S_infer']
        style_batch[i] = item['style']
        prompt_mel_batch.append(item['prompt_mel'])
        target_lengths[i] = mel_len

    return {
        'mel': mel_padded,
        'S_infer': S_infer_padded,
        'style': style_batch,
        'prompt_mel': prompt_mel_batch,  # List of variable-length tensors
        'target_lengths': target_lengths,
    }


def get_dataloader(
    data_dir,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    max_length=4096,
    min_length=100
):
    """
    Create DataLoader for training

    Args:
        data_dir: Directory with preprocessed data
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_length: Maximum sequence length
        min_length: Minimum sequence length

    Returns:
        DataLoader instance
    """
    dataset = S2MelDistillationDataset(
        data_dir=data_dir,
        max_length=max_length,
        min_length=min_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # For stable batch size
    )


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    loader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0  # For testing
    )

    print("\nTesting dataloader...")
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  mel: {batch['mel'].shape}")
        print(f"  S_infer: {batch['S_infer'].shape}")
        print(f"  style: {batch['style'].shape}")
        print(f"  prompt_mel: {len(batch['prompt_mel'])} items")
        print(f"  target_lengths: {batch['target_lengths']}")

        if i >= 2:  # Test first 3 batches
            break

    print("\n✓ Dataset test passed!")
