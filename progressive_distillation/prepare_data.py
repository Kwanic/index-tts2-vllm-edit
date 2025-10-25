"""
Prepare training data for Progressive Distillation of s2mel model

This script processes audio files and extracts all necessary features:
1. Mel-spectrogram (ground truth, optional)
2. Semantic codes (from semantic_codec)
3. Speaker style embeddings (from campplus)
4. GPT latent features

NO TEXT NEEDED - this is purely audio-to-audio distillation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from omegaconf import OmegaConf

from indextts.infer_vllm_v2 import IndexTTS2


class DistillationDataPreprocessor:
    def __init__(self, model_dir, device='cuda'):
        """
        Initialize preprocessor with IndexTTS2 models
        """
        print(f"Loading IndexTTS2 models from {model_dir}...")
        # Load models for feature extraction
        # We only need the feature extractors, not the full inference pipeline
        cfg_path = os.path.join(model_dir, "config.yaml")
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.device = device

        # Initialize only the components we need
        from transformers import SeamlessM4TFeatureExtractor
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.audio import mel_spectrogram
        import safetensors

        # 1. Semantic feature extractor (w2v-bert)
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            os.path.join(model_dir, "w2v-bert-2.0")
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(model_dir, self.cfg.w2v_stat),
            os.path.join(model_dir, "w2v-bert-2.0")
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)

        # 2. Semantic codec (for quantization)
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(model_dir, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(device).eval()

        # 3. Speaker style extractor (campplus)
        campplus_ckpt_path = os.path.join(model_dir, "campplus/campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(device).eval()

        # 4. Mel-spectrogram function
        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        print("✓ All models loaded successfully")

    @torch.no_grad()
    def get_semantic_embedding(self, audio_16k):
        """Extract semantic embedding from audio"""
        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    @torch.no_grad()
    def process_single_audio(self, audio_path, output_dir):
        """
        Process a single audio file and extract all features

        Returns:
            dict with all preprocessed features
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)

        # Resample to required rates
        audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio_22k = torch.tensor(audio_22k).unsqueeze(0)
        audio_16k = torch.tensor(audio_16k).unsqueeze(0)

        # 1. Mel-spectrogram (ground truth)
        mel = self.mel_fn(audio_22k.to(self.device).float())

        # 2. Semantic embedding and codes
        semantic_emb = self.get_semantic_embedding(audio_16k)
        _, semantic_codes = self.semantic_codec.quantize(semantic_emb)

        # Decode codes to embedding (what s2mel model receives)
        S_infer = self.semantic_codec.quantizer.vq2emb(semantic_codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)

        # 3. Speaker style embedding
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        # 4. Split into prompt and target
        # Use first ~3 seconds as prompt
        mel_len = mel.size(2)
        prompt_len = min(795, mel_len // 3)  # ~3 seconds at 22.05kHz

        prompt_mel = mel[:, :, :prompt_len]

        # Prepare output
        basename = Path(audio_path).stem
        output_data = {
            'mel': mel.cpu(),                      # [1, 80, T]
            'semantic_codes': semantic_codes.cpu(), # [1, T']
            'S_infer': S_infer.cpu(),              # [1, T', 1024]
            'style': style.cpu(),                   # [1, 192]
            'prompt_mel': prompt_mel.cpu(),        # [1, 80, prompt_len]
            'prompt_len': prompt_len,
            'target_length': mel_len,
            'audio_path': audio_path,
        }

        # Save to disk
        save_path = os.path.join(output_dir, f"{basename}.pt")
        torch.save(output_data, save_path)

        return output_data

    def process_dataset(self, audio_dir, output_dir, max_files=None):
        """
        Process entire audio dataset

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save preprocessed data
            max_files: Maximum number of files to process (for testing)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(Path(audio_dir).rglob(ext))

        if max_files:
            audio_files = audio_files[:max_files]

        print(f"Found {len(audio_files)} audio files")
        print(f"Output directory: {output_dir}")

        # Process each file
        failed = []
        for audio_path in tqdm(audio_files, desc="Processing audio"):
            try:
                self.process_single_audio(str(audio_path), output_dir)
            except Exception as e:
                print(f"Failed to process {audio_path}: {e}")
                failed.append(str(audio_path))

        print(f"\n✓ Processing complete!")
        print(f"  Successful: {len(audio_files) - len(failed)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            fail_log = os.path.join(output_dir, "failed_files.txt")
            with open(fail_log, 'w') as f:
                f.write('\n'.join(failed))
            print(f"  Failed files logged to: {fail_log}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for s2mel progressive distillation")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing training audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preprocessed features")
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM",
                        help="Path to IndexTTS-2 model directory")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = DistillationDataPreprocessor(
        model_dir=args.model_dir,
        device=args.device
    )

    # Process dataset
    preprocessor.process_dataset(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()
