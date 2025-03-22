"""
Model trainer module for MUSICAI2.
Handles audio feature extraction and model training.
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        segment_length: int = 65536,  # ~1.5 seconds at 44.1kHz
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """Initialize audio dataset.
        
        Args:
            audio_dir: Directory containing audio files
            segment_length: Length of audio segments in samples
            hop_length: Hop length for spectrogram
            n_mels: Number of mel bands
        """
        self.audio_dir = Path(audio_dir)
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # List all audio files
        self.audio_files = list(self.audio_dir.glob('*.mp3'))
        
        # Setup mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random segment from an audio file.
        
        Args:
            idx: Index of audio file
            
        Returns:
            Tuple of (input_features, target_features)
        """
        # Load audio file
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Get random segment
        if waveform.size(1) > self.segment_length:
            start = torch.randint(0, waveform.size(1) - self.segment_length, (1,))
            segment = waveform[:, start:start + self.segment_length]
        else:
            # Pad if too short
            segment = torch.nn.functional.pad(
                waveform,
                (0, self.segment_length - waveform.size(1))
            )
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(segment)
        
        # Log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        return mel_spec, mel_spec  # Input and target are the same for autoencoder


class MusicGenreModel(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        hidden_dims: List[int] = [256, 512, 1024]
    ):
        """Initialize music genre model.
        
        Args:
            n_mels: Number of mel bands
            hidden_dims: List of hidden dimensions
        """
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        in_channels = 1  # Mono audio
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        hidden_dims.reverse()
        
        for i, h_dim in enumerate(hidden_dims[1:], 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    hidden_dims[i-1],
                    h_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            
        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(
                hidden_dims[-1],
                1,  # Output channels
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed output tensor
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize model trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
                
        return total_loss / len(dataloader)

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        loss: float
    ):
        """Save model checkpoint.
        
        Args:
            save_path: Path to save checkpoint
            epoch: Current epoch number
            loss: Current loss value
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
