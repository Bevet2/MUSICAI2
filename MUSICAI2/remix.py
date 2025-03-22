"""
Remix module for MUSICAI2.
Handles song transformation between different genres using trained models.
"""

import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
import json
from demucs.pretrained import get_model
from demucs.apply import apply_model
import librosa
from ..data.youtube import YouTubeAPI
from ..training.model_trainer import MusicGenreModel


class RemixEngine:
    def __init__(
        self,
        models_dir: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize remix engine.
        
        Args:
            models_dir: Directory containing trained genre models
            device: Device to use for processing
        """
        self.models_dir = Path(models_dir)
        self.device = device
        self.youtube = YouTubeAPI()
        
        # Load model information
        info_path = self.models_dir / 'model_info.json'
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
            
        # Load genre models
        self.genre_models = {}
        for genre, model_path in self.model_info['models'].items():
            self.genre_models[genre] = self._load_genre_model(model_path)
            
        # Initialize Demucs for source separation
        self.separator = get_model('htdemucs')
        self.separator.to(device)

    def _load_genre_model(self, model_path: str) -> MusicGenreModel:
        """Load a trained genre model.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with same parameters as training
        model = MusicGenreModel(
            n_mels=self.model_info['training_args']['n_mels']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model

    def separate_sources(
        self,
        audio_path: str,
        output_dir: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate audio into vocals and instrumental.
        
        Args:
            audio_path: Path to audio file
            output_dir: Optional directory to save separated sources
            
        Returns:
            Tuple of (vocals, instrumental) as tensors
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Apply source separation
        sources = apply_model(
            self.separator,
            audio,
            device=self.device,
            progress=True
        )
        
        # Extract vocals and instrumental
        vocals = sources[0]  # First source is vocals
        instrumental = sum(sources[1:])  # Remaining sources form instrumental
        
        # Save if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torchaudio.save(
                str(output_dir / 'vocals.wav'),
                vocals.cpu(),
                sr
            )
            torchaudio.save(
                str(output_dir / 'instrumental.wav'),
                instrumental.cpu(),
                sr
            )
            
        return vocals, instrumental

    def transform_instrumental(
        self,
        instrumental: torch.Tensor,
        target_genre: str,
        segment_length: int = 65536
    ) -> torch.Tensor:
        """Transform instrumental to target genre.
        
        Args:
            instrumental: Input instrumental as tensor
            target_genre: Target genre for transformation
            segment_length: Length of segments to process
            
        Returns:
            Transformed instrumental as tensor
        """
        if target_genre not in self.genre_models:
            raise ValueError(f"No model available for genre: {target_genre}")
            
        model = self.genre_models[target_genre]
        
        # Process in segments to handle long audio
        segments = []
        for i in range(0, instrumental.size(1), segment_length):
            # Extract segment
            end = min(i + segment_length, instrumental.size(1))
            segment = instrumental[:, i:end]
            
            # Pad if needed
            if segment.size(1) < segment_length:
                segment = torch.nn.functional.pad(
                    segment,
                    (0, segment_length - segment.size(1))
                )
            
            # Convert to mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=44100,
                n_fft=2048,
                hop_length=512,
                n_mels=self.model_info['training_args']['n_mels']
            ).to(self.device)
            
            mel_spec = mel_transform(segment)
            mel_spec = torch.log(mel_spec + 1e-9)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
            
            # Transform through model
            with torch.no_grad():
                transformed_spec = model(mel_spec.unsqueeze(0))[0]
            
            # Convert back to audio using Griffin-Lim
            inverse_mel = torchaudio.transforms.InverseMelScale(
                n_stft=1025,
                n_mels=self.model_info['training_args']['n_mels']
            ).to(self.device)
            
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=2048,
                hop_length=512
            ).to(self.device)
            
            # Reconstruct audio
            stft = inverse_mel(torch.exp(transformed_spec))
            audio_segment = griffin_lim(stft)
            
            segments.append(audio_segment)
            
        # Concatenate segments
        return torch.cat(segments, dim=0)

    def remix_song(
        self,
        input_path: str,
        target_genre: str,
        output_path: Optional[str] = None,
        temp_dir: Optional[str] = None
    ) -> Optional[str]:
        """Transform a song into target genre.
        
        Args:
            input_path: Path to input audio file
            target_genre: Target genre for transformation
            output_path: Optional path for output file
            temp_dir: Optional directory for temporary files
            
        Returns:
            Path to remixed song if output_path provided
        """
        if temp_dir:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Separate sources
            vocals, instrumental = self.separate_sources(
                input_path,
                temp_dir
            )
            
            # Transform instrumental
            transformed = self.transform_instrumental(
                instrumental,
                target_genre
            )
            
            # Mix with original vocals
            mixed = vocals + transformed.unsqueeze(0)
            
            # Normalize
            mixed = mixed / mixed.abs().max()
            
            # Save if output path provided
            if output_path:
                torchaudio.save(output_path, mixed.cpu(), 44100)
                return output_path
                
            return None
            
        except Exception as e:
            print(f"Error remixing song: {e}")
            return None

    def remix_from_youtube(
        self,
        youtube_url: str,
        target_genre: str,
        output_dir: str
    ) -> Optional[str]:
        """Download and remix a YouTube song.
        
        Args:
            youtube_url: YouTube video URL
            target_genre: Target genre for transformation
            output_dir: Directory for output files
            
        Returns:
            Path to remixed song if successful
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download song
            downloaded_path = self.youtube.download(
                youtube_url,
                str(output_dir)
            )
            
            if not downloaded_path:
                return None
                
            # Create output path
            output_path = output_dir / f"remixed_{target_genre}.mp3"
            
            # Remix song
            return self.remix_song(
                downloaded_path,
                target_genre,
                str(output_path),
                str(output_dir / 'temp')
            )
            
        except Exception as e:
            print(f"Error processing YouTube song: {e}")
            return None
