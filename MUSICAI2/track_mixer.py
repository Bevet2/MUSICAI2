"""
Track mixer module for MUSICAI2.
Handles blending multiple audio tracks and mixing with vocals.
"""

import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from typing import List, Optional, Tuple
from ..remix.remix import RemixEngine


class TrackMixer:
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize track mixer.
        
        Args:
            device: Device to use for processing
        """
        self.device = device
        self.remix_engine = RemixEngine(models_dir='models', device=device)

    def load_and_normalize(
        self,
        audio_path: str,
        target_sr: int = 44100
    ) -> Tuple[torch.Tensor, int]:
        """Load and normalize audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Tuple of (normalized_audio, sample_rate)
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Normalize
        audio = audio / audio.abs().max()
        
        return audio, target_sr

    def align_lengths(
        self,
        tracks: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Align tracks to same length.
        
        Args:
            tracks: List of audio tensors
            
        Returns:
            List of aligned audio tensors
        """
        # Find max length
        max_length = max(track.size(1) for track in tracks)
        
        # Pad shorter tracks
        aligned_tracks = []
        for track in tracks:
            if track.size(1) < max_length:
                padding = max_length - track.size(1)
                track = torch.nn.functional.pad(track, (0, padding))
            aligned_tracks.append(track)
            
        return aligned_tracks

    def crossfade(
        self,
        track1: torch.Tensor,
        track2: torch.Tensor,
        fade_length: int = 44100  # 1 second
    ) -> torch.Tensor:
        """Apply crossfade between two tracks.
        
        Args:
            track1: First audio track
            track2: Second audio track
            fade_length: Length of fade in samples
            
        Returns:
            Crossfaded audio
        """
        # Create fade curves
        fade_out = torch.linspace(1, 0, fade_length)
        fade_in = torch.linspace(0, 1, fade_length)
        
        # Apply fades
        track1_end = track1[:, -fade_length:] * fade_out
        track2_start = track2[:, :fade_length] * fade_in
        
        # Combine
        crossfade = track1_end + track2_start
        
        # Join parts
        result = torch.cat([
            track1[:, :-fade_length],
            crossfade,
            track2[:, fade_length:]
        ], dim=1)
        
        return result

    def apply_effects(
        self,
        audio: torch.Tensor,
        reverb_amount: float = 0.3,
        eq_bands: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Apply audio effects.
        
        Args:
            audio: Input audio tensor
            reverb_amount: Amount of reverb to apply (0-1)
            eq_bands: List of gain values for frequency bands
            
        Returns:
            Processed audio
        """
        # Apply reverb
        if reverb_amount > 0:
            # Simple convolution reverb
            decay = torch.exp(-torch.arange(44100) / (44100 * reverb_amount))
            reverb = torch.nn.functional.conv1d(
                audio,
                decay.view(1, 1, -1).to(audio.device),
                padding=44100
            )
            audio = audio + reverb_amount * reverb
            
        # Apply EQ if provided
        if eq_bands:
            # Simple multi-band EQ
            n_bands = len(eq_bands)
            band_width = 22050 // n_bands  # Nyquist frequency / number of bands
            
            # Apply gain to each frequency band
            fft = torch.fft.rfft(audio, dim=1)
            freqs = torch.fft.rfftfreq(audio.size(1), 1/44100)
            
            for i, gain in enumerate(eq_bands):
                band_start = i * band_width
                band_end = (i + 1) * band_width
                mask = (freqs >= band_start) & (freqs < band_end)
                fft[:, mask] *= gain
                
            audio = torch.fft.irfft(fft, dim=1)
            
        return audio

    def blend_tracks(
        self,
        track_paths: List[str],
        crossfade_duration: float = 1.0,
        reverb_amount: float = 0.3,
        eq_bands: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Blend multiple tracks together.
        
        Args:
            track_paths: List of paths to audio files
            crossfade_duration: Duration of crossfade in seconds
            reverb_amount: Amount of reverb to apply
            eq_bands: List of gain values for frequency bands
            
        Returns:
            Blended audio tensor
        """
        # Load and normalize all tracks
        tracks = []
        for path in track_paths:
            audio, _ = self.load_and_normalize(path)
            tracks.append(audio)
            
        # Align track lengths
        tracks = self.align_lengths(tracks)
        
        # Apply crossfading between consecutive tracks
        fade_samples = int(crossfade_duration * 44100)
        result = tracks[0]
        
        for next_track in tracks[1:]:
            result = self.crossfade(
                result,
                next_track,
                fade_samples
            )
            
        # Apply effects
        result = self.apply_effects(
            result,
            reverb_amount,
            eq_bands
        )
        
        # Final normalization
        result = result / result.abs().max()
        
        return result

    def create_mashup(
        self,
        youtube_urls: List[str],
        output_path: str,
        temp_dir: str = 'temp',
        crossfade_duration: float = 1.0,
        reverb_amount: float = 0.3,
        eq_bands: Optional[List[float]] = None
    ) -> Optional[str]:
        """Create mashup from YouTube tracks.
        
        Args:
            youtube_urls: List of YouTube video URLs
            output_path: Path for output file
            temp_dir: Directory for temporary files
            crossfade_duration: Duration of crossfade in seconds
            reverb_amount: Amount of reverb to apply
            eq_bands: List of gain values for frequency bands
            
        Returns:
            Path to output file if successful
        """
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download and separate instrumentals
            instrumental_paths = []
            
            for i, url in enumerate(youtube_urls):
                # Download track
                track_path = self.remix_engine.youtube.download(
                    url,
                    str(temp_dir)
                )
                
                if not track_path:
                    continue
                    
                # Separate and keep instrumental
                _, instrumental = self.remix_engine.separate_sources(
                    track_path,
                    str(temp_dir / f'track_{i}')
                )
                
                instrumental_path = temp_dir / f'track_{i}_instrumental.wav'
                torchaudio.save(str(instrumental_path), instrumental.cpu(), 44100)
                instrumental_paths.append(str(instrumental_path))
                
            if not instrumental_paths:
                return None
                
            # Blend instrumentals
            blended = self.blend_tracks(
                instrumental_paths,
                crossfade_duration,
                reverb_amount,
                eq_bands
            )
            
            # Save result
            torchaudio.save(output_path, blended.cpu(), 44100)
            
            return output_path
            
        except Exception as e:
            print(f"Error creating mashup: {e}")
            return None
        finally:
            # Clean up temp files
            if temp_dir.exists():
                for file in temp_dir.glob('*'):
                    try:
                        file.unlink()
                    except:
                        pass
