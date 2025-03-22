"""
Audio preprocessing module for MUSICAI2.
Handles audio normalization, silence trimming, and format standardization.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment


class AudioProcessor:
    def __init__(
        self,
        target_sr: int = 44100,
        target_db: float = -20.0,
        min_silence_len: int = 1000,
        silence_threshold: int = -40
    ):
        """Initialize audio processor.
        
        Args:
            target_sr: Target sample rate in Hz
            target_db: Target dB level for normalization
            min_silence_len: Minimum silence length in ms
            silence_threshold: Silence threshold in dB
        """
        self.target_sr = target_sr
        self.target_db = target_db
        self.min_silence_len = min_silence_len
        self.silence_threshold = silence_threshold

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None

    def normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume to target dB.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        if audio is None or len(audio) == 0:
            return None
            
        # Calculate current dB
        current_db = 20 * np.log10(np.max(np.abs(audio)))
        
        # Calculate adjustment needed
        db_adjustment = self.target_db - current_db
        
        # Apply gain
        return audio * (10 ** (db_adjustment / 20))

    def trim_silence(self, audio_path: str) -> Optional[AudioSegment]:
        """Trim silence from start and end of audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed AudioSegment or None if failed
        """
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Trim silence
            trimmed_audio = audio.strip_silence(
                silence_len=self.min_silence_len,
                silence_thresh=self.silence_threshold,
                padding=100
            )
            
            return trimmed_audio
        except Exception as e:
            print(f"Error trimming silence from {audio_path}: {e}")
            return None

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Process a single audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Optional path for processed file
            
        Returns:
            Path to processed file or None if failed
        """
        if output_path is None:
            output_path = input_path.replace('.mp3', '_processed.mp3')
            
        try:
            # Trim silence first
            trimmed_audio = self.trim_silence(input_path)
            if trimmed_audio is None:
                return None
                
            # Export to temporary file
            temp_path = input_path.replace('.mp3', '_temp.wav')
            trimmed_audio.export(temp_path, format='wav')
            
            # Load with librosa for further processing
            audio, sr = self.load_audio(temp_path)
            if audio is None:
                return None
                
            # Normalize volume
            normalized_audio = self.normalize_volume(audio)
            if normalized_audio is None:
                return None
                
            # Save processed audio
            sf.write(output_path, normalized_audio, self.target_sr)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error processing file {input_path}: {e}")
            return None

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None
    ) -> int:
        """Process all audio files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Optional output directory path
            
        Returns:
            Number of successfully processed files
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        # Process all MP3 files
        for audio_file in input_dir.glob('*.mp3'):
            if output_dir:
                output_path = output_dir / audio_file.name
            else:
                output_path = None
                
            if self.process_file(str(audio_file), str(output_path) if output_path else None):
                processed_count += 1
                
        return processed_count
