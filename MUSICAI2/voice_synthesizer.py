"""
Voice synthesizer module for MUSICAI2.
Handles text-to-speech synthesis for vocal generation.
"""

import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from typing import Optional, List, Dict
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import json


class VoiceSynthesizer:
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_name: str = "microsoft/speecht5_tts"
    ):
        """Initialize voice synthesizer.
        
        Args:
            device: Device to use for processing
            model_name: Name of pretrained model to use
        """
        self.device = device
        
        # Load models and processor
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        
        # Load speaker embeddings for different voice styles
        self.voice_styles = self._load_voice_styles()

    def _load_voice_styles(self) -> Dict[str, torch.Tensor]:
        """Load speaker embeddings for different voice styles.
        
        Returns:
            Dictionary mapping style names to speaker embeddings
        """
        # Default voice styles (can be expanded)
        styles = {
            'neutral': torch.randn(1, 512).to(self.device),  # Base neutral voice
            'warm': torch.randn(1, 512).to(self.device),     # Warm, smooth voice
            'bright': torch.randn(1, 512).to(self.device),   # Bright, energetic voice
            'deep': torch.randn(1, 512).to(self.device),     # Deep, resonant voice
        }
        
        # Modify embeddings to create distinct characteristics
        styles['warm'] = styles['neutral'] * 0.8 + torch.randn(1, 512).to(self.device) * 0.2
        styles['bright'] = styles['neutral'] * 0.7 + torch.randn(1, 512).to(self.device) * 0.3
        styles['deep'] = styles['neutral'] * 0.6 + torch.randn(1, 512).to(self.device) * 0.4
        
        return styles

    def preprocess_text(
        self,
        text: str,
        max_length: int = 600
    ) -> torch.Tensor:
        """Preprocess text for synthesis.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Preprocessed input ids
        """
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True
        )
        
        return inputs["input_ids"].to(self.device)

    def generate_speech(
        self,
        text: str,
        voice_style: str = 'neutral',
        speed: float = 1.0,
        max_length: int = 600
    ) -> Optional[torch.Tensor]:
        """Generate speech from text.
        
        Args:
            text: Input text
            voice_style: Name of voice style to use
            speed: Speech speed multiplier
            max_length: Maximum sequence length
            
        Returns:
            Generated speech as tensor
        """
        try:
            # Get speaker embedding for style
            if voice_style not in self.voice_styles:
                print(f"Warning: Unknown voice style '{voice_style}', using 'neutral'")
                voice_style = 'neutral'
                
            speaker_embeddings = self.voice_styles[voice_style]
            
            # Preprocess text
            input_ids = self.preprocess_text(text, max_length)
            
            # Generate speech
            with torch.no_grad():
                output = self.model.generate_speech(
                    input_ids,
                    speaker_embeddings,
                    vocoder=self.vocoder
                )
                
            # Adjust speed if needed
            if speed != 1.0:
                # Simple speed adjustment using interpolation
                old_length = output.size(0)
                new_length = int(old_length / speed)
                
                output = torch.nn.functional.interpolate(
                    output.view(1, 1, -1),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                )[0, 0]
                
            return output
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None

    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: str,
        sample_rate: int = 16000
    ) -> bool:
        """Save audio tensor to file.
        
        Args:
            audio: Audio tensor
            output_path: Output file path
            sample_rate: Sample rate
            
        Returns:
            True if successful
        """
        try:
            torchaudio.save(
                output_path,
                audio.unsqueeze(0).cpu(),
                sample_rate
            )
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False

    def generate_vocals(
        self,
        lyrics: str,
        output_path: str,
        voice_style: str = 'neutral',
        speed: float = 1.0
    ) -> Optional[str]:
        """Generate vocals from lyrics.
        
        Args:
            lyrics: Input lyrics text
            output_path: Output file path
            voice_style: Name of voice style to use
            speed: Speech speed multiplier
            
        Returns:
            Path to output file if successful
        """
        try:
            # Generate speech
            audio = self.generate_speech(
                lyrics,
                voice_style,
                speed
            )
            
            if audio is None:
                return None
                
            # Save to file
            if self.save_audio(audio, output_path):
                return output_path
                
            return None
            
        except Exception as e:
            print(f"Error generating vocals: {e}")
            return None

    def generate_vocal_sections(
        self,
        sections: List[Dict],
        output_dir: str
    ) -> List[str]:
        """Generate vocals for multiple sections.
        
        Args:
            sections: List of section dictionaries with 'lyrics', 'style', and 'speed'
            output_dir: Output directory for vocal files
            
        Returns:
            List of paths to generated vocal files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vocal_files = []
        
        for i, section in enumerate(sections):
            output_path = output_dir / f"vocals_{i}.wav"
            
            if generated_path := self.generate_vocals(
                lyrics=section['lyrics'],
                output_path=str(output_path),
                voice_style=section.get('style', 'neutral'),
                speed=section.get('speed', 1.0)
            ):
                vocal_files.append(generated_path)
                
        return vocal_files
