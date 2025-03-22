"""
Script to train models for all genres in MUSICAI2.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict
import json
from train_models import train_genre_model


def train_all_genres(
    data_dir: str,
    models_dir: str,
    genres: List[str],
    epochs: int = 100,
    batch_size: int = 32,
    segment_length: int = 65536,
    n_mels: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, str]:
    """Train models for all specified genres.
    
    Args:
        data_dir: Base directory containing processed audio files
        models_dir: Directory to save trained models
        genres: List of genres to train models for
        epochs: Number of training epochs
        batch_size: Batch size for training
        segment_length: Length of audio segments
        n_mels: Number of mel bands
        device: Device to use for training
        
    Returns:
        Dictionary mapping genres to their best model checkpoints
    """
    models = {}
    
    for genre in genres:
        print(f"\n{'='*50}")
        print(f"Training model for genre: {genre}")
        print(f"{'='*50}")
        
        try:
            best_model = train_genre_model(
                genre=genre,
                data_dir=data_dir,
                models_dir=models_dir,
                epochs=epochs,
                batch_size=batch_size,
                segment_length=segment_length,
                n_mels=n_mels,
                device=device
            )
            
            models[genre] = best_model
            print(f"Successfully trained model for {genre}")
            
        except Exception as e:
            print(f"Error training model for {genre}: {e}")
            continue
            
    return models


def main():
    parser = argparse.ArgumentParser(description="Train all music genre models")
    parser.add_argument("--data", "-d", default="F:/processed_datasets",
                      help="Base directory containing processed audio files")
    parser.add_argument("--models", "-m", default="models",
                      help="Directory to save trained models")
    parser.add_argument("--genres", "-g", nargs="+",
                      default=["pop", "rock", "jazz", "classical", "electronic"],
                      help="List of genres to train models for")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--segment-length", "-s", type=int, default=65536,
                      help="Audio segment length in samples")
    parser.add_argument("--n-mels", "-n", type=int, default=128,
                      help="Number of mel bands")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu',
                      help="Device to use for training")
    
    args = parser.parse_args()
    
    # Train all models
    models = train_all_genres(
        data_dir=args.data,
        models_dir=args.models,
        genres=args.genres,
        epochs=args.epochs,
        batch_size=args.batch_size,
        segment_length=args.segment_length,
        n_mels=args.n_mels,
        device=args.device
    )
    
    # Save model paths to JSON
    model_info = {
        'models': models,
        'training_args': vars(args)
    }
    
    info_path = Path(args.models) / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
        
    print(f"\nTraining completed. Model information saved to: {info_path}")


if __name__ == "__main__":
    main()
