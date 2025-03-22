"""
Script to train individual genre models for MUSICAI2.
"""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model_trainer import AudioDataset, MusicGenreModel, ModelTrainer


def train_genre_model(
    genre: str,
    data_dir: str,
    models_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    segment_length: int = 65536,
    n_mels: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """Train a model for a specific genre.
    
    Args:
        genre: Genre name
        data_dir: Base directory containing processed audio files
        models_dir: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size for training
        segment_length: Length of audio segments
        n_mels: Number of mel bands
        device: Device to use for training
        
    Returns:
        Path to trained model checkpoint
    """
    print(f"\nTraining model for genre: {genre}")
    
    # Setup paths
    genre_data_dir = Path(data_dir) / genre
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = AudioDataset(
        audio_dir=str(genre_data_dir),
        segment_length=segment_length,
        n_mels=n_mels
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model and trainer
    model = MusicGenreModel(n_mels=n_mels)
    trainer = ModelTrainer(model, device=device)
    
    # Training loop
    best_loss = float('inf')
    best_checkpoint = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        loss = trainer.train_epoch(dataloader, epoch)
        
        # Save checkpoint if best so far
        if loss < best_loss:
            best_loss = loss
            checkpoint_path = models_dir / f"{genre}_model_best.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, loss)
            best_checkpoint = checkpoint_path
            print(f"New best model saved with loss: {loss:.6f}")
            
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = models_dir / f"{genre}_model_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, loss)
            
    return str(best_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train music genre models")
    parser.add_argument("--data", "-d", default="F:/processed_datasets",
                      help="Base directory containing processed audio files")
    parser.add_argument("--models", "-m", default="models",
                      help="Directory to save trained models")
    parser.add_argument("--genre", "-g", required=True,
                      help="Genre to train model for")
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
    
    # Train model
    best_model = train_genre_model(
        genre=args.genre,
        data_dir=args.data,
        models_dir=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        segment_length=args.segment_length,
        n_mels=args.n_mels,
        device=args.device
    )
    
    print(f"\nTraining completed. Best model saved at: {best_model}")


if __name__ == "__main__":
    main()
