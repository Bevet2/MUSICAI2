"""
Dataset organization script for MUSICAI2.
Processes and organizes audio files by genre.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
from audio_processor import AudioProcessor


def organize_dataset(
    input_dir: str,
    output_dir: str,
    genres: List[str],
    process_audio: bool = True
) -> dict:
    """Organize and process audio files by genre.
    
    Args:
        input_dir: Base input directory containing genre subdirectories
        output_dir: Base output directory for processed datasets
        genres: List of genres to process
        process_audio: Whether to apply audio processing
        
    Returns:
        Dictionary with statistics per genre
    """
    input_base = Path(input_dir)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Initialize audio processor if needed
    processor = AudioProcessor() if process_audio else None
    
    stats = {}
    
    for genre in genres:
        print(f"\nProcessing {genre} dataset...")
        
        # Setup paths
        input_genre_dir = input_base / genre
        output_genre_dir = output_base / genre
        output_genre_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_genre_dir.exists():
            print(f"Warning: Input directory for {genre} not found")
            continue
            
        # Process files
        total_files = 0
        processed_files = 0
        failed_files = 0
        
        for audio_file in input_genre_dir.glob('*.mp3'):
            total_files += 1
            output_path = output_genre_dir / audio_file.name
            
            try:
                if process_audio:
                    # Process and save
                    if processor.process_file(str(audio_file), str(output_path)):
                        processed_files += 1
                    else:
                        failed_files += 1
                else:
                    # Just copy
                    shutil.copy2(audio_file, output_path)
                    processed_files += 1
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                failed_files += 1
                
        # Save statistics
        stats[genre] = {
            'total': total_files,
            'processed': processed_files,
            'failed': failed_files
        }
        
        print(f"Completed {genre}:")
        print(f"  Total files: {total_files}")
        print(f"  Successfully processed: {processed_files}")
        print(f"  Failed: {failed_files}")
        
    return stats


def validate_dataset(
    dataset_dir: str,
    min_duration: float = 30.0,
    max_duration: float = 600.0
) -> List[Path]:
    """Validate processed audio files.
    
    Args:
        dataset_dir: Directory containing processed files
        min_duration: Minimum valid duration in seconds
        max_duration: Maximum valid duration in seconds
        
    Returns:
        List of invalid files
    """
    import librosa
    
    invalid_files = []
    dataset_path = Path(dataset_dir)
    
    for audio_file in dataset_path.rglob('*.mp3'):
        try:
            duration = librosa.get_duration(path=str(audio_file))
            
            # Check duration
            if duration < min_duration or duration > max_duration:
                invalid_files.append(audio_file)
                print(f"Invalid duration ({duration:.1f}s): {audio_file}")
                
        except Exception as e:
            print(f"Error validating {audio_file}: {e}")
            invalid_files.append(audio_file)
            
    return invalid_files


def main():
    parser = argparse.ArgumentParser(description="Organize and process music datasets")
    parser.add_argument("--input", "-i", default="F:/datasets",
                      help="Base input directory containing genre subdirectories")
    parser.add_argument("--output", "-o", default="F:/processed_datasets",
                      help="Output directory for processed datasets")
    parser.add_argument("--genres", "-g", nargs="+",
                      default=["pop", "rock", "jazz", "classical", "electronic"],
                      help="List of genres to process")
    parser.add_argument("--no-processing", action="store_true",
                      help="Skip audio processing (just organize files)")
    parser.add_argument("--validate", action="store_true",
                      help="Validate processed files after organization")
    
    args = parser.parse_args()
    
    # Organize datasets
    stats = organize_dataset(
        input_dir=args.input,
        output_dir=args.output,
        genres=args.genres,
        process_audio=not args.no_processing
    )
    
    # Validate if requested
    if args.validate:
        print("\nValidating processed datasets...")
        invalid_files = validate_dataset(args.output)
        if invalid_files:
            print(f"\nFound {len(invalid_files)} invalid files")
        else:
            print("\nAll files passed validation")


if __name__ == "__main__":
    main()
