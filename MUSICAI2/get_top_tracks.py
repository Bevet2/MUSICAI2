"""
Script to automatically collect top songs for each genre from YouTube.
Uses the YouTubeAPI class to search and download songs.
"""

import os
from pathlib import Path
from typing import List, Optional
import argparse
from youtube import YouTubeAPI


def get_genre_dataset(
    genre: str,
    output_dir: str,
    num_songs: int = 1000,
    batch_size: int = 50
) -> List[str]:
    """Download top songs for a specific genre.
    
    Args:
        genre: Music genre to search for
        output_dir: Base directory for saving downloads
        num_songs: Total number of songs to download
        batch_size: Number of songs to search for in each batch
        
    Returns:
        List of paths to downloaded files
    """
    # Create genre-specific output directory
    genre_dir = Path(output_dir) / genre
    genre_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize YouTube API
    yt = YouTubeAPI(output_path=str(genre_dir))
    
    # Search queries to try
    search_queries = [
        f"top {genre} songs",
        f"best {genre} music",
        f"popular {genre} hits",
        f"{genre} classics",
        f"{genre} essentials"
    ]
    
    downloaded_files = []
    current_query_idx = 0
    
    # Keep downloading until we reach the target number
    while len(downloaded_files) < num_songs and current_query_idx < len(search_queries):
        query = search_queries[current_query_idx]
        print(f"\nSearching for: {query}")
        
        # Calculate how many songs we still need
        remaining = num_songs - len(downloaded_files)
        current_batch = min(batch_size, remaining)
        
        # Search and download batch
        new_files = yt.search_and_download(
            query=query,
            limit=current_batch
        )
        
        # Add successfully downloaded files
        downloaded_files.extend(new_files)
        print(f"Downloaded {len(new_files)} songs. Total: {len(downloaded_files)}/{num_songs}")
        
        # Move to next query if this one didn't yield enough results
        if len(new_files) < current_batch:
            current_query_idx += 1
            
    return downloaded_files


def main():
    parser = argparse.ArgumentParser(description="Download top songs by genre from YouTube")
    parser.add_argument("--output", "-o", default="F:/datasets",
                      help="Base directory to save downloaded songs")
    parser.add_argument("--num-songs", "-n", type=int, default=1000,
                      help="Number of songs to download per genre")
    parser.add_argument("--genres", "-g", nargs="+",
                      default=["pop", "rock", "jazz", "classical", "electronic"],
                      help="List of genres to download")
    parser.add_argument("--batch-size", "-b", type=int, default=50,
                      help="Number of songs to search for in each batch")
    
    args = parser.parse_args()
    
    # Process each genre
    for genre in args.genres:
        print(f"\n{'='*50}")
        print(f"Processing genre: {genre}")
        print(f"{'='*50}")
        
        downloaded = get_genre_dataset(
            genre=genre,
            output_dir=args.output,
            num_songs=args.num_songs,
            batch_size=args.batch_size
        )
        
        print(f"\nCompleted {genre}: {len(downloaded)} songs downloaded")


if __name__ == "__main__":
    main()
