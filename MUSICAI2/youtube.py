"""
YouTube search and download functionality for MUSICAI2.
Uses yt-dlp for efficient video downloading and processing.
"""

import os
from typing import List, Dict, Optional
import yt_dlp
from pathlib import Path


class YouTubeAPI:
    def __init__(self, output_path: str = "downloads"):
        """Initialize YouTube API wrapper.
        
        Args:
            output_path: Directory to save downloaded files
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for YouTube videos.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of video information dictionaries
        """
        search_opts = {
            **self.ydl_opts,
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(search_opts) as ydl:
            # Add search query parameters
            full_query = f"ytsearch{limit}:{query}"
            try:
                results = ydl.extract_info(full_query, download=False)
                if not results:
                    return []
                
                # Extract relevant information
                videos = []
                for entry in results['entries']:
                    if entry:
                        videos.append({
                            'id': entry.get('id', ''),
                            'title': entry.get('title', ''),
                            'duration': entry.get('duration', 0),
                            'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                            'thumbnail': entry.get('thumbnail', ''),
                        })
                return videos
            except Exception as e:
                print(f"Error searching YouTube: {e}")
                return []

    def download(self, video_url: str, output_path: Optional[str] = None) -> Optional[str]:
        """Download a YouTube video as MP3.
        
        Args:
            video_url: YouTube video URL
            output_path: Optional custom output path
            
        Returns:
            Path to downloaded file or None if failed
        """
        if output_path:
            download_path = Path(output_path)
        else:
            download_path = self.output_path
            
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Configure download options
        download_opts = {
            **self.ydl_opts,
            'outtmpl': str(download_path / '%(title)s.%(ext)s'),
        }

        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(video_url, download=False)
                if not info:
                    return None
                
                # Generate output filename
                filename = ydl.prepare_filename(info)
                mp3_filename = str(Path(filename).with_suffix('.mp3'))
                
                # Download if file doesn't exist
                if not os.path.exists(mp3_filename):
                    ydl.download([video_url])
                
                return mp3_filename
                
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    def search_and_download(self, query: str, limit: int = 1) -> List[str]:
        """Search for videos and download them.
        
        Args:
            query: Search query string
            limit: Maximum number of videos to download
            
        Returns:
            List of paths to downloaded files
        """
        # Search for videos
        videos = self.search(query, limit=limit)
        if not videos:
            return []
            
        # Download each video
        downloaded_files = []
        for video in videos[:limit]:
            if file_path := self.download(video['url']):
                downloaded_files.append(file_path)
                
        return downloaded_files
