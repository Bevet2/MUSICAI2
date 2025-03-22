"""
Flask backend for MUSICAI2 web interface.
Handles API endpoints for song processing and music creation.
"""

import os
from pathlib import Path
from typing import List, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
from ..remix.remix import RemixEngine
from ..creation.track_mixer import TrackMixer
from ..creation.voice_synthesizer import VoiceSynthesizer
from ..data.youtube import YouTubeAPI

app = Flask(__name__)
CORS(app)

# Initialize components
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEMP_FOLDER = 'temp'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

youtube = YouTubeAPI()
remix_engine = RemixEngine(models_dir='models')
track_mixer = TrackMixer()
voice_synth = VoiceSynthesizer()


@app.route('/api/search', methods=['POST'])
def search_youtube():
    """Search YouTube for songs."""
    data = request.json
    query = data.get('query', '')
    limit = data.get('limit', 10)
    
    try:
        results = youtube.search(query, limit=limit)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get available genre models."""
    try:
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        return jsonify({'genres': list(model_info['models'].keys())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remix', methods=['POST'])
def remix_song():
    """Remix a song to a different genre."""
    data = request.json
    youtube_url = data.get('youtube_url')
    target_genre = data.get('target_genre')
    
    if not youtube_url or not target_genre:
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Create unique output directory
        output_dir = Path(OUTPUT_FOLDER) / str(hash(youtube_url))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process song
        output_path = remix_engine.remix_from_youtube(
            youtube_url,
            target_genre,
            str(output_dir)
        )
        
        if not output_path:
            return jsonify({'error': 'Failed to process song'}), 500
            
        return jsonify({
            'success': True,
            'output_path': output_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/create', methods=['POST'])
def create_song():
    """Create a new song from multiple tracks and vocals."""
    data = request.json
    youtube_urls = data.get('youtube_urls', [])
    lyrics = data.get('lyrics', '')
    voice_style = data.get('voice_style', 'neutral')
    
    if not youtube_urls:
        return jsonify({'error': 'No YouTube URLs provided'}), 400
        
    try:
        # Create unique output directory
        output_dir = Path(OUTPUT_FOLDER) / str(hash(''.join(youtube_urls)))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create instrumental mashup
        instrumental_path = str(output_dir / 'instrumental.wav')
        mashup_path = track_mixer.create_mashup(
            youtube_urls,
            instrumental_path,
            str(Path(TEMP_FOLDER) / str(hash(''.join(youtube_urls))))
        )
        
        if not mashup_path:
            return jsonify({'error': 'Failed to create instrumental'}), 500
            
        # Generate vocals if lyrics provided
        if lyrics:
            vocals_path = str(output_dir / 'vocals.wav')
            vocals_path = voice_synth.generate_vocals(
                lyrics,
                vocals_path,
                voice_style
            )
            
            if vocals_path:
                # Mix vocals with instrumental
                final_path = str(output_dir / 'final.wav')
                
                # Load both audio files
                instrumental, sr = track_mixer.load_and_normalize(mashup_path)
                vocals, _ = track_mixer.load_and_normalize(vocals_path)
                
                # Mix with proper levels
                mixed = instrumental * 0.7 + vocals * 0.3
                mixed = mixed / mixed.abs().max()
                
                # Save final mix
                import torchaudio
                torchaudio.save(final_path, mixed.cpu(), sr)
                
                output_path = final_path
            else:
                output_path = mashup_path
        else:
            output_path = mashup_path
            
        return jsonify({
            'success': True,
            'output_path': output_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<path:filename>')
def download_file(filename):
    """Download a processed audio file."""
    try:
        return send_file(
            filename,
            as_attachment=True,
            download_name=os.path.basename(filename)
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/voice-styles', methods=['GET'])
def get_voice_styles():
    """Get available voice styles."""
    try:
        styles = voice_synth.voice_styles.keys()
        return jsonify({'styles': list(styles)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
