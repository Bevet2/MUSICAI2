# 🎵 MUSICAI2

An AI-powered platform for music transformation and creation.

## 🎯 Overview
MUSICAI2 allows users to:
- Transform songs into different musical genres using AI
- Create new music by blending multiple tracks
- Generate AI vocals with custom lyrics
- Search and process YouTube songs directly

## 🧱 Project Structure
```
musicai2/
├── datasets/           # Training data organized by genre
├── models/            # Trained AI models
├── src/
│   ├── data/         # Dataset collection and processing
│   │   ├── youtube.py
│   │   ├── get_top_tracks.py
│   │   ├── audio_processor.py
│   │   └── collect_datasets.py
│   ├── training/     # Model training scripts
│   │   ├── model_trainer.py
│   │   ├── train_models.py
│   │   └── train_all.py
│   ├── remix/        # Genre transformation
│   │   └── remix.py
│   ├── creation/     # Music creation and blending
│   │   ├── track_mixer.py
│   │   └── voice_synthesizer.py
│   └── web/          # Web interface
│       ├── static/
│       │   ├── css/
│       │   │   └── style.css
│       │   └── js/
│       │       └── app.js
│       ├── templates/
│       │   └── index.html
│       └── api.py
└── requirements.txt   # Project dependencies
```

## 🛠️ Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Features
- **Genre Transformation**: Convert songs into different musical styles
- **Music Creation**: Blend multiple tracks and add AI vocals
- **YouTube Integration**: Search and process songs directly
- **Web Interface**: User-friendly access to all features

## 🚀 Usage
1. Start the web server:
```bash
python src/web/api.py
```

2. Open `http://localhost:5000` in your browser
3. Search for a song or create new music!
