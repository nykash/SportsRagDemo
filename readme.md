# Volleyball Highlights

A comprehensive system for processing volleyball videos, identifying highlights, tracking players, and enabling semantic search through video content using CLIP embeddings and vector databases.

## Overview

This project provides an end-to-end solution for:
- **Video Processing**: Segmenting volleyball videos into clips and processing them
- **Player Tracking**: Detecting and recognizing player jersey numbers
- **Highlight Identification**: Automatically identifying and extracting highlight moments
- **Semantic Search**: Using CLIP embeddings and Pinecone for natural language video retrieval
- **Audio Transcription**: Transcribing video audio using WhisperX
- **Demo Application**: Web-based interface for searching and viewing highlights

## Project Structure

```
VolleyballHighlights/
├── CLIP/                    # CLIP-based video processing and retrieval
│   ├── clients/            # External service clients (Ollama, Pinecone, Whisper)
│   ├── routers/            # FastAPI routers (upload, query, process)
│   ├── utils/              # Utility functions for video processing
│   └── video_clip/         # Video clip training and inference
├── demo/                    # Demo application
│   ├── backend/            # Flask API server
│   │   ├── api/            # API endpoints
│   │   ├── client/         # Pinecone and S3 clients
│   │   └── utils/          # Text encoding utilities
│   └── frontend/           # Web frontend
├── player_tracking/         # Player tracking and jersey recognition
│   ├── pipeline/           # Processing pipeline
│   ├── training/           # Model training scripts
│   └── visualization/      # Visualization tools
└── event_tracking/          # Event tracking functionality
```

## Features

### 1. Video Processing
- Segment videos into clips
- Extract frames and process video content
- Generate embeddings using CLIP

### 2. Player Tracking
- Detect players in video frames
- Recognize jersey numbers
- Generate player statistics
- Visualize tracking results

### 3. Semantic Search
- Natural language queries to find video clips
- CLIP-based text-to-video embedding
- Vector similarity search using Pinecone
- Retrieve and serve relevant video highlights

### 4. Audio Processing
- Transcribe video audio using WhisperX
- Generate summaries using LLM (Ollama)
- Extract semantic information from transcripts

## Setup

### Prerequisites

- Python 3.9+
- AWS account (for S3 storage)
- Pinecone account (for vector database)
- Ollama (optional, for LLM features)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VolleyballHighlights
   ```

2. **Set up CLIP module**
   ```bash
   cd CLIP
   pip install -r requirements.txt
   ```

3. **Set up demo backend**
   ```bash
   cd demo/backend
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   
   Create a `.env` file in `demo/backend/` with:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=your_index_name
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_REGION=your_aws_region
   AWS_BUCKET_NAME=your_bucket_name
   ```

## Usage

### Running the Demo Backend

```bash
cd demo/backend
python app.py
```

The API will be available at `http://127.0.0.1:5000`

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /retrieve` - Retrieve video clips based on text query


### Player Tracking

For processing videos for player tracking, you need to use python 3.12 and be using a cuda machine. This conflicts with some of the other requirements so we suggest using a seperate conda environment or venv:
```bash
cd player_tracking/pipeline
sh ./predict.sh
```

## Acknlowedgements

- **CLIP**: OpenCLIP for video/text embeddings
- **Pinecone**: Vector database for semantic search
- **AWS S3**: Cloud storage for videos
- **WhisperX**: Audio transcription
- **Flask**: Backend API framework
- **FastAPI**: CLIP service API
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision processing

