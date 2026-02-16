# 🎥 Real-Time Video Understanding using Moondream2

Cloud-hosted / local multimodal video analysis pipeline using an open-source Vision-Language Model.

## 🚀 What It Does

Upload a video (or record from webcam) → provide a prompt → get **timestamped AI analysis** of every frame.

**Example Output:**
```
[00:01] A person is standing in a kitchen holding a wooden spoon
[00:02] The person is stirring a pot on the stove
[00:03] Close-up of a bubbling red sauce in the pot
```

## 🏗️ Architecture

```
User (Browser)
  ↓
Upload Video + Prompt
  ↓
FastAPI Backend
  ↓
Frame Extraction (OpenCV, 1 frame/sec)
  ↓
Resize to 512px width
  ↓
Moondream2 VLM Inference
  ↓
Aggregate Timestamped Results
  ↓
Return JSON → Display in UI
```

## 🖥️ Run Locally

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster inference

### Setup
```bash
# Clone and enter project
cd Video_Pipeline

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Open the UI
Navigate to **http://localhost:8000** in your browser.

## ☁️ Run on Google Colab / Kaggle (Free GPU)

1. Open `notebooks/video_pipeline_colab.ipynb` in Colab or Kaggle
2. Set runtime to **T4 GPU** (Colab) or **P100 GPU** (Kaggle)
3. Run all cells — the model loads and you can analyze videos directly!

## 📦 Project Structure

```
Video_Pipeline/
├── app/
│   ├── main.py              # FastAPI routes
│   ├── model.py             # Moondream2 loading & inference
│   ├── video_utils.py       # Frame extraction & processing
│   ├── config.py            # Configuration
│   └── static/
│       ├── index.html       # Frontend UI
│       └── styles.css       # Dark theme styling
├── notebooks/
│   └── video_pipeline_colab.ipynb  # Colab/Kaggle notebook
├── requirements.txt
└── README.md
```

## 🧱 Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | Moondream2 (2B params, ~2.3GB VRAM quantized) |
| Backend | FastAPI + Uvicorn |
| Video Processing | OpenCV |
| Frontend | HTML/CSS/JS (dark theme, glassmorphism) |
| Deployment | Local / Colab / Kaggle |

## 🎯 Constraints

| Constraint | Limit | Reason |
|-----------|-------|--------|
| Upload size | 20 MB | Prevent OOM + reasonable latency |
| Webcam recording | 30 sec | Keep frame count manageable |
| Frame sampling | 1/sec | Balance coverage vs. speed |
| Frame resize | 512px | Reduce VRAM per frame |

## ⏱️ Latency

| Device | Per Frame | 10 Frames |
|--------|-----------|-----------|
| T4 GPU | ~0.5-1s | ~5-10s |
| CPU | ~3-7s | ~30-70s |

## 🔮 Future Improvements

- Real-time WebSocket streaming
- GPU batching for multiple frames
- Frame change detection (skip duplicate frames)
- RAG integration for entity knowledge lookup
- Docker containerization
- Kubernetes autoscaling