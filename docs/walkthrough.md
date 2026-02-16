# Video Understanding Pipeline — Walkthrough

## What Was Built
A local video analysis app that uses **Moondream2** (vision-language model) to generate timestamped descriptions of video frames, with a FastAPI backend and browser-based UI.

## Key Files

| File | Purpose |
|------|---------|
| [config.py](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/config.py) | Device detection, limits, sampling config |
| [video_utils.py](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/video_utils.py) | OpenCV frame extraction & validation |
| [model.py](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/model.py) | Moondream2 model manager (GPU/CPU) |
| [main.py](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/main.py) | FastAPI endpoints (analyze, health, history) |
| [history.py](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/history.py) | JSON file-based analysis history storage |
| [index.html](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/static/index.html) | Frontend UI (upload, results, history panel) |
| [styles.css](file:///c:/Users/91944/Downloads/3.Projects%20Portfolio/Projects/Video_Pipeline/app/static/styles.css) | Dark glassmorphism theme |

## What Was Tested

### 1. CPU Inference (Initial)
- Model loaded on CPU (float32, ~3.85 GB RAM)
- **~4-5 min per frame** — 7 frames took ~28 min
- Analysis results were correct (described video content accurately)

### 2. GPU Acceleration (MX450 2GB)
- Installed `torch-2.10.0+cu128` (CUDA-enabled PyTorch)
- CUDA detected: `True`, Build: `12.8`
- Model loaded on CUDA (float16) — reports 3691 MB (spills into shared memory)
- **~1 min per frame** with GPU — **~4x speedup**

> [!WARNING]
> The MX450 has only 2GB VRAM. The model spills into shared system memory, which means performance varies depending on other GPU usage (e.g., browser with GPU acceleration).

### 3. Health Check API
```json
{
    "status": "ok",
    "model": "vikhyatk/moondream2",
    "device": "cuda",
    "model_loaded": true,
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce MX450",
    "max_file_size_mb": 20,
    "max_webcam_seconds": 30
}
```

### 4. Analysis History (New Feature)
- Analyses auto-saved to `history/` as JSON files
- History panel shows past analyses in the UI
- Click to reload results, delete to remove

## How to Run

```powershell
cd "c:\Users\91944\Downloads\3.Projects Portfolio\Projects\Video_Pipeline"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000**

## Performance Summary

| Mode | Per-Frame | 7 Frames | Notes |
|------|-----------|----------|-------|
| CPU (float32) | ~4 min | ~28 min | Baseline |
| GPU MX450 (float16) | ~1 min | ~7 min | 4x faster |
| Colab T4 GPU | ~2-3 sec | ~15 sec | Recommended for heavy use |
