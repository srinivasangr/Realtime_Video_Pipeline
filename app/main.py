"""
FastAPI application for the Video Understanding Pipeline.
Serves the frontend and exposes the /api/analyze endpoint.
"""
import os
import uuid
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_WEBCAM_SECONDS,
    UPLOAD_DIR,
    STATIC_DIR,
    DEVICE,
    MODEL_ID,
    FRAME_SAMPLE_INTERVAL,
)
from app.video_utils import validate_video, extract_frames, get_video_duration
from app.model import model_manager
from app.history import save_analysis, list_analyses, get_analysis, delete_analysis

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload the model at startup."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("🎥 Video Understanding Pipeline — Starting Up")
    logger.info(f"   Device : {DEVICE}")
    logger.info(f"   Model  : {MODEL_ID}")
    logger.info("=" * 60)

    model_manager.load_model()

    yield

    logger.info("Shutting down...")


# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Video Understanding Pipeline",
    description="Upload a video + prompt → AI-powered timestamped analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health Check ───────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check endpoint with system info."""
    import torch

    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "model_loaded": model_manager.is_loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "max_webcam_seconds": MAX_WEBCAM_SECONDS,
    }


# ─── Analyze Video ──────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    prompt: str = Form(..., description="What to analyze in the video"),
):
    """
    Upload a video file + prompt → returns timestamped frame-by-frame analysis.
    """
    start_time = time.time()

    # ── 1. Validate file size ───────────────────────────────────────
    # Read content to check size (FastAPI doesn't expose content-length reliably)
    content = await video.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB. "
                   f"Your file is {len(content) / (1024*1024):.1f}MB.",
        )

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # ── 2. Save temp file ───────────────────────────────────────────
    file_ext = os.path.splitext(video.filename or "video.mp4")[1].lower()
    temp_filename = f"{uuid.uuid4().hex}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        logger.info(f"📁 Saved upload: {temp_filename} ({len(content)/1024:.0f} KB)")

        # ── 3. Validate video ───────────────────────────────────────
        is_valid, error_msg = validate_video(temp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid video: {error_msg}")

        duration = get_video_duration(temp_path)
        logger.info(f"📹 Video duration: {duration:.1f}s")

        # ── 4. Extract frames ───────────────────────────────────────
        frames = extract_frames(temp_path, interval_sec=FRAME_SAMPLE_INTERVAL)
        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted from the video.")

        # ── 5. Run inference ────────────────────────────────────────
        results = model_manager.analyze_frames(frames, prompt.strip())

        elapsed = time.time() - start_time
        logger.info(f"✅ Analysis complete in {elapsed:.1f}s")

        # Save to history
        original_name = video.filename or "video.mp4"
        analysis_id = save_analysis(
            video_filename=original_name,
            prompt=prompt.strip(),
            duration=duration,
            frames_analyzed=len(results),
            processing_time=elapsed,
            device=model_manager.device,
            results=results,
        )

        return {
            "success": True,
            "id": analysis_id,
            "prompt": prompt.strip(),
            "video_duration_seconds": round(duration, 2),
            "frames_analyzed": len(results),
            "processing_time_seconds": round(elapsed, 2),
            "results": results,
        }

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"🗑️  Cleaned up: {temp_filename}")


# ─── History Endpoints ──────────────────────────────────────────────
@app.get("/api/history")
async def get_history():
    """List all saved analyses (summaries only)."""
    return {"analyses": list_analyses()}


@app.get("/api/history/{analysis_id}")
async def get_history_item(analysis_id: str):
    """Get a single saved analysis with full results."""
    record = get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


@app.delete("/api/history/{analysis_id}")
async def delete_history_item(analysis_id: str):
    """Delete a saved analysis."""
    if delete_analysis(analysis_id):
        return {"success": True, "message": "Analysis deleted"}
    raise HTTPException(status_code=404, detail="Analysis not found")


# ─── Serve Frontend ─────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        content="<h1>Video Pipeline</h1><p>Frontend not found. Place index.html in app/static/</p>",
        status_code=200,
    )


# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
