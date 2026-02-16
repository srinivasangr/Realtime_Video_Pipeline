"""
Configuration constants for the Video Understanding Pipeline.
"""
import os
import torch

# ─── File & Recording Limits ───────────────────────────────────────
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 20 MB
MAX_WEBCAM_SECONDS = 30
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

# ─── Frame Extraction ──────────────────────────────────────────────
FRAME_SAMPLE_INTERVAL = 1       # Extract 1 frame per second
FRAME_MAX_WIDTH = 512           # Resize frames to max 512px width
FRAME_JPEG_QUALITY = 85         # JPEG quality for frame encoding

# ─── Model Configuration ───────────────────────────────────────────
MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2025-01-09"   # pinned revision for reproducibility

# Auto-detect best available device
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()

# ─── Server / Paths ────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
