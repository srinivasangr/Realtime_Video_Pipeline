"""
Video processing utilities: frame extraction, resizing, and validation.

OpenCV on Windows cannot decode WebM (VP8/VP9) files.
For WebM files (from webcam recording), we use imageio + ffmpeg as a fallback.
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from app.config import FRAME_SAMPLE_INTERVAL, FRAME_MAX_WIDTH, ALLOWED_VIDEO_EXTENSIONS
import os
import logging

logger = logging.getLogger(__name__)


def _is_webm(video_path: str) -> bool:
    """Check if a file is WebM format."""
    return os.path.splitext(video_path)[1].lower() == ".webm"


def _read_with_imageio(video_path: str):
    """Read video using imageio + ffmpeg (handles WebM on Windows)."""
    import imageio.v3 as iio
    return iio.imread(video_path, plugin="pyav")


# ─── Validation ─────────────────────────────────────────────────────

def validate_video(video_path: str) -> Tuple[bool, str]:
    """Validate that a video file can be opened and has readable frames."""
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"

    if _is_webm(video_path):
        # Use imageio for WebM since OpenCV can't handle it on Windows
        try:
            import imageio.v3 as iio
            meta = iio.immeta(video_path, plugin="pyav")
            logger.info(f"WebM validation OK via imageio: duration={meta.get('duration', '?')}s")
            return True, ""
        except Exception as e:
            logger.warning(f"imageio validation failed: {e}")
            # Fall through to OpenCV attempt
    
    # Standard OpenCV validation
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file. It may be corrupted."

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count > 0 and fps > 0:
        cap.release()
        return True, ""

    # Try reading a frame directly
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return True, ""

    return False, "Video has no readable frames."


# ─── Duration ───────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """Return the duration of a video in seconds."""
    if _is_webm(video_path):
        try:
            import imageio.v3 as iio
            meta = iio.immeta(video_path, plugin="pyav")
            duration = float(meta.get("duration", 0))
            if duration > 0:
                logger.info(f"WebM duration via imageio: {duration:.1f}s")
                return duration
        except Exception as e:
            logger.warning(f"imageio duration failed: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return 0.0


# ─── Frame Extraction ──────────────────────────────────────────────

def resize_frame(frame: np.ndarray, max_width: int = FRAME_MAX_WIDTH) -> np.ndarray:
    """Resize a frame to max_width while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _extract_frames_imageio(
    video_path: str,
    interval_sec: float,
    max_width: int,
) -> List[Tuple[float, Image.Image]]:
    """Extract frames from WebM using imageio + ffmpeg."""
    import imageio.v3 as iio

    logger.info(f"Extracting frames via imageio (WebM)...")
    
    # Read metadata to get fps
    meta = iio.immeta(video_path, plugin="pyav")
    fps = float(meta.get("fps", 30.0))
    duration = float(meta.get("duration", 0))
    
    frame_interval = max(1, int(fps * interval_sec))
    
    logger.info(
        f"WebM: fps={fps:.1f}, duration={duration:.1f}s, "
        f"interval={interval_sec}s, frame_interval={frame_interval}"
    )

    results: List[Tuple[float, Image.Image]] = []
    frame_idx = 0
    next_grab = 0

    # Read frames using iterator for memory efficiency
    for frame in iio.imiter(video_path, plugin="pyav"):
        if frame_idx >= next_grab:
            timestamp = frame_idx / fps
            # frame is already RGB from imageio
            frame_resized = resize_frame(frame, max_width)
            pil_image = Image.fromarray(frame_resized)
            results.append((timestamp, pil_image))
            next_grab = frame_idx + frame_interval
        
        frame_idx += 1

    logger.info(f"Extracted {len(results)} frames from WebM via imageio")
    return results


def _extract_frames_opencv(
    video_path: str,
    interval_sec: float,
    max_width: int,
) -> List[Tuple[float, Image.Image]]:
    """Extract frames using OpenCV (for MP4/AVI/MOV)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0
    
    duration = total_frames / fps if total_frames > 0 else 0
    frame_interval = max(1, int(fps * interval_sec))

    logger.info(
        f"Extracting frames: fps={fps:.1f}, duration={duration:.1f}s, "
        f"interval={interval_sec}s, expected_frames={int(duration / interval_sec) + 1 if duration > 0 else '?'}"
    )

    results: List[Tuple[float, Image.Image]] = []
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame = resize_frame(frame, max_width)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        results.append((timestamp, pil_image))
        frame_idx += frame_interval

    cap.release()
    logger.info(f"Extracted {len(results)} frames from video")
    return results


def extract_frames(
    video_path: str,
    interval_sec: float = FRAME_SAMPLE_INTERVAL,
    max_width: int = FRAME_MAX_WIDTH,
) -> List[Tuple[float, Image.Image]]:
    """
    Extract frames from a video at the given interval.
    Uses imageio for WebM files, OpenCV for everything else.
    """
    if _is_webm(video_path):
        try:
            return _extract_frames_imageio(video_path, interval_sec, max_width)
        except Exception as e:
            logger.warning(f"imageio extraction failed: {e}, trying OpenCV...")

    return _extract_frames_opencv(video_path, interval_sec, max_width)


# ─── Helpers ────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
