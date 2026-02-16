"""
Simple JSON-file based history storage for analysis results.
Each analysis is saved as a separate JSON file in the history/ directory.
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history")


def _ensure_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def save_analysis(
    video_filename: str,
    prompt: str,
    duration: float,
    frames_analyzed: int,
    processing_time: float,
    device: str,
    results: List[Dict[str, Any]],
) -> str:
    """Save an analysis result and return its ID."""
    _ensure_dir()
    analysis_id = uuid.uuid4().hex[:12]
    record = {
        "id": analysis_id,
        "video_filename": video_filename,
        "prompt": prompt,
        "video_duration_seconds": round(duration, 2),
        "frames_analyzed": frames_analyzed,
        "processing_time_seconds": round(processing_time, 2),
        "device": device,
        "created_at": datetime.now().isoformat(),
        "results": results,
    }

    filepath = os.path.join(HISTORY_DIR, f"{analysis_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Saved analysis {analysis_id} ({video_filename})")
    return analysis_id


def list_analyses() -> List[Dict[str, Any]]:
    """List all saved analyses (summary only, without full results)."""
    _ensure_dir()
    analyses = []

    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        filepath = os.path.join(HISTORY_DIR, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                record = json.load(f)
            # Return summary without the full results array
            analyses.append({
                "id": record["id"],
                "video_filename": record.get("video_filename", "Unknown"),
                "prompt": record.get("prompt", ""),
                "video_duration_seconds": record.get("video_duration_seconds", 0),
                "frames_analyzed": record.get("frames_analyzed", 0),
                "processing_time_seconds": record.get("processing_time_seconds", 0),
                "device": record.get("device", "unknown"),
                "created_at": record.get("created_at", ""),
            })
        except Exception as e:
            logger.warning(f"Skipping invalid history file {fname}: {e}")

    # Sort by date, newest first
    analyses.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return analyses


def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get a single analysis by ID (full results included)."""
    filepath = os.path.join(HISTORY_DIR, f"{analysis_id}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_analysis(analysis_id: str) -> bool:
    """Delete a saved analysis by ID."""
    filepath = os.path.join(HISTORY_DIR, f"{analysis_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"🗑️  Deleted analysis {analysis_id}")
        return True
    return False
