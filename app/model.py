"""
Moondream2 Vision-Language Model manager.
Handles model loading and per-frame inference.
"""
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
from app.config import MODEL_ID, MODEL_REVISION, DEVICE
from app.video_utils import format_timestamp

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for the Moondream2 model."""

    _instance: Optional["ModelManager"] = None
    _model = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """Load Moondream2 into memory. Call once at startup.
        
        Strategy for low-VRAM GPUs (e.g. MX450 2GB):
        1. Try GPU with float16 (~1.9GB) first
        2. If VRAM too low or OOM, fallback to CPU with float32
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping.")
            return

        from transformers import AutoModelForCausalLM
        import torch

        self._actual_device = DEVICE  # Track what we actually use

        # Try GPU first if available
        if DEVICE == "cuda":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🖥️  GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

                if vram_gb < 1.8:
                    logger.warning(f"⚠️  VRAM ({vram_gb:.1f}GB) too low for float16. Falling back to CPU.")
                    self._actual_device = "cpu"
                else:
                    logger.info(f"Loading model {MODEL_ID} (rev={MODEL_REVISION}) on CUDA (float16)...")
                    torch.cuda.empty_cache()
                    self._model = AutoModelForCausalLM.from_pretrained(
                        MODEL_ID,
                        revision=MODEL_REVISION,
                        trust_remote_code=True,
                        device_map={"": "cuda"},
                        dtype=torch.float16,
                    )
                    self._is_loaded = True
                    used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    logger.info(f"✅ Model loaded on CUDA (float16) — using {used_mb:.0f} MB VRAM")
                    return

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"⚠️  GPU load failed ({e}). Falling back to CPU...")
                torch.cuda.empty_cache()
                self._actual_device = "cpu"

        # CPU fallback
        try:
            logger.info(f"Loading model {MODEL_ID} (rev={MODEL_REVISION}) on CPU (float32)...")
            self._model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                device_map={"": "cpu"},
                dtype=torch.float32,
            )
            self._is_loaded = True
            logger.info(f"✅ Model loaded successfully on CPU")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return getattr(self, '_actual_device', DEVICE)

    def analyze_frames(
        self,
        frames: List[tuple],  # List of (timestamp_seconds, PIL.Image)
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Run Moondream2 inference on each extracted frame.

        Args:
            frames: List of (timestamp_sec, PIL.Image) from video_utils.extract_frames
            prompt: User's natural language query about the video

        Returns:
            List of {"timestamp": "MM:SS", "seconds": float, "description": str}
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        total = len(frames)
        logger.info(f"Analyzing {total} frames with prompt: '{prompt[:80]}...'")

        for idx, (timestamp_sec, pil_image) in enumerate(frames):
            try:
                logger.info(f"  Frame {idx + 1}/{total} @ {format_timestamp(timestamp_sec)}")

                # Moondream2 native API: model.query(image, question)
                answer = self._model.query(pil_image, prompt)

                # The API returns {"answer": "..."} 
                description = answer.get("answer", str(answer)) if isinstance(answer, dict) else str(answer)

                results.append({
                    "timestamp": format_timestamp(timestamp_sec),
                    "seconds": round(timestamp_sec, 2),
                    "description": description.strip(),
                    "frame_index": idx + 1,
                })

                # Clear CUDA cache between frames to prevent OOM on low-VRAM GPUs
                if self._actual_device == "cuda":
                    import torch
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"  Error on frame {idx + 1}: {e}")
                results.append({
                    "timestamp": format_timestamp(timestamp_sec),
                    "seconds": round(timestamp_sec, 2),
                    "description": f"[Error analyzing frame: {str(e)}]",
                    "frame_index": idx + 1,
                })

        logger.info(f"✅ Analysis complete: {len(results)} frames processed")
        return results


# Global singleton
model_manager = ModelManager()
