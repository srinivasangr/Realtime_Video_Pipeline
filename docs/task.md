# Video Understanding Pipeline MVP — Task Checklist

## Phase 1: Planning
- [x] Research models (chose Moondream2 for local CPU/GPU)
- [x] Design architecture (FastAPI + static frontend)
- [x] Create implementation plan

## Phase 2: Backend
- [x] `app/config.py` — settings, device auto-detection
- [x] `app/video_utils.py` — frame extraction with OpenCV
- [x] `app/model.py` — Moondream2 model manager (singleton)
- [x] `app/main.py` — FastAPI endpoints (`/api/analyze`, `/api/health`)
- [x] `requirements.txt` — dependencies

## Phase 3: Frontend
- [x] `app/static/index.html` — upload, webcam, prompt, results UI
- [x] `app/static/styles.css` — dark theme, glassmorphism, animations

## Phase 4: Extras
- [x] `notebooks/video_pipeline_colab.ipynb` — Colab GPU notebook
- [x] `Readme.md` — full documentation

## Phase 5: Dependency Fixes
- [x] Fix `pyvips` binary installation on Windows
- [x] Fix `cv2` / `einops` import issues
- [x] Add `accelerate` for `device_map` support
- [x] Fix `torch_dtype` → `dtype` deprecation

## Phase 6: GPU Acceleration (MX450 2GB)
- [x] Install CUDA-enabled PyTorch (`torch-2.10.0+cu128`)
- [x] Verify CUDA detection (`True`, build `12.8`)
- [x] Update `model.py` — float16 GPU loading, VRAM check, CPU fallback
- [x] Add CUDA cache clearing between frames
- [x] Fix `total_mem` → `total_memory` attribute bug
- [x] Test GPU inference (~1 min/frame vs ~4 min/frame CPU)

## Phase 7: Analysis History Feature
- [x] `app/history.py` — JSON file-based storage (save/list/get/delete)
- [x] Update `main.py` — auto-save + 3 history endpoints
- [x] Update `index.html` — history panel UI with view/delete
- [x] Update `styles.css` — history card styling
