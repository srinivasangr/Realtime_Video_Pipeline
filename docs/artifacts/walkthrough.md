# Video Understanding Pipeline - Fixes & Updates Walkthrough

## Overview
This document summarizes the changes made to stabilize the Kaggle notebooks, add requested features (History), and resolve errors across multiple versions.

## 1. Notebook Versions & Status

| Notebook | Path | Status | Key Features |
| :--- | :--- | :--- | :--- |
| **V5 (Stable)** | `notebooks/video_pipeline_kaggle_v5.ipynb` | ✅ **Recommended** | Single-GPU, FP32 Moondream, Qwen/SmolVLM fixes, History, Robust |
| **V2 (Legacy)** | `version1/video-pipeline-kaggleversion2.ipynb` | ✅ **Updated** | Fixed UI (Single Model), Added History Feature |
| **Kaggle2** | `version1/video-pipeline-kaggle2.ipynb` | ✅ **Patched** | Fixed `AttributeError`, Updated Deps (git+transformers), Added History, Robust Loading |

## 2. Key Fixes Applied

### A. Moondream2 Stability (`AttributeError` & `!!!!!!`)
- **Issue**: Moondream2 caused `AttributeError: ...mark_tied_weights_as_initialized` and output `!!!!!!` on T4 GPUs.
- **Fix**: 
    - Applied a safe monkey-patch to `transformers.PreTrainedModel`.
    - Forced `torch_dtype=torch.float32` for Moondream2 loading (fixes instability/garbage output).

### B. Dependency Hell (Qwen/SmolVLM)
- **Issue**: `ImportError` for `PreTrainedConfig` (SmolVLM) and `is_offline_mode` (Qwen).
- **Fix**: Updated installation cells to use the latest `transformers` from GitHub:
  ```bash
  !pip install -q git+https://github.com/huggingface/transformers.git accelerate
  ```

### C. History Feature Port
- **Request**: User wanted the "Analysis History" log in older notebooks.
- **Action**: Injected the History UI HTML/JS and backend API endpoints into `v2` and `kaggle2` notebooks.

### D. UI Cleanup (V2)
- **Issue**: V2 notebook UI showed options for 4 models, but backend only supported Moondream2.
- **Fix**: Simplified V2 UI to only show Moondream2, matching its actual capability.

## 3. How to Use
1.  **Download** the desired notebook from the artifacts folder.
2.  **Upload** to Kaggle.
3.  **Run All Cells**.
4.  **Launch UI** via the ngrok link printed in the final cell.

## 4. Verification
- **V5**: Verified with full pipeline test (mock).
- **V2/Kaggle2**: Verified code patches for syntax and logic correctness.

## Artifacts Saved
The following files have been saved to the artifacts directory:
- `video_pipeline_kaggle_v5.ipynb`
- `video-pipeline-kaggleversion2.ipynb`
- `video-pipeline-kaggle2.ipynb`
