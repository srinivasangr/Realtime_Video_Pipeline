# Video Pipeline — Multi-Model Features

## Feature 1: Model Selection UI + Multi-Model Loading
- [x] Update install cell to add `qwen_vl_utils` and upgrade transformers
- [x] Create `ModelManager` class with lazy load/unload/query
- [x] Implement model-specific inference for all 4 models
- [x] Update `analyze_video` to use `ModelManager`
- [x] Add model selector dropdown to UI HTML
- [x] Update `/api/analyze` to accept `model` parameter
- [ ] Test all 4 models load and produce output (re-testing with V5)

## Feature 2: Dual GPU Utilization (On Hold - Stability Issues)
- [x] Load model copies on both GPUs
- [x] Split frames across GPUs with ThreadPoolExecutor
- [x] Merge results in timestamp order
- [-] Verify both GPUs show utilization (reverted to single-GPU in V5)

## Feature 5: Stability & Bug Fixes (V5)
- [x] Fix Moondream2 `!!!!!!` output (FP32 precision)
- [x] Fix `Incorrect image source` for Qwen/SmolVLM (absolute paths)
- [x] Revert to Single-GPU for stability
- [x] Fix JSON syntax in v5 notebook

## Feature 3: Real-Time Streaming
- [x] Add SSE endpoint `POST /api/analyze-stream`
- [x] Stream each frame result as SSE event
- [x] Update frontend to use EventSource/ReadableStream
- [x] Results appear one-by-one with animation
- [x] Restore analysis history UI

## Feature 4: GPU Optimization
- [x] Enable mixed precision with autocast (via inference_mode/tf32)
- [x] Batch frame pre-processing (or optimized generate)
- [x] Pin memory for GPU transfers (implicit via optimizations)
- [ ] Benchmark and verify utilization (user testing)

## Ad-hoc Tasks (Legacy Versions)
## Ad-hoc Tasks (Legacy Versions)
- [x] Port 'Analysis History' UI from `v1` to `v2` notebook (`version1/video-pipeline-kaggleversion2.ipynb`)
- [x] Fix `AttributeError` and add History to `version1/video-pipeline-kaggle2.ipynb` (User Request)
- [x] Clean up UI in `v2` to remove unused models

## Final Step
- [x] Save all modified notebooks to artifacts directory
