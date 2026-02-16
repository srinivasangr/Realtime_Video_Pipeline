# Multi-Model Video Pipeline — Kaggle Notebook

4 features to add to `video_pipeline_kaggle.ipynb`, implemented one at a time.

## Feature 1: Model Selection UI + Multi-Model Loading

### Models & Loading Code

| Model | HF ID | Class | VRAM (fp16) |
|-------|--------|-------|-------------|
| Moondream2 | `vikhyatk/moondream2` | `AutoModelForCausalLM` + `trust_remote_code` | ~4 GB |
| SmolVLM | `HuggingFaceTB/SmolVLM-Instruct` | `AutoModelForVision2Seq` + `AutoProcessor` | ~4 GB |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` + `qwen_vl_utils` | ~6 GB |
| Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` | `Qwen3VLForConditionalGeneration` + `AutoProcessor` (needs transformers ≥4.57) | ~4 GB |

### Approach: Lazy Loading (one model at a time)
- **NOT** loading all 4 models simultaneously (would need ~18 GB, exceeding single T4's 15 GB)
- Load selected model on demand, unload previous model to free VRAM
- UI sends `model` parameter with `/api/analyze` request
- Backend swaps models when selection changes

### UI Changes
- Add a **model selector dropdown** in the Input panel (below upload, above prompt)
- 4 options with model names + size labels
- Selected model shown in results metadata

### Notebook Cell Changes

#### Cell 1 (Install): Add `qwen_vl_utils` package
#### Cell 3 (Model Loading): Replace single model load with a `ModelManager` class
```python
class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model = None
        self.processor = None
    
    def load(self, model_name, device="cuda:0"):
        if model_name == self.current_model_name:
            return  # already loaded
        self.unload()
        # load based on model_name...
    
    def unload(self):
        del self.model, self.processor
        torch.cuda.empty_cache()
    
    def query(self, image, prompt):
        # unified interface dispatching to model-specific code
```
#### Cell 4 (Video Utils): `analyze_video` uses `manager.query()` 
#### UI Cell: Add `<select>` dropdown
#### Server Cell: `/api/analyze` accepts `model` form field

---

## Feature 2: Dual GPU Utilization

### Approach: Parallel Frame Processing
- Load same model on **both** GPUs (mirror)
- Split frames: even frames → GPU 0, odd frames → GPU 1
- Use `ThreadPoolExecutor` for concurrent inference
- **2× throughput** for analysis

> [!IMPORTANT]
> This only works if each model fits in a single T4 (15 GB). Qwen2.5-VL-3B at ~6 GB fits easily. We'd load two copies.

### Alternative if model is too large
- Load model on GPU 0, use GPU 1 for pre-processing (resize/convert)
- Less speedup but still offloads work

---

## Feature 3: Real-Time Streaming Results

### Approach: Server-Sent Events (SSE)
- New endpoint: `POST /api/analyze-stream` → SSE stream
- Each frame result sent as a `data:` event immediately
- UI uses `EventSource` or `fetch` with `ReadableStream` to display results as they arrive
- Each frame result card appears with animation as it comes in

---

## Feature 4 (Bonus): 100% GPU Utilization
- Use `torch.cuda.amp.autocast` for mixed precision
- Batch pre-processing: extract ALL frames first, then feed to model
- Minimize CPU↔GPU transfer overhead
- Pin memory for frame tensors

---

## Execution Order
1. **Feature 1** — Model Selection + Loading (this PR)
2. **Feature 3** — Real-time streaming (natural next step, improves UX)
3. **Feature 2** — Dual GPU (performance boost)
4. **Feature 4** — GPU optimization (final polish)

## Verification Plan
- After each feature, restart Kaggle kernel and run all cells
- Test each model selection works
- Verify VRAM usage stays within 15 GB per GPU
- Test streaming updates appear frame-by-frame in browser
