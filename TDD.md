# Akatsuki — Technical Design Document (TDD)

> Living changelog for all major design decisions and code changes.
> Most recent entries first.
> By dreamraster · dreaMSCend

---

## How to read this document

Each entry carries:
- **Date** — when the change landed
- **Area** — file(s) affected
- **Problem** — what was broken or missing
- **Decision** — what was chosen and why (including alternatives rejected)
- **Files changed**

---

## Changelog

---

### 2026-03-29 — VLM GRPO crash: `lm_head` float32/bfloat16 dtype mismatch

**Area:** `hmlcore/trainer.py`

**Problem:** Qwen2-VL (and other VLMs) crashed during GRPO with:
```
RuntimeError: expected scalar type Float but found BFloat16
File "modeling_qwen2_vl.py", line 1480, in forward
    logits = self.lm_head(hidden_states[:, slice_indices, :])
```
Root cause: `prepare_model_for_kbit_training` aggressively upcasts all non-quantized parameters (including `lm_head`, `embed_tokens`, LoRA A/B matrices) to float32 for training stability. TRL's GRPO generation step runs **outside the autocast context**, so float32 `lm_head` receives bfloat16 hidden states → dtype crash.

The existing fix in `model.py` (non-Unsloth path only) runs at load time. Unsloth's `get_peft_model` and gradient checkpointing setup can re-upcast parameters afterwards, so the `model.py` fix is insufficient for Unsloth models or any re-upcasting that occurs post-load.

**Fix:** Added a dtype sweep in `run_grpo()`, executed just before `GRPOTrainer(...)` is constructed:
```python
_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
_n_fixed = sum(1 for p in model.parameters() if p.dtype == torch.float32)
if _n_fixed:
    for _p in model.parameters():
        if _p.dtype == torch.float32:
            _p.data = _p.data.to(_compute_dtype)
    logger.info("  ⚙️  Re-cast %d float32 param(s) → %s before GRPO (lm_head dtype fix).",
                _n_fixed, _compute_dtype)
```
This covers all code paths (Unsloth + standard PEFT) and any re-upcasting that occurs between model load and GRPO start (e.g., during SFT or Unsloth's gradient checkpointing setup).

**Files changed:** `hmlcore/trainer.py` — `run_grpo()`: float32→compute_dtype sweep before `GRPOTrainer` init

---

### 2026-03-28 — Multimodal GRPO and Spatial Interaction Rewards

**Area:** `hmlcore/nodes/grpo_node.py`, `hmlcore/data.py`, `hmlcore/rewards.py`, `hmlcore/model.py`, `vlm_scene_builder.py`

**Problem:** The pipeline previously skipped GRPO training for VLMs because of text-only rollout failures. There was also no specialized reward logic for 2D spatial tasks (placement, connectivity).

**Decision:**
1. **Multimodal GRPO**: Removed the `is_multimodal` skip in `GRPONode`. Updated `data.py` to preserve image data in `raw_messages`, allowing the VLM processor to render vision tokens during rollouts.
2. **Spatial Reward System**: Added a `scene` domain to `rewards.py`.
    - `check_spatial_precision`: Euclidean distance reward using $5.0 \cdot e^{-dist/100}$ to provide a smooth gradient for coordinate accuracy.
    - `check_scene_connectivity`: Connectivity validation looking for specific object IDs in the model's JSON output.
3. **VRAM Optimization**: To support 4GB VRAM hardware, `model.py` was updated to target VLM projection modules (`m_proj`, `v_proj`) and enforce strict 4-bit quantization via Unsloth.
4. **Synthetic Data**: Created `vlm_scene_builder.py` as a standalone utility to generate randomized 2D environment scenes for training bootstrapping.

**Files changed:** `hmlcore/nodes/grpo_node.py`, `hmlcore/data.py`, `hmlcore/rewards.py`, `hmlcore/model.py`, `vlm_scene_builder.py`

---

---

### 2026-03-26 — GGUF export limited to BF16/F16; dynamic quant types missing

**Area:** `hmlcore/config.py`, `hmlcore/nodes/output_node.py`, `hmlcore/nodes/pruner_node.py`, `hmlcore/dense_pruner.py`

**Problem 1 — `_GGUF_QUANTS` / `--quantize` choices too narrow:**
Only `{"f16", "q8_0", "q4_k"}` recognized. Passing `--quantize iq1_s` was rejected by argparse. Even if it got through, it wasn't in `_GGUF_QUANTS` → OutputNode fell to `_safe_save_pretrained` (HF bf16), never calling `save_pretrained_gguf`.

**Fix 1:** Expanded both to all common Unsloth quant types: `f16`, `q8_0`, `q6_k`, `q5_k_m`, `q5_k`, `q4_k_m`, `q4_k`, `q3_k_m`, `q2_k`, `iq4_xs`, `iq3_xxs`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq1_s`, `iq1_m`.

**Problem 2 — No per-tensor GGUF quantization for dynamicquant:**
`save_pretrained_gguf` applies one quant type to all tensors. Binary-quantized layers get `f16`/`q4_k` just like important ones — pre-quantization is wasted at GGUF stage.

**Fix 2 — Dynamic GGUF guidance + tracking chain:**
- `drop_dense_layers` now returns `(model, drop_indices | None)` instead of just `model`.
- `pruner_node.py` captures `(model, quant_indices)` and stores `quant_indices` → `ctx["quantized_layers"]`; MoE path stores `quant_info` → `ctx["quantized_experts"]`.
- `output_node.py` calls `_log_dynamic_gguf_guidance()` after every save when `--dynamicquant` is active.
- `_log_dynamic_gguf_guidance()` prints ready-to-run `llama-quantize` commands with `--tensor-type blk.{i}.*.weight=q2_k` overrides for pre-quantized layers, `base_quant` for everything else. Uses `q2_k` (no imatrix needed) instead of `iq1_s` (requires imatrix). Checks `hidden_size % 256` alignment.

**Files changed:** `hmlcore/config.py`, `hmlcore/nodes/output_node.py`, `hmlcore/nodes/pruner_node.py`, `hmlcore/dense_pruner.py`

---

### 2026-03-26 — Calibration fails on Unsloth-patched models; hook-based fallback

**Area:** `hmlcore/dense_pruner.py`

**Problem:** `AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'` on every call variant during pre-flight probe for Qwen2 models trained through Unsloth. Root cause: Unsloth replaces `Qwen2Attention.forward` with a custom kernel that calls `self.apply_qkv` (a fused QKV method). This method is registered on the attention object during Unsloth's full model setup. When we call the decoder layer directly (bypassing `model.forward`), the fused kernel is called but `apply_qkv` was never registered on the standalone attention object → `AttributeError`. Because every call variant hits the same patched attention `forward`, all 15+ variants fail identically.

**Fix:** Added `_compute_layer_importance_via_hooks()`. Instead of calling layers directly, it:
1. Registers `register_forward_pre_hook` (captures `h_in = args[0]`) and `register_forward_hook` (captures `h_out = output[0]`) on every decoder block.
2. Runs the real `model.forward(input_ids, use_cache=False)` — Unsloth's patched attention runs normally through the full model pipeline.
3. Computes ShortGPT importance from the captured `(h_in, h_out)` pairs per layer.
4. Always removes hooks in a `finally` block.

Hooks are compatible with all Unsloth patches, BnB quantization, and standard HF models. When the pre-flight probe fails (any reason), `_compute_layer_importance` now calls `_compute_layer_importance_via_hooks` instead of returning index-order fallback scores.

**Files changed:** `hmlcore/dense_pruner.py` — `_compute_layer_importance_via_hooks()` (new); probe failure path switches to it instead of index fallback

---

### 2026-03-26 — `quantize_linear_1bit` not persisting; `Qwen2DecoderLayer` probe silent failure

**Area:** `hmlcore/quant.py`, `hmlcore/dense_pruner.py`

**Problem 1 — `binary_rows=0.0%` for all quantized layers:**
Two bugs in the original `quantize_linear_1bit`:
1. `linear.weight.data = w_bin` — creates a new Python-level storage reference but does NOT modify the existing memory that `state_dict()`, the model graph, and optimizer point to. The quantized values were visible inside the function but vanished after return. (Layer 21's MLP worked by fluke — its weight parameter was a leaf with no grad and no alias.)
2. `torch.sign(0) = 0` — NF4 quantized weights have exactly-zero values (0 is one of the 16 NF4 levels). `sign(0) * scale = 0`, making rows have values `{0, ±scale}` instead of `{±scale}`. The `_binary_row_fraction` check sees spread > 0% and reports 0% binary.

**Fix 1:** Use `linear.weight.data.copy_(w_bin)` (in-place write to existing storage) and `torch.where(w >= 0, ones, -ones)` (no zeros). All computation in float32 before casting back to avoid bf16 precision loss during `mean(dim=-1)`.

**Problem 2 — `Qwen2DecoderLayer` probe fails silently; no diagnostic about WHY:**
`_call_layer` swallowed all exceptions silently — the probe error message showed only "Could not call layer with any known signature" with no indication of what was actually failing. Could be RoPE missing, BnB kernel error, Unsloth patch incompatibility, etc.

Additional issue: `_call_rotary_emb` used a hardcoded `(hidden, position_ids)` signature that may fail for some transformers versions. And the `rotary_emb` lookup didn't include PEFT-wrapped paths like `base_model.model.model.rotary_emb`.

**Fix 2:**
- `_call_layer` now captures `_last_exc` and `_last_kwargs`; `RuntimeError` message includes `last_kwargs` key names and `last_error` type+message.
- Pre-flight probe now imports `traceback` and logs the full stack trace when the probe fails.
- Added `_call_rotary_emb()` helper that tries 4 signatures: `(x, pos_ids)`, `(x, position_ids=pos_ids)`, `(x, seq_len=N)`, `(x)`. Returns `(cos, sin)` or None. Used in both probe and calibration loop.
- Added `base_model.model.model.rotary_emb` and `base_model.model.rotary_emb` to rotary_emb lookup paths (PEFT LoRA wrapping).
- Rotary probe now logs a `⚠️` warning (instead of silently ignoring) when all rotary_emb signatures fail.

**Files changed:**
- `hmlcore/quant.py` — `quantize_linear_1bit`: `.data.copy_()` + `torch.where` + float32 computation
- `hmlcore/dense_pruner.py` — `_call_layer`: captures last exception; `_compute_layer_importance`: `_call_rotary_emb()` helper, extended rotary_emb lookup paths, improved probe error logging with full traceback

---

### 2026-03-25 — GGUF conversion fails after pruning: `bitsandbytes` quant config not stripped

**Area:** `hmlcore/nodes/output_node.py`

**Problem:** After BnB dequantization and pruning, `model.config.quantization_config` still carries `quant_type: "bitsandbytes"`. `convert_hf_to_gguf.py` reads `config.json`, sees this, calls `dequant_model()`, and raises `NotImplementedError: Quant method is not yet supported: 'bitsandbytes'` — because the weights are already plain floats and there is nothing to unpack.

**Fix:** Added `_strip_bnb_config(model)` helper that removes `quantization_config` and `_pre_quantization_dtype` from `model.config` (both via `setattr(None)` and `cfg.__dict__.pop()`).  Called in two places:
1. **At the end of `_dequantize_bnb_model()`** — the canonical place where weights become plain float. Also called if there are no BnB layers but the config still references them (happens when a pre-quantized model is loaded but no BnB layers are actually present after merge).
2. **At the start of `_safe_save_pretrained()`** — safety net for models that bypass `_dequantize_bnb_model` (e.g. originally loaded in float but cloned from a BnB hub checkpoint whose config.json has quantization_config).

**Files changed:** `hmlcore/nodes/output_node.py` — `_strip_bnb_config()` (new), called from `_dequantize_bnb_model` and `_safe_save_pretrained`

---

### 2026-03-25 — `--dynamicquant`: score-guided 1-bit degradation instead of removal

**Area:** `hmlcore/quant.py` (new), `hmlcore/config.py`, `hmlcore/moe.py`, `hmlcore/dense_pruner.py`, `hmlcore/nodes/pruner_node.py`, `hmlcore/nodes/pipeline_check.py`

**Motivation:** Removing low-scored experts/layers is destructive and irreversible. An alternative — inspired by Unsloth Dynamic 2.0 GGUFs — is to keep all experts/layers but assign them drastically different precision levels based on their importance score. Important layers stay at full bf16; redundant ones are degraded to 1-bit (binary) weights. When the model is later exported to GGUF, those already-degraded weights compress to IQ1_S territory with minimal additional KL-divergence penalty.

**Design:**
- `--dynamicquant` is only valid when a pruning option is active (`--prune_experts` / `--prune_only` / `--prune_ratio`). `apply_args` exits with an error otherwise.
- The same REAP/ShortGPT scoring and selection logic is used; only the final action changes: "remove" → "1-bit quantize".
- Quantization is *simulated*: `sign(w) × mean|w_row|` — weights stay in bf16/fp32, no packed integer format needed. The sign pattern carries the effective 1 bit; per-output-neuron scale preserves magnitude.
- MoE: quantizes each low-scoring expert module individually (`quantize_module_1bit`). Stacked 3D tensor layout (Unsloth fused kernels) falls back to removal with a warning.
- Dense: quantizes all Linear layers inside each low-scored decoder block; block is retained in the ModuleList (layer count unchanged).
- `pipeline_check.py` shows `[dynamicquant: 1-bit degrade instead of remove]` on the Pruning line.

**Files changed:**
- `hmlcore/quant.py` — new: `quantize_linear_1bit()`, `quantize_module_1bit()`
- `hmlcore/config.py` — `--dynamicquant` flag + `apply_args` validation
- `hmlcore/moe.py` — `quantize_moe_experts()` + `reap_prune_moe(dynamicquant=)`
- `hmlcore/dense_pruner.py` — `_quantize_dense_layers()` + `drop_dense_layers(dynamicquant=)`
- `hmlcore/nodes/pruner_node.py` — reads `args.dynamicquant`, passes to both pruning calls
- `hmlcore/nodes/pipeline_check.py` — `_compute_stage_plan(dynamicquant=)` + plan label

---

### 2026-03-25 — Layer calibration fails on MixtralDecoderLayer; improved model diagnostics

**Area:** `hmlcore/dense_pruner.py`, `hmlcore/nodes/pipeline_check.py`

**Problem 1 — All `_call_layer` variants fail on Mixtral and transformers ≥ 4.47 architectures:**
transformers ≥ 4.47 refactored RoPE out of decoder layers. `MistralAttention` / `MixtralAttention` now expect pre-computed `position_embeddings=(cos, sin)` passed from outside. Without it: `TypeError: cannot unpack NoneType` on every variant → 128 identical warnings → fallback to index-order pruning.

**Fix 1 — `_call_layer` extended:**
- `position_embeddings` parameter (optional `(cos, sin)` tuple)
- 12 new `position_embeddings` variants (plain, with position_ids/cache_position, with output_router_logits=False for MoE+RoPE)
- 7 new `output_router_logits=False` variants for MoE layers (Mixtral, OLMoE, DeepSeek-MoE)
- `position_embeddings` variants tried FIRST when available

**Fix 2 — `_compute_layer_importance` finds rotary_emb once, computes per sample:**
Searches `model.rotary_emb`, `model.model.rotary_emb`. Computes `(cos, sin)` per sample and passes to `_call_layer`. Pre-flight probe on a fake 16-token input before the 128-sample loop — fails fast with clear ❌ diagnostic and returns fallback scores immediately.

**Problem 2 — `pipeline_check.py` missing VLM detection and library versions:**
**Fix:** `_check_vlm()` detects VLMs via class name, `config.model_type`, vision config attrs, model attrs. `_get_lib_versions()` reports transformers/trl/peft/unsloth versions on every run.

**Files changed:**
- `hmlcore/dense_pruner.py` — `_call_layer` + `_compute_layer_importance`
- `hmlcore/nodes/pipeline_check.py` — `_check_vlm()`, `_get_lib_versions()`, topology label, report

---

### 2026-03-23 — TRL `GRPOConfig` ValueError: `generation_batch_size` + `steps_per_generation` both set

**Area:** `hmlcore/trainer.py`
**Problem:** TRL ≥ 0.14 added `steps_per_generation` to `GRPOConfig`. In some builds the field has a non-None default. `GRPOConfig.__post_init__` also auto-computes `generation_batch_size` from `num_generations`. After the first init, both fields are non-None. `prepare_peft_model()` inside `GRPOTrainer.__init__` then calls `dataclasses.replace(args, gradient_checkpointing=False)` which re-runs `__post_init__` with both values copied → `ValueError: 'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time`.
**Decision (4 iterations):**

1. Pass `steps_per_generation=None` at construction → resolved "both configured" but caused divisibility error: TRL auto-computes `generation_batch_size = per_device_train_batch_size = 1`, which is not divisible by `num_generations=8`.

2. Also pass `generation_batch_size=num_generations*batch_size` → resolved divisibility, but `__post_init__` then auto-computes `steps_per_generation` FROM `generation_batch_size`. Object state after first init: **both non-None again**. `dataclasses.replace` copies both → second `__post_init__` raises the original error again.

3. Clear `grpo_config.steps_per_generation = None` post-construction → `dataclasses.replace` copies `None`, second `__post_init__` recomputes it → init succeeds. But `trainer.args` is the same object (the cleared one), so `trainer.args.steps_per_generation = None` during training → `TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'` in `get_train_dataloader`.

4. **Final fix — `_grpo_config_compat()` context manager:**
   Monkey-patch `GRPOConfig.__post_init__` only for the duration of `GRPOTrainer.__init__`. The patch makes `__post_init__` idempotent: when both fields are non-None (the `dataclasses.replace` re-run case), it clears `steps_per_generation` first so the original logic recomputes it cleanly — no ValueError. Because the patch is scoped to `__init__` only, `trainer.args.steps_per_generation` remains the valid computed value and training proceeds normally.

   `grpo_config` is passed with `steps_per_generation=None, generation_batch_size=N*B` at construction; `__post_init__` computes `steps_per_generation` from `generation_batch_size`. `grpo_config` itself is never mutated post-construction. Compatible with old TRL (field absent → context manager is a no-op).

**Files changed:** `hmlcore/trainer.py` — added `_grpo_config_compat()` context manager; `run_grpo()` wraps `GRPOTrainer(...)` in it; removed post-construction clear

---

### 2026-03-23 — ShortGPT layer call fails after Unsloth GRPO; explicit_reap skips instead of falling back

**Area:** `hmlcore/dense_pruner.py`, `hmlcore/nodes/pruner_node.py`, `tests/test_ohm_pipeline.py`

**Problem 1 — `_call_layer` fails on Unsloth-patched Qwen3/Qwen2 after GRPO training:**
Unsloth patches `Qwen3DecoderLayer.forward` globally during SFT/GRPO training. The patched signature requires `cache_position` (a 1-D `torch.arange` tensor) as a required argument. All 5 existing `_call_layer` variants lacked it, so all calibration samples failed → fallback to index-order pruning (no ShortGPT scoring). Standalone `--prune_only` tests passed because Unsloth has not yet patched the class at that point in execution.

**Fix 1:** Added three new `cache_position` variants to `_call_layer`'s try-list. The correct signature is picked automatically by the existing trial loop.

**Problem 2 — `explicit_reap` guard returned instead of falling back to ShortGPT:**
When `--prune_experts` was passed on a dense model, PrunerNode logged a warning and returned early — pruning was completely skipped. Better UX: warn the user that REAP isn't possible, then run ShortGPT as the best-effort alternative.

**Fix 2:** Removed the `return` from the `explicit_reap` block. Execution falls through to `find_decoder_layers` + `drop_dense_layers` (ShortGPT). Warning message updated.

**Problem 3 — `TestSFTGRPOAndShortGPT` used `--prune_only` (skips training) then asserted `_sft_complete`:**
`--prune_only` explicitly skips SFT+GRPO. The test intent was "full pipeline including pruning after training", so the flag was wrong. Changed to `--prune_experts` which runs SFT+GRPO then triggers ShortGPT via the new fallback.

**Files changed:**
- `hmlcore/dense_pruner.py` — `_call_layer`: added `cache_position` variants
- `hmlcore/nodes/pruner_node.py` — `explicit_reap` block: `return` → fallthrough + updated warning
- `tests/test_ohm_pipeline.py` — `TestSFTGRPOAndShortGPT`: `--prune_only` → `--prune_experts`; `TestREAPNoMoEGuard`: updated docstring to reflect ShortGPT fallback

---

### 2026-03-22 — `dense_pruner` NameError: `cal` → `cal_texts`

**Area:** `hmlcore/dense_pruner.py`
**Problem:** `_compute_layer_importance` raised `NameError: name 'cal' is not defined` on the first calibration sample, crashing all 5 ShortGPT tests. The variable holding the sample list is `cal_texts` but the three `except` warning blocks referenced the old name `cal` from before the calibration refactor. The NameError inside the `except` clause masked the original exception and propagated as `Dense layer pruning failed`.
**Decision:** Rename `len(cal)` → `len(cal_texts)` in all three warning callsites (tokenisation, embedding extraction, layer call failures).
**Files changed:** `hmlcore/dense_pruner.py` (lines 270, 283, 297)

---

### 2026-03-22 — BnB dequantization fix: load WITH quantization_config + explicit dequantize

**Area:** `hmlcore/nodes/pruner_node.py`, `hmlcore/nodes/output_node.py`
**Problem:** 6 tests failing with `RuntimeError: size mismatch` during BnB merge.  Root cause: the previous "fix" stripped `quantization_config` from `AutoConfig` before calling `from_pretrained`, creating a clean bf16 model (weights shaped `[1024, 3072]`), but the on-disk safetensors still contained BnB-packed weights (`[1572864, 1]`) — transformers raised `RuntimeError: ignore_mismatched_sizes=False`.  The stripping approach was fundamentally wrong.

**Decision:** Remove all `quantization_config` stripping.  Load the base model WITH `quantization_config` intact so BnB correctly unpacks the on-disk weights.  Then, after `merge_and_unload()`, explicitly replace all `Linear4bit`/`Linear8bitLt` layers with standard `nn.Linear` using `bnb.functional.dequantize_4bit` (dispatches to C++/CUDA depending on device).

**Why alternatives were rejected:**
- `ignore_mismatched_sizes=True`: loads wrong weights (random), not a fix
- CPU-only reload without quantization_config: shape mismatch persists
- Direct `merge_and_unload()` on quantized model: raises `NotImplementedError` (BnB limitation)

**New helper:** `_dequantize_bnb_model(model, dtype)` in `output_node.py` — walks all named modules, replaces BnB linear layers with clean `nn.Linear`, handles bias and device placement.  Imported into `pruner_node.py` to share implementation.

**Also fixed:** `pruner_node._merge_lora_via_bf16_reload` now loads directly onto GPU when CUDA is available (avoids Unsloth's `apply_qkv` kernel init failure that occurs when tensors are moved CPU→GPU after load).

**Affected tests (6 failures resolved):**
- `TestShortGPT::test_shortgpt_bnb/instruct` — needed BnB merge to succeed before ShortGPT
- `TestREAPNoMoEGuard::test_reap_no_moe_bnb/instruct/vlm` — PrunerNode merges before REAP guard check
- `TestSFTGRPOAndShortGPT::test_full_pipeline_bnb/instruct` — same BnB merge path
- `TestOutputFormats::test_bnb_merge_*` — `_peft_merge_save` fallback had same root cause

**Files changed:**
- `hmlcore/nodes/output_node.py` — added `_dequantize_bnb_model()`; fixed `_peft_merge_save` to load WITH quantization_config, call `_dequantize_bnb_model` after merge
- `hmlcore/nodes/pruner_node.py` — removed quantization_config stripping; added GPU-first load; imports and calls `_dequantize_bnb_model` from output_node

---

### 2026-03-22 — Calibration strategy from moe-compress

**Area:** Calibration quality for REAP + ShortGPT
**Problem:** Both pruners always selected the first N dataset rows — order-dependent, no signal about sample quality. Very short samples contribute noise; very long samples spike VRAM.
**Decision:** Introduce `hmlcore/calibration.py` (shared) implementing four selection strategies from the moe-compress project (github.com/0xSero/moe-compress):

| Strategy | Behaviour | Best for |
|---|---|---|
| `longest` | Sort descending by estimated token count | Default — maximises hidden-state transitions per sample → more reliable ShortGPT/REAP scores |
| `shortest` | Sort ascending | Short-turn diversity |
| `random` | Seed-controlled shuffle | Reproducible cross-run comparison |
| `first` | Natural dataset order | Legacy behaviour |

Token estimation: `max(1, len(text) // 4)` (moe-compress heuristic).
Pre-filtering: skip samples `< 10` estimated tokens (no signal) or `> max_cal_length` tokens (VRAM spike).

Text extraction fallback chain (upgraded):
1. Known fields: `prompt`, `text`, `instruction`, `question`, `input`, `content`, `output`, `response`, `answer`
2. Chat messages format (`messages` / `conversations` list → `"role: content"`)
3. Concatenate all string values as last resort

**New CLI flag:** `--calibration_strategy longest|shortest|random|first` (default `longest`)

**What was NOT taken from moe-compress:** orchestration pipeline, HTML/MD report renderer, two-lane bundle (adds config complexity for marginal gain given domain-specific training data).

**Files changed:**
- `hmlcore/calibration.py` — NEW — shared sample builder
- `hmlcore/dense_pruner.py` — `_compute_layer_importance` now accepts `List[str]`; `drop_dense_layers` calls `build_calibration_samples`, adds `calibration_strategy` param
- `hmlcore/moe.py` — `compute_reap_scores` / `reap_prune_moe` use `build_calibration_samples`, add `calibration_strategy` param; also unwraps multimodal processor tokenizer before tokenisation
- `hmlcore/config.py` — `--calibration_strategy` arg added
- `hmlcore/nodes/pruner_node.py` — reads `args.calibration_strategy`, passes to both pruners

---

### 2026-03-22 — Test suite rewrite (35 tests, 3 models)

**Area:** `tests/test_ohm_pipeline.py`
**Problem:** Suite used `qwen-moe` (now `qwen-vlm`) and `qwen-bnb-4` (now `qwen-bnb`) — module-level skip fired silently, all 23 tests became no-ops. Only 2 models tested; VLM path under-covered. Failure messages lacked diagnostic detail.
**Decision:** Full rewrite with:
- Three models: `qwen-bnb` (Qwen3, BnB 4-bit, dense), `qwen-instruct` (Qwen2, BnB 4-bit, dense), `qwen-vlm` (Qwen3.5, VLM, SSM-hybrid)
- Per-model `pytest.mark.skipif` on missing folder instead of module-level skip — surviving models still run
- 35 tests across 8 classes: `TestSFT`, `TestGRPO`, `TestShortGPT`, `TestREAPNoMoEGuard`, `TestSFTAndGRPO`, `TestSFTGRPOAndShortGPT`, `TestOutputFormats`, `TestPipelineCheck`
- `_diagnose()` helper: every failing assertion prints exit code, all WARNING/ERROR/exception lines, and last 60 lines of combined stdout+stderr
- `TestREAPActualPruning` class (4 tests) gated on `models/qwen-moe-real` — activates automatically when a real MoE model is present

**Files changed:**
- `tests/test_ohm_pipeline.py` — full rewrite

---

### 2026-03-22 — REAP guard: `--prune_experts` vs `--prune_only`

**Area:** `hmlcore/nodes/pruner_node.py`
**Problem (bug):** `apply_args()` sets `prune_experts=True` whenever `--prune_only` is passed. The new "no MoE" guard checked only `prune_experts`, so `--prune_only` on a dense model also triggered the guard and skipped ShortGPT — the intended fallback path never ran.
**Decision:** Distinguish explicit vs implicit `prune_experts`:

```python
explicit_reap = (
    getattr(args, "prune_experts", False)
    and not getattr(args, "prune_only", False)
)
if explicit_reap:
    logger.warning("⚠️  --prune_experts was passed but ... no MoE layers")
    return   # skip; do not fall through to ShortGPT
```

Routing table after fix:

| Flag | Dense model | MoE model |
|---|---|---|
| `--prune_experts` | ⚠️ warn + skip | ✅ REAP |
| `--prune_only` | ✅ ShortGPT | ✅ REAP |

**Files changed:**
- `hmlcore/nodes/pruner_node.py`

---

### 2026-03-22 — Strip `quantization_config` before base model reload

**Area:** BnB re-quantization during LoRA merge
**Problem:** `_merge_lora_via_bf16_reload` (PrunerNode) and `_peft_merge_save` (OutputNode) reloaded the base model with `AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=bfloat16, ...)`. When the base model's `config.json` contains an embedded `quantization_config` block (present on all BnB-quantized models), transformers auto-applies 4-bit quantization regardless of `torch_dtype`. Result: the "bf16 reload" was still quantized → `state_dict()` contained `uint8` tensors → GGUF converter rejected them as BnB-quantized.

Passing `quantization_config=None` does NOT override this — `None` is the default that triggers the embedded config lookup.

**Decision:** Load `AutoConfig` first, delete `quantization_config` attribute if present, pass the stripped config explicitly:

```python
reload_cfg = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
if hasattr(reload_cfg, "quantization_config"):
    del reload_cfg.quantization_config
base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    config=reload_cfg,    # ← no quantization_config → clean bf16 load
    torch_dtype=dtype,
    ...
)
```

Applied in both reload sites: PrunerNode and OutputNode `_peft_merge_save`.

**Files changed:**
- `hmlcore/nodes/pruner_node.py` — `_merge_lora_via_bf16_reload`
- `hmlcore/nodes/output_node.py` — `_peft_merge_save`

---

### 2026-03-21 — OutputNode: stale weight files + 3-tier save fallback

**Area:** GGUF conversion after pruning
**Problems:**
1. A `model.safetensors` from a failed prior run sat alongside a fresh `pytorch_model.bin` — GGUF converters prefer safetensors, loading the old 4-bit weights.
2. Newer transformers versions track weight transformations at load time (`revert_weight_conversion`) and call `reverse_op` on save; when `reverse_op` raises `NotImplementedError` with an empty message, both safetensors and `safe_serialization=False` paths fail silently.
3. `shutil.rmtree` raises `PermissionError` on Windows for memory-mapped safetensors files held by another process.

**Decisions:**
- `_purge_stale_weights(directory)` — per-file deletion via `glob + os.remove + try/except` (no `rmtree`). Runs before every save to `finale/`.
- `_safe_save_pretrained(model, tokenizer, save_dir)` — 3-tier fallback:
  1. `model.save_pretrained(save_dir)` → safetensors (preferred)
  2. `model.save_pretrained(save_dir, safe_serialization=False)` → pytorch_model.bin
  3. `torch.save(model.state_dict(), "pytorch_model.bin")` + `config.save_pretrained()` → bypasses all transformers save hooks including `revert_weight_conversion`

**Files changed:**
- `hmlcore/nodes/output_node.py`

---

### 2026-03-21 — OutputNode: improve exception reporting

**Area:** `hmlcore/nodes/runner.py`
**Problem:** `NodeError: OutputNode raised an unexpected error: ???` — `NotImplementedError` from `revert_weight_conversion` has an empty `__str__`, making the outer catch `f"{type(exc).__name__}: {exc}"` print nothing useful.
**Decision:** Switch to `logger.exception()` (prints full traceback) and include `type(exc).__name__` explicitly in the NodeError message.

**Files changed:**
- `hmlcore/nodes/runner.py`

---

### 2026-03-21 — Pipeline pre-flight compatibility report

**Area:** New feature — `hmlcore/nodes/pipeline_check.py`
**Problem:** Users running unsupported model architectures (SSM-hybrid, multimodal with GRPO) discovered incompatibilities only after minutes of training, with cryptic errors.
**Decision:** Add `run_pipeline_check(model, tokenizer, args, is_multimodal)` called from InputNode immediately after model load. Prints a compatibility table before any training begins:

```
╔══════════════════════════════════════════╗
║        Pipeline Compatibility Report     ║
╟──────────────────┬───┬────┬──────┬──────╢
║ Architecture     │SFT│GRPO│Prune │Output║
║ Dense transformer│ ✅ │ ✅ │ ✅   │ ✅  ║
╚══════════════════╧═══╧════╧══════╧══════╝
```

Detects: MoE topology, Mamba/SSM hybrid, dense transformer, multimodal processor, BnB quantization. Never raises — wrapped in broad `except`.

**Files changed:**
- `hmlcore/nodes/pipeline_check.py` — NEW
- `hmlcore/nodes/input_node.py` — calls `run_pipeline_check`

---

### 2026-03-21 — Mamba/SSM hybrid detection

**Area:** `hmlcore/dense_pruner.py`
**Problem:** ShortGPT drops layers by renumbering the remaining block indices. GGUF/llama.cpp hardcodes block types at specific positions in the architecture definition. When SSM (Mamba) and Attention blocks are interleaved and layers are dropped, the block-position mapping becomes corrupted → `blk.N.ssm_conv1d.weight` missing → unloadable GGUF file.
**Decision:** `_is_hybrid_ssm(layers)` — scans all sub-module names in the layer list against `_SSM_ATTRS` (`ssm_conv1d`, `dt_proj`, `A_log`, `x_proj`, `dt_layernorm`, `mixer`, `conv1d`). If any match, `find_decoder_layers` returns `(None, None)` with a warning. This prevents ShortGPT from running on hybrid architectures.

**Files changed:**
- `hmlcore/dense_pruner.py`

---

### 2026-03-21 — Layer-by-layer calibration (bypass patched model.forward)

**Area:** `hmlcore/dense_pruner.py`
**Problem:** Calibration used `model(**inputs)` with hooks. Multiple framework patches intercept `model.forward`:
- Unsloth patches `apply_qkv` / rotary kernels — only initialised when model loads to CUDA via `from_pretrained`. Loading to CPU then `.cuda()` leaves these as `None` → `AttributeError: 'Qwen2Attention' has no attribute 'apply_qkv'`.
- Gradient checkpointing recompute hooks interfere with `@torch.no_grad()` calibration.
- Multimodal processor wraps tokenizer — `tokenizer(text)` routes through image parser → `Incorrect image source`.

**Decision:** Replace hooks + `model(**inputs)` with direct layer-by-layer forward:
1. `_get_initial_hidden_states(model, input_ids)` — resolves embedding layer via `_EMBED_PATHS` list, adds absolute position embeddings for GPT-2 style
2. `_call_layer(layer, hidden_states)` — tries 5 calling conventions (different `kwargs` combinations) until one succeeds
3. `h_in` / `h_out` captured inline inside the loop — no hooks needed
4. `model.gradient_checkpointing_disable()` before loop
5. `text_tok = getattr(tokenizer, "tokenizer", tokenizer)` — unwrap multimodal processor

**Files changed:**
- `hmlcore/dense_pruner.py`

---

### 2026-03-21 — device_map fix for Unsloth CUDA kernel init

**Area:** `hmlcore/nodes/pruner_node.py` — `_merge_lora_via_bf16_reload`
**Problem:** Loading base model with `device_map="cpu"` then calling `.cuda()` moves tensors but does not trigger Unsloth's CUDA kernel initialisation (happens only during `from_pretrained` on a CUDA device). Result: patched attention methods exist in the class but their underlying functions are uninitialised → `AttributeError`.
**Decision:** Change to `device_map={"": torch.cuda.current_device()}` — loads directly onto GPU so Unsloth's `__init__` hooks fire at the right time. Fall back to `"cpu"` when no CUDA is available. Remove the now-unnecessary `.cuda()` call.

**Files changed:**
- `hmlcore/nodes/pruner_node.py`

---

### 2026-03-21 — PrunerNode topology ordering (merge before detect)

**Area:** `hmlcore/nodes/pruner_node.py`
**Problem:** `find_decoder_layers()` was called on the pre-merge `PeftModel`. PEFT's `LoraModel.__getattr__` proxy intercepts attribute access — `model.model.layers` resolves differently through the proxy than on a bare HF model. Result: `find_decoder_layers` returned `(None, None)` → PrunerNode logged "cannot prune" and exited → `args._already_merged` was never set → OutputNode attempted a second merge → pruned size identical to unpruned → `--prune_ratio` had no effect.
**Decision:** Restructure PrunerNode execution order:
1. Merge LoRA FIRST (via `merge_and_unload` or `_merge_lora_via_bf16_reload`)
2. Set `args._already_merged = True`
3. Detect topology on the clean merged model
4. Prune

**Files changed:**
- `hmlcore/nodes/pruner_node.py`

---

### 2026-03-20 — Delete dead code: finetuner/, ohm_runner.py, ohm_finetuner_backup.py

**Area:** Repository cleanup
**Problem:** `finetuner/` was the original monolithic implementation, superseded by `hmlcore/`. `ohm_runner.py` imported from `finetuner.run` — dead import. `ohm_finetuner_backup.py` was a stale backup.
**Decision:** Delete all three. `ohm_finetuner.py` is the sole entry point; `hmlcore/` is the sole library.

**Files changed:**
- DELETED: `finetuner/` (entire directory)
- DELETED: `ohm_runner.py`
- DELETED: `ohm_finetuner_backup.py`

---

## Architecture snapshot (as of 2026-03-22)

```
akatsuki/
├── ohm_finetuner.py            Entry point — thin CLI wrapper
├── ARCHITECTURE.md             Detailed architecture reference
├── TDD.md                      This document — change history
│
└── hmlcore/
    ├── config.py               CLI args, apply_args(), global prompt tags
    ├── model.py                Model + tokenizer loading (Unsloth / PEFT)
    ├── data.py                 Dataset loading, schema normalisation
    ├── trainer.py              SFT + GRPO training wrappers
    ├── rewards.py              Reward functions + LMStudioJudge
    ├── moe.py                  REAP expert pruning (MoE)
    ├── dense_pruner.py         ShortGPT layer dropping (dense)
    ├── calibration.py          ← NEW 2026-03-22 — shared sample selector
    │
    └── nodes/
        ├── runner.py           GraphRunner (Kahn BFS topological executor)
        ├── input_node.py       Load model / tokenizer / dataset
        ├── sft_node.py         SFT warm-up
        ├── grpo_node.py        GRPO RL
        ├── pruner_node.py      REAP / ShortGPT + LoRA merge
        ├── output_node.py      Save / merge / GGUF export
        ├── pipeline_check.py   ← NEW 2026-03-21 — pre-flight compat report
        └── model_info.py       Post-stage model snapshot logging
```

### Key design invariants

| Invariant | Where enforced |
|---|---|
| Topology detection MUST run after LoRA merge | `pruner_node.py` — merge in Step 1, detect in Step 3 |
| Base model reload MUST strip `quantization_config` | `pruner_node._merge_lora_via_bf16_reload`, `output_node._peft_merge_save` |
| Unsloth kernel init requires CUDA `from_pretrained` | `pruner_node._merge_lora_via_bf16_reload` — `device_map={"": cuda:N}` |
| ShortGPT MUST NOT run on SSM/Mamba hybrid models | `dense_pruner._is_hybrid_ssm` — returns `(None, None)` |
| `--prune_experts` on dense model MUST warn and skip | `pruner_node` — `explicit_reap` guard |
| `--prune_only` on dense model MUST run ShortGPT | `pruner_node` — `prune_only` exempted from REAP guard |
| Windows: never use `shutil.rmtree` on finale/ | `output_node._purge_stale_weights` — per-file `os.remove` |
| Save failures MUST have 3-tier fallback | `output_node._safe_save_pretrained` |

### Model compatibility matrix

| Architecture | SFT | GRPO | ShortGPT | REAP | Notes |
|---|---|---|---|---|---|
| Dense LLM (Qwen2, Qwen3, LLaMA, Mistral) | ✅ | ✅ | ✅ | ⚠️ skip | `--prune_experts` warns; `--prune_only` runs ShortGPT |
| MoE LLM (Qwen3-MoE, Mixtral, OLMoE) | ✅ | ✅ | ➡️ REAP | ✅ | MoE detected → always uses REAP |
| VLM (Qwen3.5-VLM, LLaVA, Qwen2-VL) | ✅ | ❌ skip | ⚠️ skip | ⚠️ skip | GRPO skipped (multimodal); SSM-hybrid blocks pruning |
| SSM/Mamba hybrid | ✅ | ✅ | ❌ skip | ❌ skip | Block renumbering corrupts GGUF position map |

### CLI quick reference (as of 2026-03-22)

```bash
# Full pipeline — SFT → GRPO → save adapter
python ohm_finetuner.py --student_model models/qwen-bnb \
    --datasets datasets/mydata.jsonl --domain code --max_steps 100

# Full pipeline + merge to bf16 HF checkpoint
python ohm_finetuner.py ... --merge --quantize bf16

# Full pipeline + GGUF export
python ohm_finetuner.py ... --merge --quantize q4_k

# Skip SFT (GRPO only)
python ohm_finetuner.py ... --disable_sft

# ShortGPT dense pruning only (no training)
python ohm_finetuner.py ... --prune_only --prune_ratio 0.3 \
    --calibration_samples 64 --calibration_strategy longest

# REAP MoE expert pruning only (MoE model required)
python ohm_finetuner.py ... --prune_only --prune_ratio 0.5 \
    --calibration_samples 128 --calibration_strategy longest

# Full pipeline + REAP after training (MoE model)
python ohm_finetuner.py ... --prune_experts --prune_ratio 0.3
```

### Test suite (as of 2026-03-22)

```bash
pytest tests/test_ohm_pipeline.py -v --tb=short        # all 35 tests
pytest tests/test_ohm_pipeline.py -v --tb=short -k bnb  # qwen-bnb only
```

Test models:

| Folder | Architecture | Quantization | MoE | VLM |
|---|---|---|---|---|
| `models/qwen-bnb` | Qwen3ForCausalLM (28L) | BnB 4-bit | No | No |
| `models/qwen-instruct` | Qwen2ForCausalLM (24L) | BnB 4-bit | No | No |
| `models/qwen-vlm` | Qwen3_5ForConditionalGeneration | None | No | Yes (SSM-hybrid) |
| `models/qwen-moe-real` | *(placeholder — add OLMoE-1B-7B)* | — | Yes | No |
