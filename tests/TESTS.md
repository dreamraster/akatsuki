# ohm_finetuner Pipeline Tests

Smoke-test suite for `ohm_finetuner.py`. Verifies that each pipeline stage and
output format runs to completion and writes the expected artefacts.
**Quality is not assessed** — only that the pipeline exits cleanly.

---

## Quick start

```bash
# Full suite
pytest tests/test_ohm_pipeline.py -v --tb=short

# Skip slow GPU tests (collection / import check only)
pytest tests/test_ohm_pipeline.py --collect-only

# LLM tests only
pytest tests/test_ohm_pipeline.py -v -m "not vlm and not moe"

# VLM tests only
pytest tests/test_ohm_pipeline.py -v -m vlm

# Output format tests only
pytest tests/test_ohm_pipeline.py -v -k TestOutputFormats
```

---

## Test models

| Directory | Architecture | Type | MoE |
|---|---|---|---|
| `models/qwen-bnb-4` | `Qwen3ForCausalLM` | Dense text LLM (4-bit bnb) | No |
| `models/qwen-moe` | `Qwen3_5ForConditionalGeneration` | Multimodal VLM | No |
| `models/qwen-moe-real` | *(add real MoE model here)* | MoE text or VLM-MoE | Yes |

> `qwen-moe-real` does not exist yet. Tests in `TestREAPActualPruning` are
> automatically skipped until a model is placed there.

---

## Shared base args (all tests)

| Arg | Value | Reason |
|---|---|---|
| `--max_steps` | `2` | Minimum to exercise the training loop |
| `--num_generations` | `2` | Minimum valid GRPO rollout count |
| `--lora_rank` | `8` | Small adapter — fast init, low VRAM |
| `--max_length` | `1024` | Exercises template path; trainer truncates |
| `--calibration_samples` | `4` | Minimum to register REAP hooks |
| `--batch_size` | `1` | Lowest memory footprint |
| `--domain` | `code` | Matches test dataset content |
| `--datasets` | `teichiai-claude-4.5-high-reasoning-250x.jsonl` | Smallest available dataset (1.4 MB) |

---

## Test inventory

### `TestSFT` — SFT stage isolation

| Test | Model | What is checked |
|---|---|---|
| `test_sft_llm` | LLM | `sft/sft_complete` sentinel written |
| `test_sft_vlm` | VLM | `sft/sft_complete` written; `"Multimodal model detected"` in logs |

SFT is tested with the full default pipeline (GRPO also runs). The sentinel
file `sft/sft_complete` is the reliable proof that SFT completed.

---

### `TestGRPO` — GRPO stage isolation

Uses `--disable_sft` to skip SFT and isolate GRPO.

| Test | Model | What is checked |
|---|---|---|
| `test_grpo_llm` | LLM | `"Starting Step 2"` in logs |
| `test_grpo_skipped_vlm` | VLM | `"Multimodal model detected"` in logs (GRPO auto-skipped) |

VLM models are detected at runtime via class name containing
`"ConditionalGeneration"` or config having `vision_config`. GRPO is skipped
automatically because text-only GRPO cannot run on vision-language models
(their `compute_3d_position_ids` requires visual tokens).

---

### `TestREAP` — REAP pruning isolation

Uses `--prune_only` (implies `--prune_experts --merge`, skips SFT + GRPO).

| Test | Model | What is checked |
|---|---|---|
| `test_reap_llm_no_moe` | LLM | No-MoE skip message in logs; exit 0 |
| `test_reap_vlm_no_moe` | VLM | No-MoE skip message in logs; exit 0 |
| `test_reap_custom_ratio` | LLM | `--prune_ratio 0.25 --calibration_samples 8` accepted without error |

Neither current test model has MoE layers. The pre-check in `ohm_finetuner.py`
calls `find_moe_layers()` before the expensive LoRA merge and bails early with a
clear error message when no MoE layers are found, avoiding the 4-bit merge
warning that would otherwise be emitted unnecessarily.

---

### `TestSFTAndGRPO` — Combined SFT → GRPO pipeline

| Test | Model | What is checked |
|---|---|---|
| `test_sft_grpo_llm` | LLM | SFT sentinel + `"Starting Step 2"` in logs |
| `test_sft_grpo_vlm` | VLM | SFT sentinel + `"Multimodal model detected"` in logs |

---

### `TestSFTGRPOAndREAP` — Full pipeline with `--prune_experts`

Exercises all three stages in one invocation. With the current test models
(no MoE), REAP is skipped via the pre-check; this validates the graceful
end-to-end path.

| Test | Model | What is checked |
|---|---|---|
| `test_full_pipeline_llm` | LLM | SFT sentinel + GRPO start + no-MoE skip |
| `test_full_pipeline_vlm` | VLM | SFT sentinel + multimodal GRPO-skip + no-MoE skip |

---

### `TestOutputFormats` — Merge and quantization outputs

All tests use `--merge --quantize <fmt>`. GGUF formats (`f16`, `q8_0`, `q4_k`)
call Unsloth's `save_pretrained_gguf` when available; if Unsloth cannot export
GGUF for the architecture, the code falls back to HF format and logs a warning.
Both outcomes are accepted — tests only require `finale/` to be non-empty.

| Test | Model | Format | HF expected | GGUF expected |
|---|---|---|---|---|
| `test_output_bf16` | LLM | `bf16` | `finale/config.json` | — |
| `test_output_f16` | LLM | `f16` | fallback | `*.gguf` |
| `test_output_q8` | LLM | `q8_0` | fallback | `*.gguf` |
| `test_output_q4k` | LLM | `q4_k` | fallback | `*.gguf` |
| `test_output_bf16_vlm` | VLM | `bf16` | `finale/` non-empty | — |
| `test_output_f16_vlm` | VLM | `f16` | fallback | `*.gguf` |
| `test_output_q8_vlm` | VLM | `q8_0` | fallback | `*.gguf` |
| `test_output_q4k_vlm` | VLM | `q4_k` | fallback | `*.gguf` |

---

### `TestREAPActualPruning` — Real MoE pruning *(skipped until model added)*

Skipped automatically when `models/qwen-moe-real/` does not exist.
To activate: place a model with actual MoE layers (e.g. Qwen3-30B-A3B-Instruct)
at that path and update `_MOE_MODEL` in the test file if needed.

| Test | What is checked |
|---|---|
| `test_prune_only_moe` | `"REAP pruning complete"` in logs; `finale/` non-empty |
| `test_sft_grpo_reap_moe` | Full pipeline; REAP actually prunes; `finale/` non-empty |
| `test_prune_moe_merge_bf16` | Prune + bf16 HF save |
| `test_prune_moe_merge_q4k` | Prune + q4_k GGUF export |

---

## Coverage matrix

| | LLM | VLM | Real MoE |
|---|:---:|:---:|:---:|
| SFT | ✅ | ✅ | — |
| GRPO | ✅ | ✅ skip | — |
| REAP | ✅ no-MoE | ✅ no-MoE | ✅ skipif |
| SFT + GRPO | ✅ | ✅ | — |
| SFT + GRPO + REAP | ✅ | ✅ | ✅ skipif |
| output bf16 | ✅ | ✅ | — |
| output f16 | ✅ | ✅ | — |
| output q8_0 | ✅ | ✅ | — |
| output q4_k | ✅ | ✅ | ✅ skipif |

**Total: 24 tests** (20 always active, 4 skipped until real MoE model is added)

---

## Pytest markers

| Marker | Meaning |
|---|---|
| `gpu` | Requires a CUDA GPU |
| `slow` | Long-running (model load + training steps); excluded by `-m "not slow"` |
| `vlm` | Vision-language model specific |
| `moe` | MoE or REAP-related |

---

## Artefact locations

All tests use pytest's `tmp_path` fixture (unique temp dir per test).

| Artefact | Path | Written by |
|---|---|---|
| SFT sentinel | `<out>/sft/sft_complete` | SFT stage on completion |
| SFT adapter | `<out>/sft/adapter_config.json` + weights | SFT stage |
| GRPO checkpoints | `<out>/grpo/checkpoint-*/` | GRPO trainer (every 16 steps) |
| Final model (HF) | `<out>/finale/config.json` + weights | Merge stage |
| Final model (GGUF) | `<out>/finale/*.gguf` | Unsloth GGUF export |

---

## Known limitations

- **No true MoE model in repo** — `compute_reap_scores` and `prune_moe_experts`
  are not exercised until `models/qwen-moe-real/` is populated.
- **GGUF export depends on Unsloth** — if Unsloth does not support the model
  architecture's GGUF path, the pipeline falls back to HF format silently.
  Tests accept both; to assert `.gguf` specifically use `_has_gguf(tmp_path)`.
- **VLM calibration forward pass** — if a VLM-MoE model is added and REAP
  calibration is attempted, the `model(**text_only_inputs)` call may fail for
  the same reason as GRPO (missing visual tokens in `compute_3d_position_ids`).
  This would require passing dummy visual inputs or patching the forward method.
