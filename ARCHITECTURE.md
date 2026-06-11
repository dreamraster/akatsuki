# Akatsuki — Technical Architecture

> GRPO-based LLM distillation and pruning pipeline
> Copyright (c) 2026 dreamraster · OHM TECH

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Layout](#2-project-layout)
3. [Entry Point & CLI](#3-entry-point--cli)
4. [Node-Graph Pipeline](#4-node-graph-pipeline)
5. [Nodes — Detail](#5-nodes--detail)
   - [InputNode](#51-inputnode)
   - [SFTNode](#52-sftnode)
   - [GRPONode](#53-grponode)
   - [PrunerNode](#54-prunernode)
   - [OutputNode](#55-outputnode)
6. [Model Loading](#6-model-loading)
7. [Dataset Loading & Preprocessing](#7-dataset-loading--preprocessing)
8. [Reward Functions](#8-reward-functions)
9. [Pruning Algorithms](#9-pruning-algorithms)
   - [REAP — MoE Expert Pruning](#91-reap--moe-expert-pruning)
   - [Bonsai & DLP — Modern Dense Pruning](#92-bonsai--dlp--modern-dense-pruning)
   - [PRISM — Data Pruning](#93-prism--data-pruning)
   - [PRISM-DQ — Dynamic Quantization](#94-prism-dq--dynamic-quantization)
   - [X-Token — Cross-Tokenizer Distillation](#95-x-token--cross-tokenizer-distillation)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)
    - [Pre-flight Compatibility Check](#101-pre-flight-compatibility-check)
    - [Stage Model Snapshots](#102-stage-model-snapshots)
    - [Windows Compatibility](#103-windows-compatibility)
    - [Unsloth Integration](#104-unsloth-integration)
11. [Data Flow Diagram](#11-data-flow-diagram)
12. [Configuration Reference](#12-configuration-reference)
13. [Architecture Constraints & Known Limitations](#13-architecture-constraints--known-limitations)

---

## 1. Overview

Akatsuki is a modular pipeline for fine-tuning and compressing language models using GRPO (Group Relative Policy Optimisation) with optional SFT warm-up and post-training pruning. It produces models optimised for specialised domains (math, code, general) and exports them as HuggingFace checkpoints or GGUF files for llama.cpp / Ollama.

**Key design principles:**

- **Node-graph pipeline** — each stage is a self-describing node with declared I/O keys; a topological executor wires them together
- **Two model backends** — Unsloth (fast, 4-bit CUDA) with PEFT fallback (standard BitsAndBytes + LoRA)
- **Domain-aware rewards** — rule-based math scoring, LLM-judged code/general scoring
- **Intrinsically diverse data selection** — PRISM engine for pruning redundant samples before training
- **Post-training pruning** — REAP for MoE architectures, Bonsai/DLP for dense transformers
- **Resume at any stage** — sentinel files and HF checkpoint detection allow resuming SFT or GRPO mid-run

---

## 2. Project Layout

```
akatsuki/
├── ohm_finetuner.py          # Main entry point — thin wrapper around the pipeline
├── ohm_distiller.py          # Standalone distillation script
├── ohm_databuilder.py        # Dataset construction utilities
├── ohm_datapreprocessor.py   # Raw data preprocessing
├── vlm_scene_builder.py      # [NEW] Synthetic 2D scene dataset generator
├── ARCHITECTURE.md           # This document
│
└── hmlcore/                  # Core library
    ├── __init__.py            # Version (0.3.0)
    ├── config.py              # CLI argument parser, global prompt tags, apply_args()
    ├── model.py               # Model + tokenizer loading (Unsloth / PEFT)
    ├── data.py                # Dataset loading, schema normalisation, chat template
    ├── prism_selector.py      # PRISM data selection engine
    ├── trainer.py             # SFT + GRPO training wrappers, checkpoint helpers
    ├── rewards.py             # All reward functions + LMStudioJudge
    ├── moe.py                 # REAP expert pruning for MoE models
    ├── dense_pruner.py        # Bonsai/DLP structural pruning for dense transformers
    │
    ├── xtoken/                # X-Token knowledge distillation
    │   ├── __init__.py        # Package exports
    │   ├── distiller.py       # XTokenDistiller, XTokenConfig
    │   ├── projection.py      # ProjectionAligner, CrossTokenizerMatcher
    │   └── node.py            # XTokenNode, XTokenDistillNode pipeline nodes
    │
    └── nodes/                 # Pipeline node graph
        ├── __init__.py        # Public re-exports
        ├── base.py            # BaseNode ABC + NodeError
        ├── context.py         # NodeContext TypedDict
        ├── runner.py          # GraphRunner (Kahn topological executor)
        ├── input_node.py      # Load model / tokenizer / dataset
        ├── sft_node.py        # SFT warm-up stage
        ├── grpo_node.py       # GRPO RL stage
        ├── pruner_node.py     # REAP / ShortGPT pruning + LoRA merge
        ├── output_node.py     # Final save / merge / GGUF export
        ├── model_info.py      # Compact model snapshot logging
        └── pipeline_check.py  # Pre-flight compatibility report
```

---

## 3. Entry Point & CLI

**File:** `ohm_finetuner.py`

```python
def main():
    parser = build_parser()          # hmlcore.config
    args   = parser.parse_args()
    apply_args(args)                 # auto-enable flags, inject globals

    runner = GraphRunner([
        InputNode(),
        SFTNode(),
        GRPONode(),
        PrunerNode(),
        OutputNode(),
    ])
    runner.run(make_context(args))
```

This file is intentionally minimal — all logic lives in `hmlcore`.

### Key CLI Arguments

Flags not listed here are advanced/rarely-needed and suppressed from `--help` but still functional (e.g. `--calibration_strategy`, `--bonsai_noise`, `--dlp_scale`, `--prism_layer`, `--r_start/r_end/s_start/s_end`).

| Group | Flag | Default | Description |
|---|---|---|---|
| **Model** | `--student_model` | *(required)* | HF hub ID or local path |
| | `--lora_rank` | `32` | LoRA rank (alpha = rank×2) |
| | `--disable_unsloth` | `False` | Force PEFT fallback |
| **Data** | `--datasets` | *(required)* | Comma-separated paths or HF IDs |
| | `--domain` | `code` | `math` · `code` · `general` · `scene` |
| | `--max_length` | `2048` | Token budget (prompt + completion) |
| **Training** | `--output_dir` | `./output` | Checkpoint and output root directory |
| | `--max_steps` | `1` | Total GRPO optimiser steps |
| | `--batch_size` | `1` | Per-device batch size |
| | `--num_generations` | `4` | Rollouts per prompt (GRPO group size) |
| | `--disable_sft` | `False` | Skip SFT warm-up |
| | `--resume` | `False` | Auto-detect and resume checkpoints |
| | `--qwen_jack` | `False` | Qwopus/Qwen3-Thinking template alignment |
| **Save** | `--merge` | `False` | Merge LoRA into base weights |
| | `--quantize` | `bf16` | `bf16` · `f16` · `q8_0` · `q4_k_m` · (full list in `--help`) |
| **Pruning** | `--prune` | `False` | Prune after training — REAP for MoE, Bonsai/DLP for dense |
| | `--prune_ratio` | `None` | Fraction of experts/layers to drop; auto-enables `--prune` |
| | `--prune_only` | `False` | Skip SFT+GRPO; merge, prune, and save only |
| | `--calibration_samples` | `128` | Samples used for layer/expert importance scoring |
| | `--dynamicquant` | `False` | Quantize low-scored layers to 1-bit instead of removing |
| **Judge** | `--judge_model` | `None` | LM Studio model name for code/general scoring |
| | `--judge_url` | `http://localhost:1234` | LM Studio API base URL |
| | `--judge_timeout` | `60` | Per-request timeout (s) |
| **PRISM** | `--prism_select` | `False` | Enable PRISM data selection before training |
| | `--prism_tier` | `high` | Quality tier to keep (`high` · `mid` · `low`) |
| | `--prism_only` | `False` | Run PRISM selection then exit |
| **PRISM-DQ** | `--prism_dq` | `False` | Generate per-tensor GGUF quantization recipe |
| | `--target_bpw` | `4.0` | Target average bits-per-weight |
| | `--dq_refinement` | `False` | Enable per-block refinement pass |
| | `--dq_llama_path` | `None` | Path to `llama-quantize` binary for auto-invocation |
| | `--dq_input_gguf` | `None` | F16 GGUF to quantize with the generated recipe |

**`apply_args(args)` side-effects:**

- `--prune_ratio N` → implicitly sets `prune = True`
- `--prune_only` → implies `prune = True`
- Injects custom prompt tags into `hmlcore.config` globals (`REASONING_START`, `SYSTEM_PROMPT`, etc.)

---

## 4. Node-Graph Pipeline

### Abstractions

**`BaseNode`** (`hmlcore/nodes/base.py`):

```python
class BaseNode(ABC):
    NAME: str                            # Human-readable stage name
    INPUT_KEYS: tuple[str, ...]          # Keys this node reads from context
    OUTPUT_KEYS: tuple[str, ...]         # Keys this node writes to context

    def should_run(self, ctx) -> bool:   # Override to conditionally skip
        return True

    @abstractmethod
    def run(self, ctx) -> None:          # Reads and writes NodeContext in-place
        ...

    def _require(self, ctx, *keys):      # Raises NodeError if any key missing
        ...
```

**`NodeContext`** (`hmlcore/nodes/context.py`) — a `TypedDict` acting as the shared mutable pipeline state:

```
args            → argparse.Namespace (CLI config)
model           → PeftModel or merged HF model
tokenizer       → AutoTokenizer or Processor
use_unsloth     → bool
is_multimodal   → bool (VLM detection)
dataset         → HuggingFace Dataset
sft_dir         → str (path)
grpo_dir        → str (path)
finale_dir      → str (path)
sft_checkpoint  → str | None
grpo_checkpoint → str | None
```

### GraphRunner — Topological Executor

**`hmlcore/nodes/runner.py`**

`GraphRunner.__init__(nodes)` accepts an ordered list of nodes.

`GraphRunner.run(ctx)`:

1. Calls `_topo_sort()` — **Kahn's BFS algorithm** on the directed dependency graph formed by `INPUT_KEYS → OUTPUT_KEYS` edges. Nodes with no unmet dependencies enter the queue first; ties are broken by original list order (deterministic).
2. Iterates the sorted order:
   - Calls `node.should_run(ctx)` — logs `⏭️ Skipping` if False
   - Calls `node.run(ctx)` inside a try/except — catches `NodeError` and raises cleanly; wraps unexpected exceptions as `NodeError`
   - After each node (except `OutputNode`), calls `log_stage_model_info()` for a compact model snapshot
3. Returns final `ctx`.

Cycle detection: if Kahn's BFS cannot process all nodes (residual in-degree > 0), raises `NodeError` naming the involved nodes.

---

## 5. Nodes — Detail

### 5.1 InputNode

**File:** `hmlcore/nodes/input_node.py`
**INPUT_KEYS:** `(args,)`
**OUTPUT_KEYS:** `(model, tokenizer, use_unsloth, is_multimodal, dataset, sft_dir, grpo_dir, sft_checkpoint, grpo_checkpoint)`

**Execution order:**

```
1. Create stage directories  →  {output_dir}/sft/  and  {output_dir}/grpo/
2. Resume detection          →  find_last_checkpoint() / is_sft_complete()
3. Load model + tokenizer    →  hmlcore.model.load_model_and_tokenizer(args)
4. Multimodal detection      →  check class name + config.vision_config
5. Pre-flight check          →  pipeline_check.run_pipeline_check()
6. Chat template             →  data.setup_chat_template(tokenizer)
7. Dataset loading           →  data.load_and_preprocess_dataset(...)
8. PRISM selection           →  prism_selector.select_with_prism() (if enabled)
```

**PRISM Selection logic:**

If `--prism_select` is enabled:
- Hidden states are extracted from the student model (last layer by default).
- Embeddings are re-centered to remove global semantic drift.
- Correlation scores are computed; data is split into tiers.
- If `--prism_only` is set, the filtered dataset is saved and the process exits early.

**Resume logic:**

```
if grpo_checkpoint found  →  skip SFT entirely (weights already in GRPO checkpoint)
elif sft_complete         →  load adapter from sft_dir, resume GRPO
elif sft_checkpoint found →  resume SFT from partial checkpoint
else                      →  fresh start
```

**Multimodal detection:**

```python
is_multimodal = (
    "ConditionalGeneration" in type(model).__name__
    or hasattr(model.config, "vision_config")
)
```

VLM models are now supported in GRPO. If `is_multimodal` is true, the trainer ensures vision tokens and image data are preserved during rollouts.

---

### 5.2 SFTNode

**File:** `hmlcore/nodes/sft_node.py`
**INPUT_KEYS:** `(model, tokenizer, dataset, args, sft_dir, sft_checkpoint, grpo_checkpoint)`
**OUTPUT_KEYS:** `()` — mutates `model` in-place

**Skips when:** `disable_sft`, `prune_only`, or `grpo_checkpoint` is set.

**SFT dataset construction** (first 100 examples):

Each sample's `full_response` is formatted into the target reasoning template:

```
Case 1 — Response already has <reasoning>/<solution> tags:   use as-is
Case 2 — Response has <think>...</think>:                    convert to <reasoning>...</reasoning>
Case 3 — Plain response:                                     synthesise:
          <reasoning>
          Let me work through this step by step.
          {full_response}
          </reasoning>
          <solution>{completion}</solution>
```

The rendered template string (from `tokenizer.apply_chat_template`) is then checked to ensure the prompt prefix is present verbatim — `text.startswith(prompt_str)` — so the SFT loss mask can be computed correctly.

**Training config:**

```python
SFTConfig(learning_rate=2e-4, num_train_epochs=1, ...)
```

Saves with `trainer.save_model(sft_dir)` and writes a `sft_complete` sentinel file.

---

### 5.3 GRPONode

**File:** `hmlcore/nodes/grpo_node.py`
**INPUT_KEYS:** `(model, tokenizer, dataset, args, grpo_dir, grpo_checkpoint, is_multimodal)`
**OUTPUT_KEYS:** `()` — mutates `model` in-place

**Skips when:** `prune_only` or `dataset` is too small.

**Unsloth compatibility:** Sets `model.base.warnings_issued = {}` if missing (Unsloth expects this attribute).

**Reward functions:** Built by `build_reward_functions(args, tokenizer)` — returns `(reward_funcs, judge)`. Domain determines which functions are included (see §8).

**Training config:**

```python
GRPOConfig(
    learning_rate              = 5e-6,
    gradient_accumulation_steps = 4,
    max_prompt_length          = args.max_length // 4,
    max_completion_length      = 3 * args.max_length // 4,
    num_generations            = args.num_generations,
    max_steps                  = args.max_steps,
    save_steps                 = 50,
    save_total_limit           = 3,
)
```

After training, if a judge was created its LRU cache stats are logged and `judge.close()` is called.

---

### 5.4 PrunerNode

**File:** `hmlcore/nodes/pruner_node.py`
**INPUT_KEYS:** `(model, tokenizer, dataset, args, use_unsloth)`
**OUTPUT_KEYS:** `()` — mutates `model` in-place; sets `args._already_merged = True`

**Skips when:** neither `prune_experts` nor `prune_only` is set.

#### Critical ordering constraint

Topology detection **must run after the LoRA merge**, not before. While the model is a `PeftModel`, PEFT's `LoraModel.__getattr__` proxy hides the base model's attribute paths (e.g. `model.layers` resolves to `LoraModel.model.layers`, not through the documented `model.model.layers` path). `find_decoder_layers()` returns `(None, None)` on the pre-merge model → pruning silently exits. After `merge_and_unload()`, the model is a plain HF model and all attribute paths resolve correctly.

#### Execution order

```
Step 1 — Merge LoRA
    if _is_quantized(model):
        _merge_lora_via_bf16_reload(model, tokenizer)
            → save adapter to tmpdir
            → del model; gc.collect(); cuda.empty_cache()
            → reload base in bf16 with device_map={"": cuda:0}  ← GPU, not CPU
            → PeftModel.from_pretrained(base, tmpdir)
            → merged.merge_and_unload()
    else:
        model.merge_and_unload()  (with bf16 fallback on error)

Step 2 — Flag as merged
    args._already_merged = True

Step 3 — Detect topology  (on the clean bf16 model)
    find_moe_layers(model)     → MoE if any found
    find_decoder_layers(model) → dense transformer if not MoE

Step 4 — Prune
    MoE   → reap_prune_moe(...)
    Dense → drop_dense_layers(...)
    None  → log error, return (model already merged)
```

#### Why `device_map={"": cuda:0}` (not `"cpu"`)?

Unsloth patches attention modules with custom CUDA kernels (`apply_qkv`, `apply_rotary_emb`). These kernels are only initialised at `from_pretrained` time when the target device is CUDA. Loading to CPU then calling `.cuda()` moves tensors but never triggers kernel init — the patched forward methods exist in the class but their underlying functions are `None`. The fix: load directly onto the GPU so initialisation runs correctly.

---

### 5.5 OutputNode

**File:** `hmlcore/nodes/output_node.py`
**INPUT_KEYS:** `(model, tokenizer, args, use_unsloth)`
**OUTPUT_KEYS:** `(finale_dir,)`

Saves to `{output_dir}/finale/`.

#### Save path matrix

| `_already_merged` | `--merge` | `--quantize` | Backend | Method |
|---|---|---|---|---|
| ✓ | * | GGUF quant + Unsloth | Unsloth | `save_pretrained_gguf()` |
| ✓ | * | other | Standard HF | `save_pretrained(safe_serialization=False)` |
| ✗ | ✓ | GGUF quant + Unsloth | Unsloth | `save_pretrained_gguf()` |
| ✗ | ✓ | bf16/f16 + Unsloth | Unsloth | `save_pretrained_merged()` |
| ✗ | ✓ | any + PEFT | PEFT fallback | `_peft_merge_save()` |
| ✗ | ✗ | * | Any | LoRA adapter only |

**`_peft_merge_save()`** — Windows-safe PEFT merge:

1. Saves adapter to tmpdir
2. Reloads base model in bf16 on CPU (`device_map="cpu"`)
3. Re-attaches adapter, calls `merge_and_unload()`
4. Deduplicates shared weight pointers before building state dict
5. Saves as `pytorch_model.bin` via `torch.save()` (bypasses safetensors mmap lock)
6. Saves `config.json` and tokenizer separately

Post-save: `_log_model_stats()` logs a full summary including params, dtype, layers, vocab, MoE info, and disk size.

---

## 6. Model Loading

**File:** `hmlcore/model.py`

`load_model_and_tokenizer(args) → (model, tokenizer, use_unsloth: bool)`

### Unsloth path

```python
FastLanguageModel.from_pretrained(
    model_name         = args.student_model,
    max_seq_length     = args.max_length,
    load_in_4bit       = True,
    gpu_memory_utilization = 0.9,
)
FastLanguageModel.get_peft_model(
    model,
    r                  = args.lora_rank,
    lora_alpha         = args.lora_rank * 2,
    target_modules     = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
)
```

### PEFT fallback

```python
BitsAndBytesConfig(
    load_in_4bit           = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16,   # fp16 if bf16 not supported
    bnb_4bit_quant_type    = "nf4",
)
AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
prepare_model_for_kbit_training(model)
get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_rank*2, ...))
```

**Post-load:** `tokenizer.pad_token = tokenizer.eos_token` if unset.

---

## 7. Dataset Loading & Preprocessing

**File:** `hmlcore/data.py`

### Chat template

`setup_chat_template(tokenizer)` installs a custom Jinja2 template:

```
{system_message}{eos_token}
{user_message}
{REASONING_START}{assistant_response}{eos_token}
```

The key design constraint: `REASONING_START` is emitted identically both when `add_generation_prompt=True` (inference) and when the assistant turn exists (training). This ensures the prompt prefix tokenises identically in both cases, silencing SFTTrainer's tokenisation-mismatch warning.

### Dataset normalisation

`load_and_preprocess_dataset(paths, tokenizer, domain, max_length)` handles:

**Input formats accepted:**
- JSONL with any combination of: `instruction`/`prompt`/`question` + `response`/`output`/`answer`/`solution`
- Conversational (`messages` list with role/content dicts)
- HuggingFace dataset IDs (auto-selects `train`/`cot`/`default`/`test` split)

**Ground-truth extraction:**
```
Response contains </think>...</think>  →  text after closing tag
Response contains </thought>           →  text after closing tag
Response contains #### (GSM8K)         →  text after ####
Otherwise                              →  full response
```

**Output columns:**

| Column | Type | Usage |
|---|---|---|
| `prompt` | `str` | Rendered prompt string → GRPOTrainer |
| `raw_messages` | `list[dict]` | `[{system}, {user}]` → SFTNode chat template |
| `completion` | `str` | Ground-truth answer → reward functions |
| `full_response` | `str` | Original response → SFT formatting |

---

## 8. Reward Functions

**File:** `hmlcore/rewards.py`

All reward functions share the signature:
```python
fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]
```

### Format rewards (all domains)

| Function | Logic | Score range |
|---|---|---|
| `_match_format_exactly` | Regex match: `</reasoning>…<solution>…</solution>[ws]*eos` | 0.0 or +3.0 |
| `match_format_approximately` | ±0.5 per tag present (4 tags total) | −4.0 to +2.0 |

### Math domain

| Function | Logic | Score range |
|---|---|---|
| `_check_math_answer` | Float comparison within 1% tolerance | −2.5 to +5.0 |
| `check_math_working_steps` | Counts step-signal words + equation lines | −1.0 to +2.0 |
| `check_math_units` | Unit presence and matching in answer | −1.0 to +1.5 |
| `check_math_reasoning_quality` | Word count, line count, unique-word ratio | −1.5 to +1.3 |

### Code domain

| Mode | Function | Score range |
|---|---|---|
| With judge | `LMStudioJudge` — calls LM Studio `/v1/chat/completions`, extracts `N/10` score, normalises to [0, 5] | 0.0 to +5.0 |
| Without judge | `check_code_heuristic` — counts keywords (def, class, return, import, etc.) | −3.0 to +3.0 |

### Scene domain

| Function | Logic | Score range |
|---|---|---|
| `check_spatial_precision` | Euclidean distance between predicted and target [x, y] | -2.0 to +5.0 |
| `check_scene_connectivity` | Matches predicted `connect_to` ID with ground truth | -1.0 to +3.0 |

### General domain

- LLM judge if `--judge_model` set (same as code judge, generic prompt)
- Format-only rewards otherwise

### LMStudioJudge

```python
class LMStudioJudge:
    def __init__(self, model, base_url, timeout, cache_size):
        # SHA-256 keyed LRU cache
        # Thread pool for parallel requests

    def score(self, prompt, completion) -> float:
        # Returns cached result or fires HTTP POST to /v1/chat/completions
        # Extracts regex: ([0-9](?:\.[0-9])?|10(?:\.0)?)\s*/?\s*10
        # Normalises: raw_score / 10 * 5.0

    def close(self): ...   # Shuts down thread pool, logs cache stats
```

---

## 9. Pruning Algorithms

### 9.1 REAP — MoE Expert Pruning

**File:** `hmlcore/moe.py`
**Reference:** "REAP the Experts" (Cerebras Research, arXiv 2510.13999)

**Compatible architectures:** Models with `module.gate` (Linear router) + `module.experts` (weight tensors or `nn.ModuleList`). Covers Qwen3-MoE, Mixtral, DeepSeek-MoE, OLMoE, Qwen1.5-MoE.

#### REAP saliency score

For each expert `j` in a MoE layer:

```
S_j = (1 / |X_j|) * Σ_{x ∈ X_j} [ g_j(x) · ‖f_j(x)‖₂ ]

where:
  X_j    = tokens routed to expert j (top-K selection)
  g_j(x) = normalised router gate weight for token x at expert j
  f_j(x) = expert j's output activation vector for token x
```

Higher `S_j` → expert is frequently activated and produces large outputs → more important.

#### `reap_prune_moe(model, tokenizer, dataset, prune_ratio, num_samples, max_cal_length)`

1. `find_moe_layers(model)` — searches `named_modules()` for modules with both a `.gate` (Linear) and `.experts` attribute
2. `compute_reap_scores()` — registers pre-hooks on each MoE layer, runs up to `num_samples` calibration forward passes, accumulates `S_j` for all experts
3. `prune_moe_experts()` — for each MoE layer, keeps the top `(1 - prune_ratio)` experts:
   - `nn.ModuleList` layout (standard HF): replaces with pruned `nn.ModuleList`; slices router gate rows
   - Stacked 3D tensor layout (Unsloth): slices `gate_up_proj`, `down_proj` along expert dim
4. Updates `config.num_experts_per_tok` / `top_k` to clamp to remaining expert count
5. Logs before/after parameter counts and % reduction

---

### 9.2 Bonsai & DLP — Modern Dense Layer Scoring

**File:** `hmlcore/dense_pruner.py`
**Primary Reference:** "Bonsai: Gradient-Free Perturbative Pruning for Large Language Models" (arXiv 2601.04123)
**Secondary Reference:** "DLP: Dynamic Layerwise Pruning via Information Bottleneck" (arXiv 2603.08812)

**Overview:**
Akatsuki replaces ShortGPT's Block Influence (BI) cosine-similarity score with two improvements borrowed from Bonsai and DLP. The **granularity remains block-level** — entire transformer blocks are dropped, not individual attention heads or MLP channels. This preserves GGUF compatibility: llama.cpp requires `num_attention_heads`, `num_key_value_heads`, and `intermediate_size` to match the original architecture constants; only `num_hidden_layers` changes.

**Compatible architectures:** Any dense transformer with a discoverable `nn.ModuleList` of decoder blocks. Covers Llama 3/4, Qwen 3 (dense), Mistral Large/Small, Phi-4, Gemma 4, GPT-2/J, Falcon, Pythia/GPT-NeoX, and multimodal VLMs (text decoder blocks only).

**Incompatible architectures:** Mamba/SSM hybrids (Jamba, Falcon-Mamba, etc.) — block-type positions are fixed in GGUF/llama.cpp architecture definitions; renumbering after layer removal corrupts the SSM ↔ Attention type mapping.

#### Improvement 1 — Bonsai Perturbative Saliency (replaces ShortGPT BI)

ShortGPT's BI score (`1 - cosine_similarity(h_in, h_out)`) measures only how much a block changes its own input. It is insensitive to whether that change actually propagates and matters to downstream layers.

**Bonsai perturbative saliency** measures downstream sensitivity directly. For each block `M` at layer `l` (with the next block `N` at layer `l+1`):

1. Run a clean forward: `h_out = M(h_in)`
2. Inject Gaussian noise scaled to the activation: `h_noisy = h_out + ε`, where `ε ~ σ(h_out) · N(0, I)`
3. Measure downstream propagation: `S_l = E[‖N(h_noisy) − N(h_out)‖₂ / ‖ε‖₂]`

Higher `S_l` → the block's output strongly affects the next layer → more important to preserve. Blocks with low `S_l` are candidates for removal.

For the last interior block, the final (always-preserved) block serves as the downstream `N`.

**Graceful degradation — Unsloth / hook mode:** When the direct-call probe fails and calibration switches to hook mode (full `model.forward()`), injecting mid-pass noise into intermediate activations is not feasible. In hook mode, saliency falls back to the ShortGPT BI cosine metric. DLP entropy weighting still applies in both modes.

#### Improvement 2 — DLP Activation Entropy Weighting (non-uniform allocation)

ShortGPT applies the same `prune_ratio` uniformly across all interior blocks. DLP observes that blocks vary in informational complexity: some perform rich transformations (high activation entropy), others are nearly pass-through.

For each block, activation entropy is computed from the output hidden states:

```
H_l = −Σ p_i · log(p_i)
where  p_i = |h_out_l[i]| / Σ|h_out_l|   (normalised absolute activations)
```

The final ranking score used for block selection:

```
S_final_l = S_l × (H_l + δ)          (δ = 1e-6 to score zero-entropy blocks on sensitivity alone)
```

Blocks with low saliency **and** low entropy rank lowest and are removed first. High-entropy blocks are protected even if their raw saliency is middling — the large entropy multiplier keeps their score above low-H layers. In practice this produces a depth-skewed profile: early layers (high-entropy feature extractors) are preserved; late-middle layers (lower-entropy context compression) are pruned more aggressively.

#### Layer discovery

`find_decoder_layers(model)` tries attribute paths in order:

```
model.layers          LLaMA, Mistral, Qwen2/3, DeepSeek
model.model.layers    PeftModel wrapping the above
transformer.h         GPT-2, GPT-J, Falcon (old)
model.transformer.h   PeftModel wrapping GPT-2
gpt_neox.layers       Pythia, GPT-NeoX
model.gpt_neox.layers
layers                bare model
model.blocks          custom
decoder.layers        T5/BART-style
```

SSM detection: if any block in the found `ModuleList` contains `ssm_conv1d`, `dt_proj`, `A_log`, `x_proj`, `dt_layernorm`, `mixer`, or `conv1d` as a sub-module name → returns `(None, None)` with a clear warning.

#### Calibration — layer-by-layer forward

To avoid failures from framework-level patches (Unsloth CUDA kernels, custom `model.forward` overrides), calibration does NOT call `model(**inputs)`. Instead:

```
1. _get_initial_hidden_states(model, input_ids)
      → tries embed path list: model.embed_tokens, transformer.wte, gpt_neox.embed_in, ...
      → adds absolute position embeddings if model uses them (transformer.wpe, etc.)
      → returns initial hidden state tensor

2. for each layer l:
      h_in = hidden_states.detach()
      h_out = _call_layer(layer_l, h_in)
          → tries: layer(hs), layer(hs, use_cache=False),
                   layer(hs, position_ids=..., use_cache=False), ...
      S_l    += std(h_out − h_in)                          ← sensitivity proxy (residual std)
      H_l    += −Σ p_i log(p_i), p_i = |h_out_i|/Σ|h_out|  ← DLP Shannon entropy
      hidden_states = h_out                               ← clean state propagates forward
      [bonsai_noise reserved: full perturbative scoring (inject ε into h_out, call
       layer_{l+1} twice) is the target implementation — not yet active]
```

If the direct-call probe fails → hook mode (full `model.forward()`, BI cosine score + DLP entropy).  
If ALL samples fail → index-order pruning (drops last layers first) with a clear error log.

#### `drop_dense_layers(model, tokenizer, dataset, prune_ratio, ...)` — unchanged public API

1. Scores all layers via `_compute_layer_importance()` (now Bonsai perturbative + DLP entropy)
2. Always preserves first and last blocks (embedding projection + final norm are disproportionately important)
3. Sorts interior blocks by `S_final_l` ascending → drops `floor(num_layers × prune_ratio)` lowest-ranked blocks
4. Replaces `ModuleList` in-place via `setattr`
5. Updates `config.num_hidden_layers` / `n_layer` / `num_layers` to match
6. Renumbers `layer_idx` on surviving attention modules to fix KV-cache indexing
7. Logs before/after parameter counts and % reduction

#### Dense model compatibility

| Architecture family | Scoring mode | Notes |
|---|---|---|
| LLaMA 3/4, Mistral, Qwen2/3, DeepSeek dense, Phi-4, Gemma 4 | Full Bonsai+DLP | Primary targets |
| GPT-2, GPT-J, Falcon old, Pythia, GPT-NeoX | Full Bonsai+DLP | |
| Multimodal VLMs (Qwen2-VL, LLaVA, InternVL) | Full Bonsai+DLP | Text decoder blocks only; inner tokenizer used for calibration |
| Unsloth-patched models (hook-mode fallback) | BI cosine + DLP entropy | Bonsai perturbative unavailable in hook mode; DLP weighting still applies |
| Mamba/SSM hybrids (Jamba, Falcon-Mamba, Zamba) | Skipped | Block renumbering corrupts SSM↔Attention index mapping in GGUF |

---

### 9.3 PRISM — Data Pruning

**File:** `hmlcore/prism_selector.py`
**Reference:** "PRISM: Self-Pruning Intrinsic Selection Method" (Knyazev, arXiv 2502.12119)

Unlike REAP/ShortGPT which prune the model, PRISM prunes the **data**. It is used to identify semantically redundant samples that provide diminishing returns during the expensive GRPO phase.

#### Selection Criteria

PRISM ranks samples by a redundancy score $R$, derived from the pairwise correlation of hidden state embeddings. Samples with the lowest $R$ are considered the most "informative" and form the **High Quality** tier.

#### Implementation Detail

- **Implicit Re-centering**: Subtracts the mean embedding vector before correlation to stabilize the signal against feature anisotropy.
- **Chunked Processing**: The [N, N] correlation matrix is computed in chunks of 2000xN to keep memory usage below 8GB VRAM even for large datasets.
- **Diverse Seeding**: The output dataset is sorted by redundancy ascending, ensuring that the first samples seen by the SFT trainer are the most semantically diverse seeds.

---

### 9.4 PRISM-DQ — Dynamic Quantization

**File:** `hmlcore/prism_dq.py` (engine) · `hmlcore/nodes/prism_dq_node.py` (pipeline node)  
**Reference:** PRISM Dynamic Quantization as documented by [Ex0bit / PRISM-DQ](https://huggingface.co/Ex0bit/Qwen3.5-PRISM-Dynamic-Quant-GGUF)

Unlike REAP/ShortGPT/PRISM-data which prune the model or the dataset, PRISM-DQ assigns **per-tensor-class GGUF quantization types** that minimise total quantization distortion within a target bits-per-weight (BPW) budget, without requiring calibration data.

#### Key Design Distinction

PRISM-DQ is a **recipe-generation** framework, not an in-memory weight mutation. The analysis runs on the saved BF16 checkpoint (post `OutputNode`) and emits a `llama-quantize --tensor-type` command, not modified weights. This makes it compatible with the llama.cpp ecosystem out of the box.

#### Activation

Runs only when `--prism_dq` is set AND `finale_dir` is populated (i.e. `OutputNode` has saved a merged BF16 checkpoint). Requires `--merge`.

#### 7 Structural Metrics

For each weight tensor $W \in \mathbb{R}^{m \times n}$:

| # | Metric | Formula / Definition |
|---|---|---|
| 1 | **PL-Alpha-Hill** | Hill estimator on the top-$k$ eigenvalues $\lambda_i$ of $W^TW$: $\hat{\alpha} = \left( \frac{1}{k}\sum_i \ln\frac{\lambda_i}{\lambda_{k+1}} \right)^{-1}$ |
| 2 | **Spectral Dominance** | $\sigma_1 / \sum_i \sigma_i$ — rank-1 approximation quality |
| 3 | **OSQE** | MSE of optimal-scale symmetric quantization at 2, 3, 4, 6 bits |
| 4 | **Matrix Imbalance** | $\max(\text{CV}_{\text{rows}}, \text{CV}_{\text{cols}})$ — coefficient of variation |
| 5 | **Fragility** | $\log(\text{OSQE}_{2\text{bit}} / \text{OSQE}_{4\text{bit}})$ |
| 6 | **Boundary Density** | Fraction of weights within 10% of a quantization bin boundary |
| 7 | **Spectral Position Prior** | $\|\sigma_{\max}(W_l)\|_2 \times \|\sigma_{\max}(W_{L-l})\|_2$ — bidirectional depth prior |

All 7 metrics are normalised model-wide by `[min, max]` range and combined into a composite sensitivity score with fixed weights. PL-Alpha-Hill is inverted (lower alpha = heavier tail = more sensitive).

#### Lagrangian Bit Allocator

Binary-searches for a multiplier $\lambda$ such that the BPW constraint is satisfied:

```
for each tensor class c:
    q*(c) = argmin_q [ OSQE(q) × sensitivity(c) + λ × BPW(q) ]
    where q ∈ {Q2_K, Q3_K, IQ4_XS, Q4_K, Q5_K, Q6_K}

converges when |achieved_BPW - target_BPW| < 0.01
```

#### Per-Block Refinement (`--dq_refinement`)

A secondary pass compares each individual block's `OSQE_4` against the class mean. Blocks exceeding 1.5× the mean are upgraded one quant level and added as `-tensor-type "blk.(N).class=TYPE"` overrides in the final recipe.

#### Output

`finale_dir/prism_dq_recipe.sh` — a ready-to-run bash script:

```bash
llama-quantize \
    --tensor-type "attn_q=Q4_K" \
    --tensor-type "attn_v=Q4_K" \
    --tensor-type "ffn_gate=Q3_K" \
    --tensor-type "blk.(18).ffn_down=Q4_K" \  # refinement override
    input_f16.gguf output_PRISM-DQ.gguf Q3_K
```

#### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--prism_dq` | `False` | Enable PRISM-DQ (requires `--merge`) |
| `--target_bpw` | `4.0` | Target average bits-per-weight |
| `--dq_refinement` | `False` | Enable per-block refinement pass |
| `--dq_llama_path` | `None` | Path to `llama-quantize` binary for auto-invocation |
| `--dq_input_gguf` | `None` | F16 GGUF to quantize (required for auto-invocation) |

---

## 9.5 X-Token — Projection-Guided Cross-Tokenizer Knowledge Distillation

**File:** `hmlcore/xtoken/` — `distiller.py`, `projection.py`, `node.py`
**Reference:** X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation

X-Token enables knowledge distillation between models that use **different tokenizers**, addressing a limitation of traditional distillation methods that require identical vocabularies. It achieves this through projection-based alignment of embedding spaces.

### Core Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    X-Token Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Teacher Model (Frozen)              Student Model (Trainable)  │
│  ┌──────────────────┐                ┌──────────────────┐      │
│  │  Tokenizer A     │                │  Tokenizer B     │      │
│  └────────┬─────────┘                └────────┬─────────┘      │
│           │                                   │                  │
│           ▼                                   ▼                  │
│  ┌──────────────────┐                ┌──────────────────┐      │
│  │ Embeddings (d_T) │                │ Embeddings (d_S) │      │
│  │   d_T = 4096     │                │   d_S = 2048     │      │
│  └────────┬─────────┘                └──────────────────┘      │
│           │                                   │                  │
│           │    ┌───────────────────┐          │                  │
│           │    │   Projection      │          │                  │
│           │    │   Aligner         │          │                  │
│           └───▶│  (Learned Map)    │◀─────────┘                  │
│                │                   │                            │
│                └───────────────────┘                            │
│                        │                                        │
│                        ▼                                        │
│              ┌──────────────────┐                               │
│              │  Alignment Loss  │                               │
│              │  + Matching Loss │                               │
│              └──────────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.5.1 Architecture Components

#### ProjectionAligner

Projects teacher embeddings to student embedding space:

- **Linear**: Direct projection matrix `W ∈ ℝ^(d_T × d_S)`
- **MLP**: Two-layer network with GELU activation
- **Identity**: Pass-through when dimensions match

```python
from hmlcore.xtoken import ProjectionAligner

aligner = ProjectionAligner(
    teacher_embed_dim=4096,
    student_embed_dim=2048,
    hidden_dim=3072,
    projection_type="mlp",
)
```

**Multi-Teacher Support:**

For multiple teachers, set `num_teachers` and optionally `teacher_weights`:

```python
aligner = ProjectionAligner(
    teacher_embed_dim=4096,
    student_embed_dim=2048,
    hidden_dim=3072,
    projection_type="mlp",
    num_teachers=3,
    device=device,
    dtype=torch.bfloat16,
)
# Learnable weights initialized to equal values: [1/3, 1/3, 1/3]
```

#### CrossTokenizerMatcher

Learnable token matching mechanism:

- Maintains separate embedding tables for teacher and student vocabularies
- Computes cosine similarity between aligned token representations
- Enables token-level knowledge transfer across different tokenizers

#### XTokenDistiller

Main distillation orchestrator:

- Wraps teacher (frozen) and student (trainable) models
- Manages projection and matching modules
- Implements training loop with configurable losses

```python
from hmlcore.xtoken import XTokenDistiller, XTokenConfig

config = XTokenConfig(
    projection_type="mlp",
    hidden_dim=3072,
    alignment_weight=1.0,
    matching_weight=0.5,
    learning_rate=2e-4,
    temperature=2.0,
    num_teachers=1,              # Set > 1 for multi-teacher
    teacher_weights=[0.4, 0.3, 0.3],  # Optional weights
)

distiller = XTokenDistiller(
    teacher_model=teacher,
    student_model=student,
    teacher_tokenizer=teacher_tokenizer,
    student_tokenizer=student_tokenizer,
    config=config,
    num_teachers=config.num_teachers,
)
```

#### XTokenNode

Pipeline integration node for hmlcore:

```python
from hmlcore.xtoken import XTokenNode

node = XTokenNode(
    teacher_model_path="Qwen/Qwen3-35B-A3B",
    projection_type="mlp",
    hidden_dim=3072,
)
```

### 9.5.2 Training Process

#### Loss Components

```
Total Loss = Alignment Loss + Matching Loss + CE Loss

Where:
- Alignment Loss: Cosine similarity between projected teacher 
  embeddings and student embeddings (layer-wise)
- Matching Loss: Token-level similarity across different tokenizers
- CE Loss: Standard cross-entropy on student model outputs
```

#### Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each batch:                                                │
│                                                                 │
│  1. Forward Pass                                                │
│     ├── Teacher (frozen):  embed → layers → hidden states     │
│     └── Student (train):   embed → layers → hidden states     │
│                                                                 │
│  2. Projection                                                  │
│     └── Project teacher embeddings to student space           │
│                                                                 │
│  3. Compute Losses                                              │
│     ├── Alignment: cos_sim(projected_teacher, student)        │
│     ├── Matching:   token_similarity_loss                     │
│     └── CE:         standard_cross_entropy                    │
│                                                                 │
│  4. Backward Pass                                               │
│     └── Gradient updates to student + projection + matcher    │
│                                                                 │
│  5. Optimization                                                │
│     ├── Gradient clipping (max_norm=1.0)                      │
│     ├── Optimizer step                                        │
│     └── LR scheduler step                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.5.3 Pipeline Integration

X-Token integrates seamlessly with the existing Akatsuki node-graph pipeline:

```
InputNode
    ↓
XTokenNode (NEW) ── Setup distillation pipeline
    ↓
XTokenDistillNode (NEW) ── Run distillation training
    ↓
SFTNode (existing)
    ↓
GRPONode (existing)
    ↓
PrunerNode (existing)
    ↓
OutputNode (existing)
```

**Pipeline Compatibility:**

| Combination | Supported | Notes |
|---|---|---|
| XToken + SFT | ✅ | XToken runs before SFT |
| XToken + GRPO | ✅ | XToken runs before GRPO |
| XToken + Pruning | ✅ | XToken runs before pruning |
| XToken + PRISM | ✅ | PRISM selection before XToken |

### 9.5.4 CLI Integration

New arguments for X-Token distillation:

| Flag | Default | Description |
|---|---|---|
| `--teacher_model` | `None` | Teacher model path/ID (required for X-Token) |
| `--xtoken_enabled` | `False` | Enable X-Token distillation |
| `--xtoken_projection` | `mlp` | Projection type: linear, mlp, identity |
| `--xtoken_hidden_dim` | `None` | Hidden dimension for MLP projection |
| `--xtoken_save_steps` | `500` | Checkpoint save interval |
| `--xtoken_epochs` | `3` | Number of distillation epochs |

### 9.5.5 Performance Considerations

**Memory Usage:**

| Component | Memory Estimate |
|-----------|-----------------|
| Teacher model (frozen) | ~8 GB (BF16) |
| Student model (trainable) | ~2 GB (BF16) |
| Projection module | ~50 MB |
| Matcher module | ~20 MB |
| **Total (excluding optimizer states)** | ~10 GB |

**Training Speed:**

- **Forward pass**: Teacher inference + Student inference + Projection
- **Backward pass**: Gradient computation for student + projection + matcher
- **Typical throughput**: 10-50 steps/sec depending on model sizes and hardware

**GPU Requirements:**

| Model Size | Minimum GPU | Recommended |
|------------|-------------|-------------|
| Small (Qwen3-0.6B) | 4 GB | 8 GB |
| Medium (Qwen3-4B) | 8 GB | 16 GB |
| Large (Qwen3-35B) | 24 GB | 40 GB |

### 9.5.6 Use Cases

1. **Teacher-Student Distillation**: Transfer knowledge from large teacher (Qwen3-35B) to smaller student (Qwen3-0.6B)
2. **Cross-Tokenizer Alignment**: Align models with different vocabularies (e.g., Qwen → Mistral)
3. **Layer-wise Alignment**: Match specific transformer layers between models
4. **Token-level Matching**: Align token representations across different tokenizers

### 9.5.7 Multi-Teacher Distillation

X-Token supports combining knowledge from **multiple teacher models** into a single student:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Teacher Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Teacher 1 (Frozen)    Teacher 2 (Frozen)    Teacher N (Frozen)│
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐   │
│  │ Tokenizer A │       │ Tokenizer B │       │ Tokenizer N │   │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘   │
│         │                    │                   │              │
│         ▼                    ▼                   ▼              │
│  ┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐  │
│  │ Embeddings  │       │ Embeddings  │       │ Embeddings  │  │
│  │  (d_T1)     │       │  (d_T2)     │       │  (d_TN)     │  │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘  │
│         │                    │                   │              │
│         └───────┬────────────┼───────────────────┘              │
│                 │                                               │
│         ┌───────┴───────┐                                       │
│         │  Projection   │                                       │
│         │  + Weighted   │                                       │
│         │  Aggregation  │                                       │
│         └───────┬───────┘                                       │
│                 │                                               │
│                 ▼                                               │
│      ┌──────────────────┐                                       │
│      │  Student Model   │                                       │
│      │  (Trainable)     │                                       │
│      │  Embeddings (d_S)│                                       │
│      └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Learned Aggregation**: Per-teacher weights are learnable parameters
- **Equal Weighting**: Default initializes all teachers equally
- **Flexible Weights**: Custom weights can be specified via `teacher_weights`

**Configuration:**

```python
config = XTokenConfig(
    num_teachers=3,
    teacher_weights=[0.4, 0.3, 0.3],  # Optional, defaults to equal
    # ... other config
)
```

---

## 10. Cross-Cutting Concerns

### 10.1 Pre-flight Compatibility Check

**File:** `hmlcore/nodes/pipeline_check.py`

Runs in `InputNode` after model load, before any training. Never raises — wrapped in a broad `except` so a diagnostics failure cannot block the pipeline.

**Detects:**
- MoE topology → REAP pruning available
- Mamba/SSM hybrid → pruning will be skipped + explains why
- Dense transformer → Bonsai/DLP structural pruning available
- Multimodal processor → GRPO will be skipped
- BnB quantization type
- PEFT/LoRA trainable parameter count

**Output format:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pipeline Compatibility Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model:       Qwen3MoeForCausalLM
  Topology:    MoE  (4 expert layer(s))
  Tokenizer:   Qwen2Tokenizer
  Params:      7,614,767,104 total  |  12,582,912 trainable (LoRA/PEFT)
  Precision:   BitsAndBytes 4-bit (uint8 packed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SFT        ✅  will run
  GRPO       ✅  will run
  Pruning    ✅  REAP expert pruning (MoE)  ratio=0.30
  Output     ✅  will run
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 10.2 Stage Model Snapshots

**File:** `hmlcore/nodes/model_info.py`

`log_stage_model_info(stage, model, tokenizer, dataset)` is called by `GraphRunner` after every node except `OutputNode` (which has its own detailed stats block).

Reports:
- Model class and precision label (4-bit BnB / bfloat16 / float16 / etc.)
- Total and trainable parameter counts (trainable > 0 → tagged as LoRA adapter active)
- Memory estimate in GB
- Architecture: layers, hidden size, attention heads, FFN size
- Tokenizer: vocab size, context length
- MoE info: expert count, top-K
- Dataset size

---

### 10.3 Windows Compatibility

Several explicit workarounds are in place for Windows:

| Issue | Location | Fix |
|---|---|---|
| Safetensors mmap lock — file handle not released between save and re-load | `output_node._peft_merge_save()` | Save as `pytorch_model.bin` via `torch.save()` |
| Same mmap issue on already-merged path | `output_node.OutputNode.run()` | `model.save_pretrained(..., safe_serialization=False)` |
| `device_map="cpu"` bypasses Unsloth CUDA init | `pruner_node._merge_lora_via_bf16_reload()` | `device_map={"": cuda:0}` (load directly to GPU) |
| Shared weight pointers in state dict | `output_node._peft_merge_save()` | Deduplicate via `data_ptr()` before `torch.save()` |

---

### 10.4 Unsloth Integration

Unsloth is used as an optional accelerated backend. The integration points:

| Location | Behaviour |
|---|---|
| `hmlcore/model.py` | Tries `from unsloth import FastLanguageModel` — if installed, uses Unsloth 4-bit; else PEFT fallback |
| `hmlcore/finetuner/model.py` | `import unsloth` must precede `from unsloth import ...` (import-order constraint) |
| `grpo_node.py` | Sets `model.base.warnings_issued = {}` if missing |
| `moe.py` | `prune_moe_experts` handles both standard `nn.ModuleList` expert layout AND Unsloth's stacked 3D tensor layout |
| `pruner_node._merge_lora_via_bf16_reload` | Uses `device_map={"": cuda:0}` — Unsloth CUDA kernels only initialise at `from_pretrained` time on a CUDA device |
| `output_node` | Falls back from Unsloth's `save_pretrained_gguf`/`save_pretrained_merged` to standard PEFT merge if those raise |

---

## 11. Data Flow Diagram

```
CLI args
   │
   ▼
InputNode
   ├─ load_model_and_tokenizer()  →  4-bit PeftModel + tokenizer
   ├─ run_pipeline_check()        →  compatibility table (stdout)
   ├─ setup_chat_template()       →  custom Jinja2 template installed
   ├─ load_and_preprocess_dataset() → HF Dataset {prompt, raw_messages, completion, full_response}
   └─ select_with_prism()         →  filters dataset using student hidden states
   │
   ▼  ctx: model(4bit PeftModel), tokenizer, dataset (PRISM-filtered), use_unsloth, is_multimodal
   │
SFTNode  (skips if disable_sft / prune_only / grpo_checkpoint)
   ├─ first 100 examples
   ├─ format responses into <reasoning>/<solution> structure
   ├─ SFTTrainer (1 epoch, lr=2e-4)
   └─ saves adapter + sentinel
   │
   ▼  ctx: model(4bit PeftModel, SFT-trained)
   │
GRPONode  (skips if prune_only / is_multimodal)
   ├─ build_reward_functions()  →  domain-specific rewards
   ├─ GRPOTrainer (max_steps, lr=5e-6, 4 rollouts/prompt)
   └─ judge.close() if applicable
   │
   ▼  ctx: model(4bit PeftModel, GRPO-trained)
   │
PrunerNode  (skips if !prune_experts && !prune_only)
   ├─ merge LoRA:
   │     if quantized → _merge_lora_via_bf16_reload() [reload base on GPU]
   │     else         → model.merge_and_unload()
   ├─ args._already_merged = True
   ├─ detect topology (on clean bf16 model):
   │     MoE?   → reap_prune_moe()    [REAP calibration + expert removal]
   │     Dense? → drop_dense_layers() [Bonsai/DLP calibration + layer removal]
   │     None   → skip (model already merged)
   └─ ctx: model(bf16 merged, pruned)
   │
   ▼  ctx: model(merged bf16 or 4bit PeftModel), args._already_merged
   │
OutputNode
   ├─ if already_merged → save_pretrained(safe_serialization=False) or GGUF
   ├─ if merge+unsloth  → save_pretrained_merged() / save_pretrained_gguf()
   ├─ if merge+peft     → _peft_merge_save() [bf16 reload on CPU, pytorch_model.bin]
   └─ else              → save LoRA adapter only
   │
   ▼  {output_dir}/finale/
         pytorch_model.bin or model.safetensors
         config.json
         tokenizer files
         [model.gguf]  (if GGUF export)
```

---

## 12. Configuration Reference

### Global tags (`hmlcore/config.py`)

```python
REASONING_START = "<reasoning>"
REASONING_END   = "</reasoning>"
SOLUTION_START  = "<solution>"
SOLUTION_END    = "</solution>"
SYSTEM_PROMPT   = "You are given a problem. Think about the problem and provide your "
                  "working out. Place it between {REASONING_START} and {REASONING_END}. "
                  "Then, provide your solution between {SOLUTION_START} and {SOLUTION_END}."
```

All tags are configurable via CLI (`--r_start`, `--r_end`, `--s_start`, `--s_end`, `--system_prompt`).

### Pruning Options (`hmlcore/config.py`)

| Flag | Description |
|---|---|
| `--prune_experts` | Enable REAP expert pruning (MoE models) |
| `--prune_dense` | Enable Bonsai/DLP structural pruning (Dense models) |
| `--prune_ratio` | Target sparsity (0.0 to 1.0). Default: 0.5 |
| `--calibration_samples` | Number of samples for scoring. Default: 128 |
| `--bonsai_noise` | Noise $\epsilon$ std-dev for saliency calibration. Default: 1e-4 |
| `--dlp_scale` | Coefficient for informational entropy weighting. Default: 1.0 |
| `--dynamicquant` | 1-bit degrade targeted modules instead of removing them |

### Output directory structure

```
{output_dir}/
├── sft/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── sft_complete          ← sentinel file
│   └── checkpoint-N/         ← intermediate checkpoints
├── grpo/
│   ├── checkpoint-N/
│   └── trainer_state.json
└── finale/
    ├── pytorch_model.bin     ← merged weights (Windows safe)
    ├── config.json
    ├── tokenizer*.json
    └── [model.gguf]
```

---

## 13. Architecture Constraints & Known Limitations

### Pruning compatibility matrix

| Model type | REAP | Bonsai/DLP | Notes |
|---|---|---|---|
| MoE (Qwen3-MoE, Mixtral, OLMoE, DeepSeek-MoE) | ✅ | ✗ | REAP requires routing signals |
| Dense transformer (LLaMA, Qwen2/3, Mistral, Phi-4, Gemma 4, GPT-2, Falcon, Pythia) | ✗ | ✅ | Block-level drop; GGUF-compatible |
| Mamba/SSM hybrid (Jamba, Falcon-Mamba, Zamba) | ✗ | ✗ | Block-position semantics prevent renumbering |
| Multimodal (Qwen2-VL, LLaVA, InternVL) | ✗ | ✅ (text blocks) | Calibration uses inner text tokenizer |

### GRPO compatibility matrix

| Model type | GRPO | Reason |
|---|---|---|
| Any text-only model | ✅ | |
| Multimodal / VLM | ✗ | `compute_3d_position_ids` fails on text-only rollouts |

### LoRA merge strategies

| Model state | Strategy |
|---|---|
| 4-bit BnB quantized | Reload base in bf16 on GPU → re-attach adapter → `merge_and_unload()` |
| Float (bf16/fp16) | Direct `merge_and_unload()` with bf16 reload as fallback |
| Already merged | Pass-through (no re-merge) |

### Key technical constraints

1. **`import unsloth` must precede `from unsloth import ...`** — Unsloth patches transformers globally at import time; partial import before the full `import unsloth` causes attribute errors.
2. **Topology detection after merge** — `find_decoder_layers()` fails on `PeftModel`-wrapped models due to PEFT's `__getattr__` proxy.
3. **Unsloth CUDA init requires GPU at load time** — `device_map="cpu"` + `.cuda()` leaves CUDA kernels uninitialised.
4. **Safetensors mmap on Windows** — any save that will be re-opened in the same process must use `safe_serialization=False`.
5. **Mamba layer renumbering** — GGUF/llama.cpp hardcodes SSM vs Attention block types by index; dropping and renumbering layers produces unloadable files.
6. **Calibration sample minimum** — REAP and Bonsai/DLP both default to 128 calibration samples; fewer samples produce less reliable importance scores but pruning still runs.
7. **Unsloth/PEFT Trainer Compatibility** — When Unsloth is installed but disabled via `--disable_unsloth`, its global monkey-patches to TRL trainers (SFTTrainer/GRPOTrainer) remain active. These patches expect models to have `.for_training()` and `.for_inference()` methods. The pipeline implements these as "shims" in `hmlcore/model.py` to ensure standard PEFT models remain compatible with the patched trainers.

---

## 14. Addendum: Pipeline Recipes & Combinations

This section provides practical CLI samples for common use-cases and valid node combinations.

### 14.1 Standard Reasoning Pipeline (SFT + GRPO)
The default path for dense LLMs (e.g., Llama-3, Qwen-2.5). Performs a short SFT warm-up followed by RL.
```bash
python ohm_finetuner.py \
    --student_model models/Llama-3.1-8B \
    --datasets datasets/reasoning_data.jsonl \
    --domain math \
    --max_steps 500
```

### 14.2 Reinforcement Learning Only (Skip SFT)
Skips SFT; useful if the model already has base reasoning capabilities or you are resuming from an existing SFT adapter.
```bash
python ohm_finetuner.py \
    --student_model models/DeepSeek-R1-Distill-Qwen-7B \
    --datasets datasets/rl_data.jsonl \
    --disable_sft \
    --max_steps 1000
```

### 14.3 Data-Pruned Training (PRISM)
Uses the PRISM engine to identify and remove redundant semantic samples before training starts to optimize GPU compute.
```bash
python ohm_finetuner.py \
    --student_model models/qwen2.5-7b \
    --datasets datasets/large_unfiltered.jsonl \
    --prism_select \
    --prism_tier high \
    --max_steps 200
```

### 14.4 Model Pruning Only (REAP/Bonsai-DLP)
Calibrates and removes redundant experts (MoE) or layers (Dense) without any training. Requires `--merge` to save.
```bash
python ohm_finetuner.py \
    --student_model models/Mixtral-8x7B \
    --datasets datasets/calibration.jsonl \
    --prune_only \
    --prune_ratio 0.5 \
    --merge
```

### 14.5 Dynamic 1-Bit Experts (DynamicQuant)
Instead of removing experts, it degrades redundant ones to 1-bit weights in-place. Pair with IQ-type GGUF exports for maximum compression.
```bash
python ohm_finetuner.py \
    --student_model models/Qwen-MoE \
    --datasets datasets/calib.jsonl \
    --prune \
    --dynamicquant \
    --merge \
    --quantize iq1_m
```

### 14.6 High-Precision Merge & GGUF Export
Merges LoRA weights and exports a quantized GGUF file using Unsloth's accelerated conversion.
```bash
python ohm_finetuner.py \
    --student_model models/qwen2-7b \
    --datasets datasets/math.jsonl \
    --merge \
    --quantize q4_k
```

### 14.7 PRISM-DQ Recipe Generation
Generates a highly optimized quantization recipe based on 7 structural weight metrics. Requires a merged BF16 model.
```bash
python ohm_finetuner.py \
    --student_model models/merged_bf16 \
    --datasets datasets/dummy.jsonl \
    --merge \
    --prism_dq \
    --target_bpw 3.5 \
    --dq_refinement
```

### 14.8 LLM-as-Judge Feedback
Enables high-quality reasoning feedback for domains like Code where rule-based scoring is difficult.
```bash
python ohm_finetuner.py \
    --student_model models/Llama-3-8B \
    --datasets datasets/code_problems.jsonl \
    --domain code \
    --judge_model "gpt-4o" \
    --judge_url "http://localhost:1234/v1"
```

### 14.9 Qwopus Mode (Qwen3-Thinking)
Configures reasoning tags and training parameters to align with the Qwen3/Qwopus thinking-block standard.
```bash
python ohm_finetuner.py \
    --student_model models/Qwen3-7B \
    --datasets datasets/qwopus.jsonl \
    --qwen_jack
```

