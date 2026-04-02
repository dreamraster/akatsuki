# Akatsuki — Technical Architecture

> GRPO-based LLM distillation and pruning pipeline
> By dreamraster · dreaMSCend

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
   - [ShortGPT — Dense Layer Dropping](#92-shortgpt--dense-layer-dropping)
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
- **Post-training pruning** — REAP for MoE architectures, ShortGPT for dense transformers
- **Resume at any stage** — sentinel files and HF checkpoint detection allow resuming SFT or GRPO mid-run

---

## 2. Project Layout

```
akatsuki/
├── ohm_finetuner.py          # Main entry point — thin wrapper around the pipeline
├── ohm_distiller.py          # Standalone distillation script
├── ohm_databuilder.py        # Dataset construction utilities
├── ohm_datapreprocessor.py   # Raw data preprocessing
├── vlm_scene_builder.py      # Synthetic 2D scene dataset generator
├── ARCHITECTURE.md           # This document
│
└── hmlcore/                  # Core library
    ├── __init__.py            # Version (0.3.0)
    ├── config.py              # CLI argument parser, global prompt tags, apply_args()
    ├── model.py               # Model + tokenizer loading (Unsloth / PEFT)
    ├── data.py                # Dataset loading, schema normalisation, chat template
    ├── trainer.py             # SFT + GRPO training wrappers, checkpoint helpers
    ├── rewards.py             # All reward functions + LMStudioJudge
    ├── moe.py                 # REAP expert pruning for MoE models
    ├── dense_pruner.py        # ShortGPT layer dropping for dense transformers
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

| Group | Flag | Default | Description |
|---|---|---|---|
| **Model** | `--student_model` | *(required)* | HF hub ID or local path |
| | `--lora_rank` | `32` | LoRA rank (alpha = rank×2) |
| | `--disable_unsloth` | `False` | Force PEFT fallback |
| **Data** | `--datasets` | *(required)* | Comma-separated paths or HF IDs |
| | `--domain` | `math` | `math` · `code` · `general` |
| | `--max_length` | `2048` | Token budget (prompt + completion) |
| **Training** | `--max_steps` | `1` | Total GRPO optimiser steps |
| | `--batch_size` | `1` | Per-device batch size |
| | `--num_generations` | `4` | Rollouts per prompt (GRPO group size) |
| | `--disable_sft` | `False` | Skip SFT warm-up |
| | `--resume` | `False` | Auto-detect and resume checkpoints |
| **Save** | `--merge` | `False` | Merge LoRA into base weights |
| | `--quantize` | `bf16` | `bf16` · `f16` · `q8_0` · `q4_k` |
| **Pruning** | `--prune_ratio` | `None` | Fraction of experts/layers to drop (auto-enables pruning) |
| | `--prune_experts` | `False` | Enable pruning explicitly |
| | `--prune_only` | `False` | Skip SFT+GRPO, prune + save only |
| | `--calibration_samples` | `128` | Samples for pruning calibration |
| **Judge** | `--judge_model` | `None` | LM Studio model name for code/general scoring |
| | `--judge_url` | `localhost:1234` | LM Studio API base URL |
| | `--judge_timeout` | `60` | Per-request timeout (s) |
| | `--judge_cache_size` | `2048` | SHA-256 response cache (LRU) |

**`apply_args(args)` side-effects:**

- `--prune_ratio N` → implicitly sets `prune_experts = True`
- `--prune_only` → implies `prune_experts = True`, `disable_sft = True`
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
```

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

### 9.2 ShortGPT — Dense Layer Dropping

**File:** `hmlcore/dense_pruner.py`
**Reference:** "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect" (Men et al. 2024, arXiv 2403.03853)

**Compatible architectures:** Any dense transformer with a discoverable `nn.ModuleList` of decoder blocks. Covers LLaMA/2/3, Qwen2/3 (dense), Mistral, Phi-2/3, Gemma/2, GPT-2/J/NeoX, Pythia, Falcon, DeepSeek (dense).

**Incompatible architectures:** Mamba/SSM hybrids (Jamba, Falcon-Mamba, etc.) — block-type positions are fixed in GGUF/llama.cpp architecture definitions; renumbering after layer removal corrupts the SSM ↔ Attention type mapping.

#### Layer importance score

For each transformer block `l`:

```
I_l = 1 − mean_token( cosine_similarity(h_in_l, h_out_l) )

where:
  h_in_l  = hidden state entering block l
  h_out_l = hidden state exiting block l
```

High `I_l` → block substantially transforms its input → important to keep.
Low `I_l` → block is "transparent" (near-identity) → candidate for removal.

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

2. for each layer:
      h_in = hidden_states.detach()
      hidden_states = _call_layer(layer, hidden_states)
          → tries: layer(hs), layer(hs, use_cache=False),
                   layer(hs, attention_mask=None, use_cache=False),
                   layer(hs, position_ids=..., use_cache=False),
                   layer(hs, attention_mask=None, position_ids=..., use_cache=False)
      h_out = hidden_states.detach()
      I_l += 1 - cosine_similarity(h_in, h_out).mean()
```

If ALL samples fail (e.g. unexpected architecture), falls back to index-order pruning (drops last layers first) with a clear error log rather than silently no-op'ing.

#### `drop_dense_layers(model, tokenizer, dataset, prune_ratio, ...)`

1. Scores all layers via `_compute_layer_importance()`
2. Always preserves first and last blocks (embedding projection + final norm are disproportionately important)
3. Sorts interior layers by importance ascending → drops the `floor(num_layers × prune_ratio)` least important
4. Replaces `ModuleList` in-place via `setattr`
5. Updates `config.num_hidden_layers` / `n_layer` / `num_layers` to match
6. Logs before/after parameter counts and % reduction

---

## 10. Cross-Cutting Concerns

### 10.1 Pre-flight Compatibility Check

**File:** `hmlcore/nodes/pipeline_check.py`

Runs in `InputNode` after model load, before any training. Never raises — wrapped in a broad `except` so a diagnostics failure cannot block the pipeline.

**Detects:**
- MoE topology → REAP pruning available
- Mamba/SSM hybrid → pruning will be skipped + explains why
- Dense transformer → ShortGPT available
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
   └─ load_and_preprocess_dataset() → HF Dataset {prompt, raw_messages, completion, full_response}
   │
   ▼  ctx: model(4bit PeftModel), tokenizer, dataset, use_unsloth, is_multimodal
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
   │     Dense? → drop_dense_layers() [ShortGPT calibration + layer removal]
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

| Model type | REAP | ShortGPT | Notes |
|---|---|---|---|
| MoE (Qwen3-MoE, Mixtral, OLMoE, DeepSeek-MoE) | ✅ | ✗ | REAP requires routing signals |
| Dense transformer (LLaMA, Qwen2, Mistral, Phi, Gemma, GPT-2) | ✗ | ✅ | |
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
6. **Calibration sample minimum** — REAP and ShortGPT both default to 128 calibration samples; fewer samples produce less reliable importance scores but pruning still runs.
