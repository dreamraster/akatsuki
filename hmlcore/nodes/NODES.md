# hmlcore/nodes — Node-Graph Pipeline

Modular, node-graph–style execution pipeline for `ohm_hmlcore`.
Every pipeline stage is a self-contained `BaseNode` subclass that reads from and writes to a shared `NodeContext` dict.  `GraphRunner` topologically sorts the nodes and executes them in dependency order.

---

## Architecture

```
hmlcore/nodes/
├── base.py           BaseNode ABC + NodeError
├── context.py        NodeContext TypedDict + make_context()
├── runner.py         GraphRunner (Kahn's algorithm + executor)
├── input_node.py     InputNode   — load model / tokenizer / dataset
├── sft_node.py       SFTNode     — SFT warm-up stage
├── grpo_node.py      GRPONode    — GRPO RL stage
├── pruner_node.py    PrunerNode  — REAP MoE expert pruning
├── output_node.py    OutputNode  — save / merge / GGUF export
└── __init__.py       public re-exports

hmlcore/graph_run.py            entry point: python -m hmlcore.graph_run
```

---

## Entry point

```bash
# SFT + GRPO (default)
python -m hmlcore.graph_run \
    --student_model models/qwen-bnb-4 \
    --datasets datasets/my.jsonl \
    --domain code

# Merge to bf16 HF checkpoint
python -m hmlcore.graph_run ... --merge --quantize bf16

# Merge to GGUF (Unsloth)
python -m hmlcore.graph_run ... --merge --quantize q4_k

# REAP pruning only (skip SFT + GRPO)
python -m hmlcore.graph_run ... --prune_only

# Full pipeline: SFT → GRPO → REAP → GGUF
python -m hmlcore.graph_run ... --prune_experts --merge --quantize q4_k
```

---

## Key abstractions

### `BaseNode`

```python
class BaseNode(ABC):
    NAME: str = ""
    INPUT_KEYS:  tuple[str, ...] = ()   # keys this node reads from context
    OUTPUT_KEYS: tuple[str, ...] = ()   # keys this node writes to context

    def should_run(self, ctx) -> bool: ...   # override to skip conditionally
    def run(self, ctx) -> None: ...          # implement stage logic
```

- Nodes communicate **only** through `NodeContext` — no direct references between nodes.
- `should_run()` returning `False` skips the node without error.
- `NodeError` is raised on unrecoverable failure; `GraphRunner` surfaces it cleanly.

### `NodeContext`

Plain `dict` (typed via `TypedDict`) shared across all nodes:

| Key | Type | Set by |
|---|---|---|
| `args` | `argparse.Namespace` | caller / `make_context()` |
| `model` | any | `InputNode` |
| `tokenizer` | any | `InputNode` |
| `use_unsloth` | `bool` | `InputNode` |
| `is_multimodal` | `bool` | `InputNode` |
| `dataset` | HF `Dataset` | `InputNode` |
| `sft_dir` | `str` | `InputNode` |
| `grpo_dir` | `str` | `InputNode` |
| `sft_checkpoint` | `str \| None` | `InputNode` |
| `grpo_checkpoint` | `str \| None` | `InputNode` |
| `finale_dir` | `str` | `OutputNode` |

### `GraphRunner`

```python
runner = GraphRunner([InputNode(), SFTNode(), GRPONode(), PrunerNode(), OutputNode()])
ctx = runner.run(make_context(args))
```

1. Builds a dependency DAG: edge `P → C` exists when `P.OUTPUT_KEYS ∩ C.INPUT_KEYS ≠ ∅`.
2. Kahn's BFS produces a valid topological order.
3. For each node: calls `should_run(ctx)` → if True, calls `run(ctx)`.
4. Wraps unexpected exceptions in `NodeError` for clean error surfacing.

---

## Node catalogue

### `InputNode`

**Produces:** `model`, `tokenizer`, `use_unsloth`, `is_multimodal`, `dataset`, `sft_dir`, `grpo_dir`, `sft_checkpoint`, `grpo_checkpoint`

- Calls `hmlcore.model.load_model_and_tokenizer()`
- Detects multimodal models (`"ConditionalGeneration"` in class name or `vision_config` on config)
- Calls `hmlcore.data.setup_chat_template()` and `load_and_preprocess_dataset()`
- Resolves resume checkpoints when `--resume` is set

### `SFTNode`

**Consumes:** `model`, `tokenizer`, `dataset`, `args`, `sft_dir`, `sft_checkpoint`, `grpo_checkpoint`

Skipped when:
- `--disable_sft` is set
- `--prune_only` is set
- `grpo_checkpoint` is present (SFT weights already baked in)
- `sft/sft_complete` sentinel file exists → loads saved adapter instead

Fixes vs legacy `hmlcore/trainer.run_sft()`:
- `raw_messages` Arrow dict-of-lists normalisation
- `load_from_cache_file=False` on `map()` to bypass stale HF cache
- Guard for empty dataset slice

### `GRPONode`

**Consumes:** `model`, `tokenizer`, `dataset`, `args`, `grpo_dir`, `grpo_checkpoint`, `is_multimodal`

Skipped when:
- `--prune_only` is set
- `is_multimodal` is True (VLM `compute_3d_position_ids` requires visual tokens)

Builds reward functions via `hmlcore.rewards.build_reward_functions()`.
`LMStudioJudge` is closed in a `finally` block.

### `PrunerNode`

**Consumes:** `model`, `tokenizer`, `dataset`, `args`, `use_unsloth`

Skipped when `--prune_experts` is not set.

Stages:
1. **Pre-check** — `find_moe_layers(model)` before any merge. If empty → log error + soft skip (no 4-bit merge warning).
2. **LoRA merge** — `model.merge_and_unload()`.
3. **REAP calibration + pruning** — `hmlcore.moe.reap_prune_moe()`.

Sets `args._already_merged = True` so `OutputNode` skips re-merging.

### `OutputNode`

**Consumes:** `model`, `tokenizer`, `args`, `use_unsloth`
**Produces:** `finale_dir`

Save matrix:

| `--merge` | `--quantize` | Unsloth? | Result |
|---|---|---|---|
| no | — | — | LoRA adapter only |
| yes | `bf16` | yes | `save_pretrained_merged(..., save_method="bf16")` |
| yes | `f16`/`q8_0`/`q4_k` | yes | `save_pretrained_gguf()` → `.gguf` |
| yes | any | no | `merge_and_unload()` + `save_pretrained()` |
| — | `f16`/`q8_0`/`q4_k` | yes, `_already_merged` | `save_pretrained_gguf()` |

GGUF export failure falls back to HF format with a manual conversion hint.

---

## Adding a new node

1. Create `hmlcore/nodes/my_node.py`:

```python
from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

class MyNode(BaseNode):
    NAME = "MyNode"
    INPUT_KEYS  = ("model", "args")
    OUTPUT_KEYS = ("my_output",)

    def should_run(self, ctx):
        return getattr(ctx.get("args"), "enable_my_stage", False)

    def run(self, ctx):
        self._require(ctx, "model", "args")
        # ... do work ...
        ctx["my_output"] = result
```

2. Export from `hmlcore/nodes/__init__.py`.
3. Add to the node list in `hmlcore/graph_run.py`.
4. Add any new context keys to `NodeContext` in `context.py`.

`GraphRunner` will automatically place the new node at the correct position in the execution order based on its `INPUT_KEYS` / `OUTPUT_KEYS`.

---

## `hmlcore/config.py` — updated CLI args

| Arg | Default | Description |
|---|---|---|
| `--quantize` | `bf16` | Output format: `bf16`, `f16`, `q8_0`, `q4_k` |
| `--merge_quantization` | *(hidden)* | Legacy alias → resolved to `--quantize` in `apply_args()` |
| `--prune_experts` | `False` | Enable REAP expert pruning after training |
| `--prune_only` | `False` | Skip SFT + GRPO; run REAP only (implies `--prune_experts`) |
| `--prune_ratio` | `0.5` | Fraction of experts pruned per layer |
| `--calibration_samples` | `128` | Examples used for REAP calibration forward passes |
