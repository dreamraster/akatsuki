# By dreamraster · dreaMSCend
"""
ohm_finetuner.py — GRPO-based Distillation for Specialized Domains

Thin entry-point wrapper around the hmlcore node-graph pipeline.
All logic lives in hmlcore/nodes/:

    InputNode   — load model / tokenizer / dataset + multimodal detection
    SFTNode     — SFT formatting warm-up
    GRPONode    — GRPO reinforcement learning
    PrunerNode  — REAP (MoE) or Bonsai/DLP (Dense) pruning stage
    OutputNode  — merge / quantize / save (HF or GGUF)
    PRISMDQNode — PRISM Dynamic Quantization recipe (post-save, optional)

Usage:
    python ohm_finetuner.py \\
        --student_model models/qwen-bnb-4 \\
        --datasets datasets/teichiai-claude-4.5-high-reasoning-250x.jsonl \\
        --domain code --max_steps 100 --lora_rank 16

    # Merge to bf16 checkpoint
    python ohm_finetuner.py ... --merge --quantize bf16

    # GGUF export (Unsloth)
    python ohm_finetuner.py ... --merge --quantize q4_k

    # REAP pruning only (MoE model)
    python ohm_finetuner.py ... --prune_only

    # Skip SFT
    python ohm_finetuner.py ... --disable_sft

    # Resume
    python ohm_finetuner.py ... --resume

    # PRISM Dynamic Quantization recipe (no training, just analysis + recipe)
    python ohm_finetuner.py ... --merge --quantize bf16 --prism_dq --target_bpw 3.5

    # PRISM-DQ with auto llama-quantize invocation
    python ohm_finetuner.py ... --merge --quantize bf16 --prism_dq --target_bpw 3.5 \\
        --dq_llama_path D:/llama.cpp/build/bin/llama-quantize.exe \\
        --dq_input_gguf ./build/finale/model_f16.gguf
"""

import logging
import os
import sys

# Force UTF-8 stdout/stderr so Unsloth's emoji banner doesn't crash on
# Windows cp1252 consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from hmlcore.config import build_parser, apply_args
    from hmlcore.nodes import (
        GraphRunner, make_context,
        InputNode, ShortcutHeadNode, SFTNode, ShortcutFreezeNode,
        GRPONode, PrunerNode, OutputNode, PRISMDQNode,
    )
    from hmlcore.nodes.base import NodeError

    parser = build_parser()
    args   = parser.parse_args()
    apply_args(args)

    nodes = [InputNode()]
    if getattr(args, "xtoken_enabled", False):
        from hmlcore.xtoken import create_xtoken_pipeline
        nodes.extend(create_xtoken_pipeline(
            teacher_model_path=args.teacher_model,
            projection_type=args.xtoken_projection,
            hidden_dim=args.xtoken_hidden_dim,
            num_epochs=args.xtoken_epochs,
        ))
    nodes.extend([
        ShortcutHeadNode(),  # attach shortcut heads (skipped if --shortcut_heads not set)
        SFTNode(),
        ShortcutFreezeNode(),  # freeze/unfreeze heads for GRPO (skipped if no manager)
        GRPONode(),
        PrunerNode(),
        OutputNode(),
        PRISMDQNode(),   # Runs only when --prism_dq is set; no-ops otherwise
    ])

    runner = GraphRunner(nodes)

    try:
        ctx = runner.run(make_context(args))
    except NodeError as exc:
        logger.error("�� Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("�� Done. Output: %s", ctx.get("finale_dir", args.output_dir))


if __name__ == "__main__":
    main()
