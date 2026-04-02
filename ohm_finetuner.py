# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
ohm_finetuner.py — GRPO-based Distillation for Specialized Domains

Thin entry-point wrapper around the hmlcore node-graph pipeline.
All logic lives in hmlcore/nodes/:

    InputNode   — load model / tokenizer / dataset + multimodal detection
    SFTNode     — SFT formatting warm-up
    GRPONode    — GRPO reinforcement learning
    PrunerNode  — REAP MoE expert pruning (MoE models only)
    OutputNode  — merge / quantize / save (HF or GGUF)

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
        InputNode, SFTNode, GRPONode, PrunerNode, OutputNode,
    )
    from hmlcore.nodes.base import NodeError

    parser = build_parser()
    args   = parser.parse_args()
    apply_args(args)

    runner = GraphRunner([
        InputNode(),
        SFTNode(),
        GRPONode(),
        PrunerNode(),
        OutputNode(),
    ])

    try:
        ctx = runner.run(make_context(args))
    except NodeError as exc:
        logger.error("💥 Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("🎉 Done. Output: %s", ctx.get("finale_dir", args.output_dir))


if __name__ == "__main__":
    main()
