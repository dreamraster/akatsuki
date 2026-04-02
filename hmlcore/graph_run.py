# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
hmlcore/graph_run.py  —  Node-graph entry point for the full pipeline.

Usage:
    python -m hmlcore.graph_run --student_model <path> --datasets <path> --domain code

    # SFT + GRPO + merge to bf16 HF:
    python -m hmlcore.graph_run --student_model <model> --datasets <ds> \\
        --domain code --merge --quantize bf16

    # REAP pruning + GGUF export (MoE model required):
    python -m hmlcore.graph_run --student_model <moe_model> --datasets <ds> \\
        --domain code --prune_experts --prune_ratio 0.4 --merge --quantize q4_k

    # Prune only (skip SFT + GRPO):
    python -m hmlcore.graph_run --student_model <moe_model> --datasets <ds> \\
        --domain code --prune_only

    # Skip SFT, run GRPO only:
    python -m hmlcore.graph_run ... --disable_sft

The pipeline is a directed acyclic graph of BaseNode instances executed in
topological order.  Each node reads from and writes to a shared NodeContext
dict; no node modifies its predecessor's outputs directly.

Pipeline nodes (always registered, each may self-skip via should_run()):
    InputNode   → SFTNode → GRPONode → PrunerNode → OutputNode
"""

import logging
import sys

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

    # ── Parse CLI ─────────────────────────────────────────────────────────────
    parser = build_parser()
    args   = parser.parse_args()
    apply_args(args)

    # ── Build node graph ──────────────────────────────────────────────────────
    nodes = [
        InputNode(),
        SFTNode(),
        GRPONode(),
        PrunerNode(),
        OutputNode(),
    ]

    runner = GraphRunner(nodes)
    ctx    = make_context(args)

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        ctx = runner.run(ctx)
    except NodeError as exc:
        logger.error("💥 Pipeline failed: %s", exc)
        sys.exit(1)

    finale = ctx.get("finale_dir", args.output_dir)
    logger.info("🎉 Pipeline complete. Output: %s", finale)


if __name__ == "__main__":
    main()
