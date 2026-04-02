# By dreamraster · dreaMSCend
"""
hmlcore/nodes/sft_node.py
============================
SFTNode — Supervised Fine-Tuning warm-up stage.

Delegates to hmlcore.trainer.run_sft() which handles formatting warm-up on
a small subset (100 examples) of the primary dataset.

Skipped when:
  • args.disable_sft is True
  • args.prune_only is True
  • grpo_checkpoint is set (SFT weights already baked into RL checkpoint)

Consumes:  model, tokenizer, dataset, args, sft_dir, sft_checkpoint, grpo_checkpoint
Produces:  (nothing — model is modified in-place)
"""

from __future__ import annotations

import logging
import os

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)


class SFTNode(BaseNode):
    NAME = "SFTNode"
    INPUT_KEYS = ("model", "tokenizer", "dataset", "args",
                  "sft_dir", "sft_checkpoint", "grpo_checkpoint")
    OUTPUT_KEYS = ()

    def should_run(self, ctx: NodeContext) -> bool:
        args = ctx.get("args")
        if args is None:
            return False
        if getattr(args, "disable_sft", False):
            logger.info("⏭️  SFT skipped (--disable_sft).")
            return False
        if getattr(args, "prune_only", False):
            logger.info("⏭️  SFT skipped (--prune_only).")
            return False
        if ctx.get("grpo_checkpoint"):
            logger.info("⏭️  SFT skipped (weights already in GRPO checkpoint).")
            return False
        return True

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "model", "tokenizer", "dataset", "args", "sft_dir")
        args           = ctx["args"]
        model          = ctx["model"]
        tokenizer      = ctx["tokenizer"]
        dataset        = ctx["dataset"]
        sft_dir        = ctx["sft_dir"]
        sft_checkpoint = ctx.get("sft_checkpoint")

        # ── SFT training ──────────────────────────────────────────────────────
        from hmlcore.trainer import run_sft

        run_sft(model, tokenizer, dataset, args, sft_dir, sft_checkpoint)
