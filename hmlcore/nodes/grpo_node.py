# By dreamraster · dreaMSCend
"""
hmlcore/nodes/grpo_node.py
============================
GRPONode — Group Relative Policy Optimisation training stage.

Skipped when:
  • args.prune_only is True
  • is_multimodal is True  (VLM compute_3d_position_ids crashes on text-only inputs)
  • dataset has fewer examples than args.num_generations

Reward functions are built inside this node (via hmlcore.rewards) and the
LMStudioJudge (if active) is closed in a finally block to flush the cache.

Consumes:  model, tokenizer, dataset, args, grpo_dir, grpo_checkpoint, is_multimodal
Produces:  (nothing — model is modified in-place)
"""

from __future__ import annotations

import logging

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)


class GRPONode(BaseNode):
    NAME = "GRPONode"
    INPUT_KEYS = ("model", "tokenizer", "dataset", "args",
                  "grpo_dir", "grpo_checkpoint", "is_multimodal")
    OUTPUT_KEYS = ()

    def should_run(self, ctx: NodeContext) -> bool:
        args = ctx.get("args")
        if args is None:
            return False
        if getattr(args, "prune_only", False):
            logger.info("⏭️  GRPO skipped (--prune_only).")
            return False
        # VLM models are now supported in GRPO if the environment handles vision tokens.
        # We only log a warning to ensure the user knows they need specialized rewards.
        if ctx.get("is_multimodal", False):
            logger.info("🎨 VLM detected — running multimodal GRPO.")
        return True

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "model", "tokenizer", "dataset", "args", "grpo_dir")
        args            = ctx["args"]
        model           = ctx["model"]
        tokenizer       = ctx["tokenizer"]
        dataset         = ctx["dataset"]
        grpo_dir        = ctx["grpo_dir"]
        grpo_checkpoint = ctx.get("grpo_checkpoint")

        # Guard: need enough examples for GRPO rollouts
        num_gen = getattr(args, "num_generations", 4)
        if len(dataset) < num_gen:
            raise NodeError(
                f"GRPO requires at least {num_gen} examples (num_generations={num_gen}) "
                f"but dataset has only {len(dataset)}. "
                "Reduce --num_generations or use a larger dataset."
            )

        # ── Unsloth compatibility: ensure warnings_issued exists on base model ─
        # Unsloth sets this during SFT; when SFT is skipped it may be absent.
        try:
            base = getattr(getattr(model, "base_model", None), "model", None) \
                   or getattr(model, "model", model)
            if not hasattr(base, "warnings_issued"):
                base.warnings_issued = {}
        except Exception:
            pass

        # ── Reward functions ──────────────────────────────────────────────────
        from hmlcore.rewards import build_reward_functions
        reward_funcs, judge = build_reward_functions(args, tokenizer)

        # ── GRPO training ─────────────────────────────────────────────────────
        from hmlcore.trainer import run_grpo

        logger.info("🎯 Starting Step 2: GRPO RL (%s domain) ...", args.domain)
        try:
            run_grpo(model, tokenizer, dataset, reward_funcs, args,
                     grpo_dir, grpo_checkpoint)
        finally:
            if judge is not None:
                logger.info(judge.cache_stats())
                judge.close()

        logger.info("✅ GRPO complete.")
