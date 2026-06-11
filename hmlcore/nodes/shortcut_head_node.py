# By dreamraster · dreaMSCend
"""
hmlcore/nodes/shortcut_head_node.py
======================================
ShortcutHeadNode — builds, attaches, and wraps the model with Qwen3-style
shortcut heads before any training begins.

Shortcut heads are shallow transformer blocks attached to intermediate decoder
layers.  Each head receives the hidden state at its layer and predicts a token
further ahead (t+offset).  During training the auxiliary loss is:

    L_total = L_main + λ · Σ L_shortcut_k

The ShortcutLossWrapper auto-injects this into output.loss so no trainer
changes are required — it just works with SFTTrainer and GRPOTrainer.

Pipeline position: runs immediately after InputNode (model + tokenizer loaded).

Consumes:  model, tokenizer, args
Produces:  shortcut_manager  (ShortcutManager | None)

Lifecycle:
  1. Build shortcut heads from CLI config (--shortcut_heads, --shortcut_layers, …)
  2. Attach forward hooks on the specified decoder layers
  3. Wrap the model with ShortcutLossWrapper
  4. Store the manager in ctx so later nodes can freeze/unfreeze
"""

from __future__ import annotations

import logging

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext
from hmlcore.shortcut_heads import (
    ShortcutHeadConfig,
    ShortcutManager,
    build_shortcut_manager,
    wrap_model,
)

logger = logging.getLogger(__name__)


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string of integers, e.g. '-3,-2'."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _build_shortcut_config(args) -> ShortcutHeadConfig:
    """Build ShortcutHeadConfig from parsed CLI args."""
    enabled = getattr(args, "shortcut_heads", False)

    # Only parse layer/offset strings when enabled to avoid noise from defaults
    layer_indices = _parse_int_list(getattr(args, "shortcut_layers", "-3,-2"))
    offsets = _parse_int_list(getattr(args, "shortcut_offsets", "2,3"))

    freeze_after_sft = not getattr(args, "no_shortcut_freeze", False)

    return ShortcutHeadConfig(
        enabled=enabled,
        layer_indices=layer_indices,
        offsets=offsets,
        num_layers=getattr(args, "shortcut_depth", 2),
        hidden_dim=None,
        num_heads=4,
        loss_weight=getattr(args, "shortcut_weight", 0.1),
        freeze_after_sft=freeze_after_sft,
        dropout=0.1,
    )


class ShortcutHeadNode(BaseNode):
    """Attach shortcut heads to the model before training starts.

    Skipped when --shortcut_heads is not set.
    """

    NAME = "ShortcutHeadNode"
    INPUT_KEYS = ("model", "tokenizer", "args")
    OUTPUT_KEYS = ("shortcut_manager",)

    def should_run(self, ctx: NodeContext) -> bool:
        args = ctx.get("args")
        if args is None:
            return False
        if not getattr(args, "shortcut_heads", False):
            logger.info("⏭️  Shortcut heads disabled (pass --shortcut_heads to enable).")
            return False
        if ctx.get("shortcut_manager") is not None:
            logger.info("⏭️  Shortcut heads already attached — skipping.")
            return False
        return True

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "model", "tokenizer", "args")

        model = ctx["model"]
        args = ctx["args"]

        # ── Build config from CLI args ──────────────────────────────────────
        sc_config = _build_shortcut_config(args)

        logger.info("�� Setting up Shortcut Heads (Qwen3-style) ...")

        # ── Build and attach shortcut manager ────────────────────────────────
        manager = build_shortcut_manager(model, sc_config)
        if manager is None:
            raise NodeError(
                "ShortcutHeadNode: build_shortcut_manager returned None. "
                "Either --shortcut_heads is not set, or no compatible decoder "
                "was found in the model."
            )

        # Attach hooks on the decoder layers so hidden states are captured
        manager.attach()

        # ── Wrap the model ───────────────────────────────────────────────────
        wrapped = wrap_model(model, manager)
        ctx["model"] = wrapped

        # ── Store manager for later lifecycle stages ─────────────────────────
        # Store on both ctx (for node-to-node communication) and args
        # (so trainer.py run_sft/run_grpo can access it).
        ctx["shortcut_manager"] = manager
        args.shortcut_manager = manager

        # ── Summary ──────────────────────────────────────────────────────────
        tp = manager.trainable_params()
        total = manager.total_params()
        logger.info(
            "✅ Shortcut heads attached: %d heads, %s total params (%s trainable)",
            len(manager.shortcut_heads),
            f"{total:,}",
            f"{tp:,}",
        )
        breakdown = []
        for hi, li in enumerate(manager.layer_to_head):
            h = manager.shortcut_heads[hi]
            breakdown.append(
                f"  head#{hi}: layer={li}, offset=+{h.offset}, {h.num_params():,} params"
            )
        logger.info("Shortcut head layout:\n%s", "\n".join(breakdown))
