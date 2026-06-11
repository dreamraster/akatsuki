# By dreamraster · dreaMSCend
"""
hmlcore/nodes/shortcut_freeze_node.py
========================================
ShortcutFreezeNode — freezes or unfreezes shortcut heads between training
stages (SFT → GRPO).

By default, shortcut heads are frozen after SFT so they act as a stabilizing
prior during GRPO without consuming gradient budget.  Pass
--no_shortcut_freeze to keep them trainable through GRPO.

Pipeline position: between SFTNode and GRPONode.

Consumes:  shortcut_manager, args
Produces:  (nothing — modifies model parameters in-place)
"""

from __future__ import annotations

import logging

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)


class ShortcutFreezeNode(BaseNode):
    """Freeze/unfreeze shortcut heads between SFT and GRPO stages.

    Skipped when:
      • shortcut_manager is not present (shortcut heads disabled)
      • --prune_only is set (no training happens)
      • --disable_sft is set (no SFT → no freeze boundary)
    """

    NAME = "ShortcutFreezeNode"
    INPUT_KEYS = ("shortcut_manager", "args")
    OUTPUT_KEYS = ()

    def should_run(self, ctx: NodeContext) -> bool:
        args = ctx.get("args")
        if args is None:
            return False

        if not ctx.get("shortcut_manager"):
            return False

        # Nothing to freeze if SFT was skipped or we're prune-only
        if getattr(args, "prune_only", False):
            logger.info("⏭️  Shortcut freeze skipped (--prune_only).")
            return False
        if getattr(args, "disable_sft", False):
            logger.info("⏭️  Shortcut freeze skipped (--disable_sft, no SFT boundary).")
            return False
        return True

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "shortcut_manager", "args")

        manager = ctx["shortcut_manager"]
        args = ctx["args"]

        freeze_after_sft = not getattr(args, "no_shortcut_freeze", False)

        if freeze_after_sft:
            manager.freeze()
            logger.info(
                "�� Shortcut heads frozen after SFT (%d heads, %s params locked).",
                len(manager.shortcut_heads),
                f"{manager.total_params():,}",
            )
        else:
            manager.unfreeze()
            logger.info(
                "�� Shortcut heads kept trainable through GRPO (%d heads, %s params).",
                len(manager.shortcut_heads),
                f"{manager.trainable_params():,}",
            )
