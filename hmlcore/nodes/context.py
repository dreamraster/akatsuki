# By dreamraster · dreaMSCend
"""
hmlcore/nodes/context.py
==========================
Shared pipeline context passed between all nodes.

NodeContext is a plain dict (TypedDict for type-checker hints).  Every node
reads inputs from it and writes outputs back into it.  The keys below are
the canonical names used across all nodes — add new keys here when
introducing new nodes.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

try:
    from typing import TypedDict
except ImportError:                         # Python < 3.8 fallback
    from typing_extensions import TypedDict  # type: ignore


class NodeContext(TypedDict, total=False):
    # ── CLI args (always present after GraphRunner.run()) ────────────────────
    args: argparse.Namespace

    # ── Model + tokenizer ────────────────────────────────────────────────────
    model: Any
    tokenizer: Any
    use_unsloth: bool
    is_multimodal: bool

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: Any          # HuggingFace Dataset

    # ── Stage directories ────────────────────────────────────────────────────
    sft_dir: str
    grpo_dir: str
    finale_dir: str

    # ── Resume checkpoints (None = start fresh) ──────────────────────────────
    sft_checkpoint: Optional[str]
    grpo_checkpoint: Optional[str]

    # ── Reward functions (built inside GRPONode) ─────────────────────────────
    reward_funcs: list
    judge: Optional[Any]  # LMStudioJudge or None


def make_context(args: argparse.Namespace) -> NodeContext:
    """Create a minimal context seeded with the parsed CLI args."""
    return NodeContext(args=args)  # type: ignore[return-value]
