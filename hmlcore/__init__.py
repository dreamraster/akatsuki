# By dreamraster · dreaMSCend
"""
hmlcore
========
Core package for Akatsuki — GRPO-based distillation pipeline with node-graph execution.

Subpackages
-----------
hmlcore.nodes   — pipeline nodes (InputNode, SFTNode, GRPONode, …)
hmlcore.shortcuts — Qwen3-style shortcut heads for auxiliary lookahead training

Quickstart::

    from hmlcore.config import build_parser, apply_args
    from hmlcore.nodes import GraphRunner, InputNode, SFTNode, GRPONode, OutputNode
"""

__version__ = "0.3.0"

# Shortcut-heads public API
from hmlcore.shortcut_heads import (
    ShortcutHead,
    ShortcutHeadConfig,
    ShortcutLossWrapper,
    ShortcutManager,
    build_shortcut_manager,
    unwrap_model,
    wrap_model,
)

__all__ = [
    "__version__",
    # Shortcut heads
    "ShortcutHeadConfig",
    "ShortcutHead",
    "ShortcutManager",
    "ShortcutLossWrapper",
    "build_shortcut_manager",
    "wrap_model",
    "unwrap_model",
]
