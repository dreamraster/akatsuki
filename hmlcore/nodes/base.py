# By dreamraster · dreaMSCend
"""
hmlcore/nodes/base.py
=======================
Abstract base class for all pipeline nodes.

Every node declares:
  NAME        — human-readable label used in logs and runner output
  INPUT_KEYS  — context keys the node reads (used for dependency ordering)
  OUTPUT_KEYS — context keys the node writes (used for dependency ordering)

Nodes communicate exclusively through the shared NodeContext dict.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)


class NodeError(RuntimeError):
    """Raised when a pipeline node fails in a non-recoverable way."""


class BaseNode(ABC):
    """Abstract pipeline node.

    Subclasses must:
      1. Set NAME, INPUT_KEYS, OUTPUT_KEYS as class attributes.
      2. Implement run(ctx) to read from / write to ctx.
      3. Optionally override should_run(ctx) to skip themselves.
    """

    NAME: str = "BaseNode"
    INPUT_KEYS: tuple[str, ...] = ()
    OUTPUT_KEYS: tuple[str, ...] = ()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def should_run(self, ctx: "NodeContext") -> bool:
        """Return False to skip this node entirely.

        The default implementation returns True.  Override in subclasses for
        conditional stages (e.g. SFTNode skips when --disable_sft is set).
        """
        return True

    @abstractmethod
    def run(self, ctx: "NodeContext") -> None:
        """Execute the node's logic, reading inputs from *ctx* and writing
        outputs back into *ctx*.

        Must raise NodeError on unrecoverable failure so GraphRunner can
        surface a clean message rather than a raw traceback.
        """

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require(self, ctx: "NodeContext", *keys: str) -> None:
        """Assert that all *keys* are present in *ctx*.

        Called at the top of run() to give a clear error if a predecessor
        node failed silently (produced None) or was accidentally skipped.
        """
        missing = [k for k in keys if k not in ctx]
        if missing:
            raise NodeError(
                f"{self.NAME}: required context keys missing: {missing}. "
                "Check that predecessor nodes ran successfully."
            )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"in={self.INPUT_KEYS} out={self.OUTPUT_KEYS}>"
        )
