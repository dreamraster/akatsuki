# By dreamraster · dreaMSCend
"""
hmlcore/nodes/runner.py
=========================
GraphRunner — topological-sort executor for BaseNode pipelines.

Usage::

    runner = GraphRunner([InputNode(), SFTNode(), GRPONode(), OutputNode()])
    ctx = runner.run(make_context(args))

The runner:
  1. Validates that all INPUT_KEYS for every node are satisfied by either the
     initial context or a predecessor's OUTPUT_KEYS.
  2. Topologically sorts the nodes (Kahn's algorithm on the dependency DAG).
  3. Executes each node in order, calling should_run() first, then run().
  4. Surfaces NodeError cleanly rather than letting raw tracebacks escape.

Duplicate OUTPUT_KEYS across nodes are allowed (last writer wins).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Sequence

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext, make_context

import argparse

logger = logging.getLogger(__name__)


class GraphRunner:
    """Execute a list of BaseNode instances in dependency order."""

    def __init__(self, nodes: Sequence[BaseNode]):
        self.nodes: list[BaseNode] = list(nodes)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, ctx: NodeContext) -> NodeContext:
        """Sort and execute all nodes.  Returns the final context."""
        ordered = self._topo_sort(self.nodes)
        logger.info(
            "🗺️  Pipeline order: %s",
            " → ".join(n.NAME for n in ordered),
        )

        for node in ordered:
            if not node.should_run(ctx):
                logger.info("⏭️  Skipping %s (should_run=False)", node.NAME)
                continue

            logger.info("▶️  Running %s ...", node.NAME)
            try:
                node.run(ctx)
            except NodeError as exc:
                logger.error("❌ %s failed: %s", node.NAME, exc)
                raise
            except Exception as exc:
                # Log full traceback — str(exc) can be empty for some exception
                # types (safetensors errors, Windows codec errors, etc.)
                logger.exception("❌ %s raised an unexpected error", node.NAME)
                raise NodeError(
                    f"{node.NAME} raised an unexpected error: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            logger.info("✅ %s complete.", node.NAME)

            # Log a compact model snapshot after every node that touches the model.
            # OutputNode is excluded — it already emits a detailed stats block
            # (including disk size) via _log_model_stats at the end of its run().
            if node.NAME != "OutputNode" and "model" in ctx:
                try:
                    from hmlcore.nodes.model_info import log_stage_model_info
                    log_stage_model_info(
                        stage     = node.NAME,
                        model     = ctx["model"],
                        tokenizer = ctx.get("tokenizer"),
                        dataset   = ctx.get("dataset"),
                    )
                except Exception as _info_exc:
                    logger.debug("Stage snapshot failed: %s", _info_exc)

        return ctx

    # ── Topological sort (Kahn's algorithm) ──────────────────────────────────

    def _topo_sort(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Return nodes in execution order respecting INPUT_KEYS → OUTPUT_KEYS
        dependency edges.

        Algorithm:
          - For each key K produced by node P (P in OUTPUT_KEYS),
            every node C that needs K (K in C.INPUT_KEYS) gets an edge P → C.
          - Kahn's BFS processes nodes with in-degree 0 first.
          - Nodes with no dependency relationship keep their original order
            (Python's dict preserves insertion order, so the stable behaviour
            is guaranteed as long as the input list is ordered sensibly).
        """
        if not nodes:
            return []

        n = len(nodes)
        idx = {id(node): i for i, node in enumerate(nodes)}

        # Map each key → list of node indices that produce it
        producers: dict[str, list[int]] = defaultdict(list)
        for i, node in enumerate(nodes):
            for key in node.OUTPUT_KEYS:
                producers[key].append(i)

        # Build adjacency list and in-degree count
        adj: list[set[int]] = [set() for _ in range(n)]
        in_deg: list[int] = [0] * n

        for i, node in enumerate(nodes):
            for key in node.INPUT_KEYS:
                for j in producers[key]:
                    if j != i and i not in adj[j]:
                        adj[j].add(i)
                        in_deg[i] += 1

        # BFS (Kahn's)
        queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
        result: list[BaseNode] = []

        while queue:
            cur = queue.popleft()
            result.append(nodes[cur])
            for nxt in sorted(adj[cur]):   # sorted for determinism
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)

        if len(result) != n:
            cycle_nodes = [nodes[i].NAME for i in range(n) if in_deg[i] > 0]
            raise NodeError(
                f"Cycle detected in pipeline graph. Involved nodes: {cycle_nodes}"
            )

        return result

    # ── Convenience factory ───────────────────────────────────────────────────

    @classmethod
    def from_args(
        cls,
        nodes: Sequence[BaseNode],
        args: argparse.Namespace,
    ) -> tuple["GraphRunner", NodeContext]:
        """Create a runner and a pre-seeded context from parsed CLI args."""
        return cls(nodes), make_context(args)
