# By dreamraster · dreaMSCend
"""
hmlcore.nodes
===============
Node-graph pipeline primitives.

Public exports
--------------
BaseNode, NodeError          — abstract base and error class
NodeContext, make_context    — shared state dict + factory
GraphRunner                  — topological executor
InputNode                    — model/tokenizer/dataset loader
SFTNode                      — SFT warm-up stage
GRPONode                     — GRPO RL stage
PrunerNode                   — REAP MoE expert pruning stage
OutputNode                   — final save / merge / export

Quickstart::

    from hmlcore.nodes import (
        GraphRunner, make_context,
        InputNode, SFTNode, GRPONode, PrunerNode, OutputNode,
    )

    runner = GraphRunner([
        InputNode(), SFTNode(), GRPONode(), PrunerNode(), OutputNode(),
    ])
    ctx = runner.run(make_context(args))
    print("Saved to:", ctx["finale_dir"])
"""

from hmlcore.nodes.base    import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext, make_context
from hmlcore.nodes.runner  import GraphRunner
from hmlcore.nodes.input_node   import InputNode
from hmlcore.nodes.sft_node     import SFTNode
from hmlcore.nodes.grpo_node    import GRPONode
from hmlcore.nodes.pruner_node  import PrunerNode
from hmlcore.nodes.output_node  import OutputNode
from hmlcore.nodes.model_info   import log_stage_model_info
from hmlcore.dense_pruner       import drop_dense_layers, find_decoder_layers

__all__ = [
    "BaseNode", "NodeError",
    "NodeContext", "make_context",
    "GraphRunner",
    "InputNode",
    "SFTNode",
    "GRPONode",
    "PrunerNode",
    "OutputNode",
    "log_stage_model_info",
    "drop_dense_layers",
    "find_decoder_layers",
]
