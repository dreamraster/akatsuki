"""
X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation

This module implements X-Token, a novel knowledge distillation technique that
guides student model learning through projection-based alignment between
teacher and student embedding spaces, using cross-tokenizer matching.
"""

from __future__ import annotations

from hmlcore.xtoken.distiller import XTokenDistiller
from hmlcore.xtoken.node import XTokenNode, XTokenDistillNode, create_xtoken_pipeline
from hmlcore.xtoken.projection import ProjectionAligner

__all__ = ["XTokenDistiller", "XTokenNode", "XTokenDistillNode", "ProjectionAligner", "create_xtoken_pipeline"]
