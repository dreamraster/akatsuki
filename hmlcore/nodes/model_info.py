# By dreamraster · dreaMSCend
"""
hmlcore/nodes/model_info.py
============================
Shared utility for logging a model snapshot after each pipeline stage.

Called by GraphRunner after every node that may change the model.  Each
snapshot logs:
  - Model class + quantization state (dtype)
  - Parameter counts (total, trainable) with LoRA indicator
  - Architecture (layers, hidden size, attention heads)
  - MoE topology (if present: layer count, experts/layer, top_k)
  - Memory estimate in GB
  - Dataset size (at InputNode time)

The snapshot is intentionally compact (fits in a terminal window) and distinct
from OutputNode's detailed _log_model_stats which also measures disk size.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_SEP_W = 56


def _dominant_dtype(model) -> str:
    counts: dict[str, int] = {}
    for p in model.parameters():
        k = str(p.dtype).replace("torch.", "")
        counts[k] = counts.get(k, 0) + p.numel()
    return max(counts, key=counts.get) if counts else "unknown"


def _quant_label(dominant: str) -> str:
    """Human-readable quantization description from dominant dtype string."""
    _MAP = {
        "uint8":    "4-bit BnB (uint8 packed)",
        "int8":     "8-bit BnB (int8)",
        "float8_e4m3fn": "FP8",
        "bfloat16": "bfloat16",
        "float16":  "float16",
        "float32":  "float32",
    }
    return _MAP.get(dominant, dominant)


def _count_params(model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _mem_gb(model, dominant: str) -> float:
    _BYTES: dict = {
        "float32": 4, "float16": 2, "bfloat16": 2,
        "int8": 1, "uint8": 1,
    }
    bpp = _BYTES.get(dominant, 2)
    return sum(p.numel() for p in model.parameters()) * bpp / 1e9


def _arch_info(model) -> dict:
    """Extract common architecture fields from model.config."""
    cfg  = getattr(model, "config", None)
    info: dict = {}
    if cfg is None:
        return info
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            info["layers"] = getattr(cfg, attr)
            break
    for attr in ("hidden_size", "d_model", "n_embd"):
        if hasattr(cfg, attr):
            info["hidden"] = getattr(cfg, attr)
            break
    for attr in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(cfg, attr):
            info["heads"] = getattr(cfg, attr)
            break
    for attr in ("intermediate_size", "ffn_dim", "d_ff"):
        if hasattr(cfg, attr):
            info["intermediate"] = getattr(cfg, attr)
            break
    for attr in ("vocab_size",):
        if hasattr(cfg, attr):
            info["vocab"] = getattr(cfg, attr)
            break
    for attr in ("max_position_embeddings", "max_seq_len"):
        if hasattr(cfg, attr):
            info["max_pos"] = getattr(cfg, attr)
            break
    return info


def _moe_info(model) -> str | None:
    """Return a one-line MoE summary or None if not a MoE model."""
    try:
        from hmlcore.moe import find_moe_layers, get_num_experts
        moe_layers = find_moe_layers(model)
        if not moe_layers:
            return None
        ne    = get_num_experts(moe_layers)
        cfg   = getattr(model, "config", None)
        top_k = "?"
        if cfg:
            for attr in ("num_experts_per_tok", "top_k"):
                if hasattr(cfg, attr):
                    top_k = getattr(cfg, attr)
                    break
        return f"{len(moe_layers)} MoE layer(s)  |  {ne} experts/layer  |  top_k={top_k}"
    except Exception:
        return None


def log_stage_model_info(
    stage: str,
    model,
    tokenizer=None,
    dataset=None,
) -> None:
    """
    Log a compact model snapshot labelled with the pipeline stage name.

    Args:
        stage:     Name of the just-completed node (e.g. "InputNode").
        model:     The current model object.
        tokenizer: Optional — used for vocab size cross-check.
        dataset:   Optional — logged as example count if provided.
    """
    try:
        sep     = "━" * _SEP_W
        total, trainable = _count_params(model)
        dominant = _dominant_dtype(model)
        quant    = _quant_label(dominant)
        mem_gb   = _mem_gb(model, dominant)
        arch     = _arch_info(model)
        moe      = _moe_info(model)
        cls_name = type(model).__name__

        lora_note = ""
        if trainable > 0:
            lora_note = f"  ← LoRA adapter ({trainable / 1e6:.1f}M params)"

        logger.info(sep)
        logger.info("  Stage snapshot: %s", stage)
        logger.info(sep)
        logger.info("  Model:       %s", cls_name)
        logger.info("  Precision:   %s", quant)
        logger.info(
            "  Parameters:  %s total  |  %s trainable%s",
            f"{total:,}", f"{trainable:,}", lora_note,
        )
        logger.info("  Memory est:  %.2f GB", mem_gb)

        if arch:
            arch_parts = []
            if "layers" in arch:
                arch_parts.append(f"{arch['layers']} layers")
            if "hidden" in arch:
                arch_parts.append(f"hidden={arch['hidden']}")
            if "heads" in arch:
                arch_parts.append(f"heads={arch['heads']}")
            if "intermediate" in arch:
                arch_parts.append(f"ffn={arch['intermediate']}")
            if arch_parts:
                logger.info("  Architecture: %s", "  |  ".join(arch_parts))

            vocab_parts = []
            if "vocab" in arch:
                vocab_parts.append(f"vocab={arch['vocab']:,}")
            if "max_pos" in arch:
                vocab_parts.append(f"ctx={arch['max_pos']:,}")
            if tokenizer and hasattr(tokenizer, "vocab_size"):
                tok_v = tokenizer.vocab_size
                if tok_v and ("vocab" not in arch or tok_v != arch["vocab"]):
                    vocab_parts.append(f"tok_vocab={tok_v:,}")
            if vocab_parts:
                logger.info("  Tokenizer:   %s", "  |  ".join(vocab_parts))

        if moe:
            logger.info("  MoE:         %s", moe)

        if dataset is not None:
            logger.info("  Dataset:     %d examples", len(dataset))

        logger.info(sep)

    except Exception as exc:
        logger.debug("log_stage_model_info failed for %s: %s", stage, exc)
