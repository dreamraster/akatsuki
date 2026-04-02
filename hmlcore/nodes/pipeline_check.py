# By dreamraster · dreaMSCend
"""
hmlcore/nodes/pipeline_check.py
=================================
Pre-flight pipeline compatibility report.

Called once after model load (inside InputNode) before any training begins.
Analyses the loaded model and prints a table that shows, for every pipeline
stage, whether it will run and WHY — so the user can catch incompatibilities
before wasting hours of training time.

Public API
----------
run_pipeline_check(model, tokenizer, args, is_multimodal) -> None
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Visual markers
_YES  = "✅"
_WARN = "⚠️ "
_NO   = "❌"
_SEP  = "━" * 62


def run_pipeline_check(model, tokenizer, args, is_multimodal: bool) -> None:
    """Analyse the model and log a pipeline compatibility table."""

    try:
        _do_check(model, tokenizer, args, is_multimodal)
    except Exception as exc:
        # Never block the pipeline for a diagnostics failure
        logger.debug("Pipeline check failed (non-fatal): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Internal analysis
# ─────────────────────────────────────────────────────────────────────────────

def _do_check(model, tokenizer, args, is_multimodal: bool) -> None:
    cls_name = type(model).__name__

    # ── Quantization ─────────────────────────────────────────────────────────
    quant_label = _detect_quant(model)

    # ── Topology ─────────────────────────────────────────────────────────────
    is_moe, num_moe_layers        = _check_moe(model)
    is_mamba, mamba_attr          = _check_mamba(model)
    has_dense_layers, num_dense   = _check_dense(model)
    is_vlm                        = _check_vlm(model, tokenizer, is_multimodal)

    # ── Resolve topology label ────────────────────────────────────────────────
    topo_parts = []
    if is_vlm:
        topo_parts.append("VLM (vision-language)")
    if is_moe:
        topo_parts.append(f"MoE ({num_moe_layers} expert layer(s))")
    if is_mamba:
        topo_parts.append(f"Mamba/SSM hybrid [{mamba_attr}]")
    if has_dense_layers and not is_moe:
        topo_parts.append(f"dense transformer ({num_dense} blocks)")
    topo_label = "  +  ".join(topo_parts) if topo_parts else "Unknown / custom"

    # ── Multimodal processor ─────────────────────────────────────────────────
    has_inner_tok = hasattr(tokenizer, "tokenizer")
    tok_label = (
        f"Multimodal processor  (inner tokenizer: {type(getattr(tokenizer, 'tokenizer')).__name__})"
        if has_inner_tok
        else type(tokenizer).__name__
    )

    # ── Param count ───────────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Library versions ──────────────────────────────────────────────────────
    lib_versions = _get_lib_versions()

    # ── Stage decisions ───────────────────────────────────────────────────────
    want_sft   = not getattr(args, "disable_sft",  False)
    want_grpo  = not getattr(args, "disable_grpo", False)
    want_prune = (
        getattr(args, "prune_experts", False)
        or getattr(args, "prune_only",   False)
        or getattr(args, "prune_ratio",  None) is not None
    )

    stages = _compute_stage_plan(
        want_sft      = want_sft,
        want_grpo     = want_grpo,
        want_prune    = want_prune,
        is_moe        = is_moe,
        is_mamba      = is_mamba,
        has_dense     = has_dense_layers,
        is_multimodal = is_multimodal,
        prune_ratio   = getattr(args, "prune_ratio", None),
        dynamicquant  = getattr(args, "dynamicquant", False),
    )

    # ── Print report ──────────────────────────────────────────────────────────
    logger.info(_SEP)
    logger.info("  Pipeline Compatibility Report")
    logger.info(_SEP)
    logger.info("  Model:       %s", cls_name)
    logger.info("  Topology:    %s", topo_label)
    logger.info("  Tokenizer:   %s", tok_label)
    logger.info("  Params:      %s total  |  %s trainable (LoRA/PEFT)",
                f"{total_params:,}", f"{trainable_params:,}")
    logger.info("  Precision:   %s", quant_label)
    logger.info("  Libraries:   %s", lib_versions)
    logger.info(_SEP)

    for stage_name, icon, reason in stages:
        if reason:
            logger.info("  %-10s %s  %s", stage_name, icon, reason)
        else:
            logger.info("  %-10s %s", stage_name, icon)

    logger.info(_SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Stage planning
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stage_plan(
    want_sft, want_grpo, want_prune,
    is_moe, is_mamba, has_dense, is_multimodal,
    prune_ratio, dynamicquant=False,
) -> list[tuple[str, str, str]]:
    """Return list of (stage_name, icon, reason) tuples."""
    plan = []

    # SFT
    if want_sft:
        plan.append(("SFT", _YES, "will run"))
    else:
        plan.append(("SFT", _NO, "skipped  (--disable_sft)"))

    # GRPO
    if not want_grpo:
        plan.append(("GRPO", _NO, "skipped  (--disable_grpo)"))
    elif is_multimodal:
        plan.append(("GRPO", _YES, "will run (multimodal rollout)"))
    else:
        plan.append(("GRPO", _YES, "will run"))

    # Pruning
    _dq = "  [dynamicquant: 1-bit degrade instead of remove]" if dynamicquant else ""
    if not want_prune:
        plan.append(("Pruning", _NO,
                     "skipped  — pass --prune_ratio <0.0–1.0> to enable"))
    elif is_moe:
        ratio_str = f"  ratio={prune_ratio}" if prune_ratio is not None else ""
        plan.append(("Pruning", _YES,
                     f"REAP expert pruning (MoE){ratio_str}{_dq}"))
    elif is_mamba:
        plan.append(("Pruning", _NO,
                     "skipped  — Mamba/SSM hybrid: layer renumbering would break "
                     "GGUF block-type mapping in llama.cpp"))
    elif has_dense:
        ratio_str = f"  ratio={prune_ratio}" if prune_ratio is not None else ""
        plan.append(("Pruning", _YES,
                     f"ShortGPT layer dropping (dense transformer){ratio_str}{_dq}"))
    else:
        plan.append(("Pruning", _WARN,
                     "UNKNOWN topology — will attempt at runtime, may skip"))

    # Output is always present
    plan.append(("Output", _YES, "will run"))

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detectors
# ─────────────────────────────────────────────────────────────────────────────

def _check_moe(model) -> tuple[bool, int]:
    try:
        from hmlcore.moe import find_moe_layers
        layers = find_moe_layers(model)
        return len(layers) > 0, len(layers)
    except Exception:
        return False, 0


_SSM_ATTRS = frozenset({
    "ssm_conv1d", "dt_proj", "A_log", "x_proj",
    "dt_layernorm", "q_layernorm", "mixer", "conv1d",
})

def _check_mamba(model) -> tuple[bool, str]:
    """Return (is_hybrid, first_ssm_attr_found)."""
    # Check the actual decoder layers (not the whole model, to avoid false positives
    # from e.g. a conv layer in a vision encoder)
    try:
        from hmlcore.dense_pruner import find_decoder_layers
        layers, _ = _raw_find_decoder_layers(model)
        if layers is None:
            return False, ""
        for layer in layers:
            for name, _ in layer.named_modules():
                leaf = name.split(".")[-1]
                if leaf in _SSM_ATTRS:
                    return True, leaf
    except Exception:
        pass
    return False, ""


def _check_dense(model) -> tuple[bool, int]:
    """Return (found, num_layers) WITHOUT the Mamba guard (used for reporting)."""
    try:
        layers, _ = _raw_find_decoder_layers(model)
        if layers is not None:
            return True, len(layers)
    except Exception:
        pass
    return False, 0


def _raw_find_decoder_layers(model):
    """find_decoder_layers without the Mamba guard, for analysis only."""
    import torch
    _LAYER_PATHS = [
        "model.layers", "model.model.layers", "transformer.h",
        "model.transformer.h", "gpt_neox.layers", "model.gpt_neox.layers",
        "layers", "model.blocks", "decoder.layers",
    ]
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj, path
    return None, None


_VLM_CLASS_FRAGMENTS = frozenset({
    "VisionLanguage", "LLaVA", "Qwen2VL", "Qwen2_VL",
    "InternVL", "MiniCPMV", "CogVLM", "InstructBLIP",
    "BLIP", "Flamingo", "Otter", "mPLUG",
})

_VLM_MODEL_TYPES = frozenset({
    "llava", "llava_next", "qwen2_vl", "internvl", "minicpmv",
    "blip_2", "instructblip", "flamingo", "idefics", "cogvlm",
    "paligemma", "pixtral",
})

_VLM_CONFIG_ATTRS = frozenset({
    "vision_config", "visual_config", "image_token_id",
    "vision_feature_layer", "image_aspect_ratio",
})

_VLM_MODEL_ATTRS = frozenset({
    "vision_model", "visual", "vision_tower", "vision_encoder",
    "visual_encoder", "image_encoder",
})


def _check_vlm(model, tokenizer, is_multimodal: bool) -> bool:
    """Return True if the model is a vision-language model."""
    if is_multimodal:
        return True
    cls_name = type(model).__name__
    for frag in _VLM_CLASS_FRAGMENTS:
        if frag.lower() in cls_name.lower():
            return True
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if getattr(cfg, "model_type", "") in _VLM_MODEL_TYPES:
            return True
        for attr in _VLM_CONFIG_ATTRS:
            if hasattr(cfg, attr):
                return True
    for attr in _VLM_MODEL_ATTRS:
        if hasattr(model, attr):
            return True
    return False


def _get_lib_versions() -> str:
    """Return a compact version string for key libraries without importing them."""
    import importlib.metadata
    
    parts = []
    for lib in ("transformers", "trl", "peft", "unsloth"):
        try:
            version = importlib.metadata.version(lib)
            parts.append(f"{lib}={version}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{lib}=not installed")
    return "  |  ".join(parts)


def _detect_quant(model) -> str:
    import torch
    has_uint8 = any(p.dtype == torch.uint8 for p in model.parameters())
    has_int8  = any(p.dtype == torch.int8  for p in model.parameters())
    has_bnb   = any(
        "4bit" in type(m).__name__.lower() or "8bit" in type(m).__name__.lower()
        for _, m in model.named_modules()
    )
    if has_bnb or has_uint8:
        return "BitsAndBytes 4-bit (uint8 packed)"
    if has_int8:
        return "BitsAndBytes 8-bit (int8)"
    import torch
    counts: dict = {}
    for p in model.parameters():
        k = str(p.dtype).replace("torch.", "")
        counts[k] = counts.get(k, 0) + p.numel()
    dominant = max(counts, key=counts.get) if counts else "unknown"
    return dominant
