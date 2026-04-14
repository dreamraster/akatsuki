# By dreamraster · dreaMSCend
"""
hmlcore/dense_pruner.py
========================
ShortGPT-style layer dropping for dense (non-MoE) transformer models.

Each transformer block is scored by how much it actually transforms its input.
Blocks with high cosine-similarity between their input and output hidden states
are "transparent" — they barely change the representation — and are candidates
for removal.

Importance score for layer l:
    I_l = 1 − mean_token( cosine_similarity(h_in_l, h_out_l) )

Low score → layer is redundant → dropped first.

Reference: "ShortGPT: Layers in Large Language Models are More Redundant
           Than You Expect" (arXiv 2403.03853, Men et al. 2024)

Public API
----------
find_decoder_layers(model) -> (layer_list, attr_path) | (None, None)
drop_dense_layers(model, tokenizer, dataset, prune_ratio, num_samples, max_cal_length)
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Layer discovery
# ─────────────────────────────────────────────────────────────────────────────

# Attribute paths to try, in preference order.
# Each entry is a dot-separated path from the top-level model object.
_LAYER_PATHS = [
    "model.layers",          # LLaMA, Mistral, Qwen2, Qwen3, DeepSeek
    "model.model.layers",    # PeftModel wrapping the above
    "transformer.h",         # GPT-2, GPT-J, Falcon (old)
    "model.transformer.h",   # PeftModel wrapping GPT-2
    "gpt_neox.layers",       # Pythia, GPT-NeoX
    "model.gpt_neox.layers",
    "layers",                # Bare model
    "model.blocks",          # Some custom architectures
    "decoder.layers",        # T5/BART-style encoder-decoders
]


# Attribute names that indicate a Mamba / SSM hybrid block.
# ShortGPT layer dropping renumbers block positions, which breaks llama.cpp /
# GGUF loaders that expect specific block types at specific indices.
_SSM_ATTRS = frozenset({
    "ssm_conv1d", "dt_proj", "A_log", "x_proj",   # Mamba-1 / Falcon-Mamba
    "dt_layernorm", "q_layernorm",                  # Mamba-2
    "mixer",                                        # generic Mamba wrapper
    "conv1d",                                       # Mamba conv
})


def _is_hybrid_ssm(layers: torch.nn.ModuleList) -> bool:
    """Return True if any block contains Mamba / SSM sub-modules."""
    for layer in layers:
        for name, _ in layer.named_modules():
            leaf = name.split(".")[-1]
            if leaf in _SSM_ATTRS:
                return True
    return False


def find_decoder_layers(model) -> tuple[torch.nn.ModuleList | None, str | None]:
    """Return (layer_list, attr_path) for the model's transformer blocks.

    Tries a prioritised list of common attribute paths.  Returns (None, None)
    if no suitable list is found, or if the model is a Mamba/SSM hybrid
    (where layer renumbering would corrupt the block-position mapping).
    """
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue

        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            logger.info("  Decoder layers found at: model.%s  (%d blocks)", path, len(obj))

            if _is_hybrid_ssm(obj):
                logger.warning(
                    "⚠️  Mamba/SSM hybrid architecture detected — ShortGPT layer "
                    "dropping is NOT safe for this model type. In GGUF/llama.cpp "
                    "the block type at each position is fixed by the architecture "
                    "definition; renumbering blocks after dropping layers shifts "
                    "SSM ↔ Attention types and produces an unloadable file. "
                    "Pruning skipped."
                )
                return None, None

            return obj, path

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Importance scoring  (ShortGPT angular distance)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of attribute paths to the token embedding table.
# Used by _get_initial_hidden_states to bypass the patched top-level forward.
_EMBED_PATHS = [
    "model.embed_tokens",           # LLaMA, Qwen2/3, Mistral, Phi, Gemma, DeepSeek
    "transformer.wte",              # GPT-2, GPT-J, Falcon (old)
    "model.transformer.wte",        # PeftModel wrapping GPT-2
    "gpt_neox.embed_in",            # Pythia, GPT-NeoX
    "model.gpt_neox.embed_in",
    "embed_tokens",                 # bare / custom
    "model.embed",
    "model.tok_embeddings",         # LLaMA-1 style
]

# Absolute position embedding paths (GPT-2 style — added on top of token emb).
# RoPE models (LLaMA, Qwen, Mistral) do NOT need this; RoPE is computed
# per-layer, not at the embedding stage.
_POS_EMBED_PATHS = [
    "transformer.wpe",              # GPT-2
    "model.transformer.wpe",
]


def _get_initial_hidden_states(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the initial hidden states (token embeddings ± position embeddings).

    Bypasses the model's top-level forward so that any framework-level patches
    (Unsloth, custom forward overrides) are not invoked.
    """
    hidden = None
    for path in _EMBED_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            hidden = obj(input_ids)
            break
        except (AttributeError, Exception):
            continue

    if hidden is None:
        raise ValueError(
            f"Cannot locate token embedding layer in {type(model).__name__}. "
            f"Tried: {', '.join(_EMBED_PATHS)}"
        )

    # Add absolute position embeddings for GPT-2-style models.
    seq_len = input_ids.shape[1]
    device  = input_ids.device
    for path in _POS_EMBED_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            hidden  = hidden + obj(pos_ids)
            break
        except AttributeError:
            continue

    return hidden


def _call_layer(
    layer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple | None = None,
) -> torch.Tensor:
    """Call a single decoder layer, trying multiple calling conventions.

    Modern transformers decoder layers accept (hidden_states, **optional_kwargs).
    We try progressively more explicit signatures until one works.  This handles:
      - Standard HF layers (LLaMA, Qwen2, Mistral, Phi, Gemma, Falcon …)
      - Unsloth-patched layers (Qwen3 requires cache_position after GRPO)
      - MoE layers (Mixtral, OLMoE, DeepSeek-MoE) that return router_logits
      - transformers ≥ 4.47 RoPE refactor where (cos, sin) is pre-computed
        outside the layer and passed in as position_embeddings

    Args:
        layer:                The decoder block to call.
        hidden_states:        Input tensor of shape (batch, seq, hidden).
        position_embeddings:  Optional (cos, sin) tuple from model.rotary_emb.
                              Required by transformers ≥ 4.47 architectures
                              (Mistral-v0.3, Mixtral, Llama-3.x, etc.).
    """
    seq_len        = hidden_states.shape[1]
    device         = hidden_states.device
    position_ids   = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long, device=device)

    # ── Candidate kwarg combinations ─────────────────────────────────────────
    # Ordered from simplest to most explicit.  The loop stops at the first
    # variant that succeeds AND returns a tensor with the correct shape.
    _base_variants = [
        {},
        {"use_cache": False},
        {"attention_mask": None, "use_cache": False},
        {"position_ids": position_ids, "use_cache": False},
        {"attention_mask": None, "position_ids": position_ids, "use_cache": False},
        # transformers ≥ 4.40 / Unsloth-patched Qwen3: cache_position required
        {"cache_position": cache_position, "use_cache": False},
        {"position_ids": position_ids, "cache_position": cache_position, "use_cache": False},
        {"attention_mask": None, "position_ids": position_ids,
         "cache_position": cache_position, "use_cache": False},
        # MoE layers (Mixtral, OLMoE, DeepSeek-MoE): output_router_logits defaults
        # vary by version — always pass False so the return is (hidden_states,)
        {"output_router_logits": False, "use_cache": False},
        {"attention_mask": None, "output_router_logits": False, "use_cache": False},
        {"position_ids": position_ids, "output_router_logits": False, "use_cache": False},
        {"attention_mask": None, "position_ids": position_ids,
         "output_router_logits": False, "use_cache": False},
        {"cache_position": cache_position, "output_router_logits": False, "use_cache": False},
        {"position_ids": position_ids, "cache_position": cache_position,
         "output_router_logits": False, "use_cache": False},
        {"attention_mask": None, "position_ids": position_ids, "cache_position": cache_position,
         "output_router_logits": False, "use_cache": False},
    ]

    # transformers ≥ 4.47 RoPE refactor: MistralAttention / MixtralAttention
    # expect a pre-computed (cos, sin) pair from the model-level rotary_emb.
    # Without it their forward does "cos, sin = position_embeddings" → TypeError.
    # Try these variants FIRST when available (most likely to succeed).
    _pe_variants: list = []
    if position_embeddings is not None:
        _pe = {"position_embeddings": position_embeddings}
        _pe_variants = [
            {**_pe, "use_cache": False},
            {"position_ids": position_ids, **_pe, "use_cache": False},
            {"attention_mask": None, "position_ids": position_ids, **_pe, "use_cache": False},
            {"cache_position": cache_position, **_pe, "use_cache": False},
            {"position_ids": position_ids, "cache_position": cache_position, **_pe, "use_cache": False},
            {"attention_mask": None, "position_ids": position_ids,
             "cache_position": cache_position, **_pe, "use_cache": False},
            # MoE + position_embeddings (Mixtral ≥ 4.47)
            {"output_router_logits": False, **_pe, "use_cache": False},
            {"position_ids": position_ids, "output_router_logits": False, **_pe, "use_cache": False},
            {"attention_mask": None, "position_ids": position_ids,
             "output_router_logits": False, **_pe, "use_cache": False},
            {"cache_position": cache_position, "output_router_logits": False, **_pe, "use_cache": False},
            {"position_ids": position_ids, "cache_position": cache_position,
             "output_router_logits": False, **_pe, "use_cache": False},
            {"attention_mask": None, "position_ids": position_ids, "cache_position": cache_position,
             "output_router_logits": False, **_pe, "use_cache": False},
        ]

    _last_exc: Exception | None = None
    _last_kwargs: dict | None = None

    for kwargs in (*_pe_variants, *_base_variants):
        try:
            out = layer(hidden_states, **kwargs)
            hs  = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(hs, torch.Tensor) and hs.shape == hidden_states.shape:
                return hs
        except TypeError as exc:
            _last_exc = exc
            _last_kwargs = kwargs
            continue       # wrong signature — try next variant
        except Exception as exc:
            _last_exc = exc
            _last_kwargs = kwargs
            continue       # runtime error (e.g. uninitialised kernel) — try next

    _hint = f"  last_kwargs={list(_last_kwargs.keys()) if _last_kwargs is not None else '[]'}"
    _cause = f"  last_error={type(_last_exc).__name__}: {_last_exc}" if _last_exc else ""
    
    # Specific hint for Unsloth-patched models that were reloaded via standard HF
    if isinstance(_last_exc, AttributeError) and "apply_qkv" in str(_last_exc):
        _cause += "\n  (HINT: This looks like an Unsloth-patched model missing its CUDA kernels. Ensure the model is reloaded via FastLanguageModel.)"

    raise RuntimeError(
        f"Could not call layer {type(layer).__name__} with any known signature.\n"
        f"{_hint}\n{_cause}"
    )


@torch.no_grad()
def _compute_layer_importance_via_hooks(
    model,
    layers: torch.nn.ModuleList,
    tokenizer,
    cal_texts: list,
    max_cal_length: int,
) -> torch.Tensor:
    """Hook-based fallback for layer importance scoring.

    Instead of calling layers directly (which fails for Unsloth-patched models
    that use fused kernels like apply_qkv), this function attaches forward hooks
    to each decoder block and lets the full model.forward() run normally.  The
    hooks capture (h_in, h_out) at each layer boundary, which is exactly the
    same quantity as the direct-call approach but works regardless of how the
    model's internals have been patched.

    Called automatically when the pre-flight probe finds that direct layer calls
    fail (e.g. AttributeError: 'Qwen2Attention' has no attribute 'apply_qkv').
    """
    logger.info(
        "  ↪ Switching to hook-based calibration (full model.forward + boundary hooks) "
        "for %s — Unsloth or custom forward patches detected.",
        type(model).__name__,
    )

    device     = next(model.parameters()).device
    num_layers = len(layers)
    importance = torch.zeros(num_layers)
    counts     = torch.zeros(num_layers)

    # ── Register per-layer hooks ──────────────────────────────────────────────
    # pre-hook: capture first tensor arg as h_in
    # post-hook: capture first tensor output as h_out
    h_in_cache:  dict[int, torch.Tensor] = {}
    h_out_cache: dict[int, torch.Tensor] = {}

    def _make_pre(idx: int):
        def _hook(module, args):
            hs = args[0] if args and isinstance(args[0], torch.Tensor) else None
            if hs is not None:
                h_in_cache[idx] = hs.detach().float()
        return _hook

    def _make_post(idx: int):
        def _hook(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(hs, torch.Tensor) and hs.ndim == 3:
                h_out_cache[idx] = hs.detach().float()
        return _hook

    _hooks = []
    for i, layer in enumerate(layers):
        _hooks.append(layer.register_forward_pre_hook(_make_pre(i)))
        _hooks.append(layer.register_forward_hook(_make_post(i)))

    # ── Tokenizer unwrap (VLM processors) ────────────────────────────────────
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    _warn_count = 0
    _warn_limit = 3

    _model_call_variants = [
        lambda ids: model(ids, use_cache=False),
        lambda ids: model(ids),
        lambda ids: model(input_ids=ids, use_cache=False),
    ]

    try:
        for text in tqdm(cal_texts, desc="Layer importance (hooks)"):
            # Tokenise
            try:
                input_ids = text_tok(
                    text,
                    return_tensors = "pt",
                    truncation     = True,
                    max_length     = max_cal_length,
                    padding        = False,
                )["input_ids"].to(device)
            except Exception as exc:
                if _warn_count < _warn_limit:
                    logger.warning("Hook-cal tokenisation failed: %s", exc)
                    _warn_count += 1
                continue

            h_in_cache.clear()
            h_out_cache.clear()

            # Run model forward — try variants until one succeeds
            _ran = False
            for _variant in _model_call_variants:
                try:
                    _variant(input_ids)
                    _ran = True
                    break
                except Exception:
                    continue

            if not _ran:
                if _warn_count < _warn_limit:
                    logger.warning("Hook-cal model forward failed for a sample — skipping.")
                    _warn_count += 1
                continue

            # Score each layer from cached activations
            for i in range(num_layers):
                if i not in h_in_cache or i not in h_out_cache:
                    continue
                h_in  = h_in_cache[i].view(-1,  h_in_cache[i].shape[-1])
                h_out = h_out_cache[i].view(-1, h_out_cache[i].shape[-1])
                if h_in.shape != h_out.shape:
                    continue
                cos = F.cosine_similarity(h_in, h_out, dim=-1).clamp(-1.0, 1.0)
                importance[i] += (1.0 - cos.mean().item())
                counts[i]     += 1

    finally:
        for h in _hooks:
            h.remove()

    # ── Normalise ─────────────────────────────────────────────────────────────
    active = counts > 0
    if not active.any():
        logger.error(
            "❌ Hook-based calibration: all samples failed — "
            "falling back to index-order pruning."
        )
        return torch.arange(num_layers, dtype=torch.float).flip(0)

    importance[active] /= counts[active]

    logger.info("  Layer importance scores via hooks (higher = more important):")
    for i, (imp, cnt) in enumerate(zip(importance.tolist(), counts.tolist())):
        bar = "█" * int(imp * 20) + "░" * (20 - int(imp * 20))
        logger.info("    Layer %2d │ %s │ %.4f  (n=%d)", i, bar, imp, int(cnt))

    return importance


@torch.no_grad()
def _compute_layer_importance(
    model,
    layers: torch.nn.ModuleList,
    tokenizer,
    cal_texts: list,
    max_cal_length: int,
) -> torch.Tensor:
    """Return a tensor of shape (num_layers,) with importance scores.

    Importance_l = 1 − mean_token( cosine_similarity(h_in_l, h_out_l) )
    Higher score → layer changes the representation more → more important.

    Attempts direct layer-by-layer forward first (bypasses model.forward so
    framework patches do not interfere).  If the pre-flight probe detects that
    direct calls fail (e.g. Unsloth-patched models with fused apply_qkv),
    automatically falls back to hook-based scoring via _compute_layer_importance_via_hooks.

    Args:
        cal_texts: Pre-selected list of plain strings from
                   hmlcore.calibration.build_calibration_samples().
    """
    model.eval()
    # Gradient checkpointing is only needed for backprop; disable it so the
    # no_grad calibration passes are not affected by recompute hooks.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    device     = next(model.parameters()).device
    num_layers = len(layers)
    importance = torch.zeros(num_layers)
    counts     = torch.zeros(num_layers)

    # ── Locate rotary embedding module ────────────────────────────────────────
    # transformers ≥ 4.47 refactored RoPE: (cos, sin) is now pre-computed at
    # the model level and passed into each attention layer as position_embeddings.
    # Without it, all MistralAttention / MixtralAttention calls fail with
    # "cannot unpack non-iterable NoneType object".
    _rotary_emb = None
    for _re_path in (
        "model.rotary_emb",                          # LLaMA, Qwen2 base model
        "model.model.rotary_emb",                    # PeftModel → inner model
        "base_model.model.model.rotary_emb",         # PeftModel LoRA wrapping
        "base_model.model.rotary_emb",               # PeftModel LoRA (single wrap)
        "rotary_emb",                                # bare model
    ):
        _obj = model
        try:
            for _attr in _re_path.split("."):
                _obj = getattr(_obj, _attr)
            if callable(_obj):
                _rotary_emb = _obj
                logger.debug(
                    "  Rotary embedding found at %s — will supply position_embeddings "
                    "to each layer call (transformers ≥ 4.47 RoPE refactor).",
                    _re_path,
                )
                break
        except AttributeError:
            continue

    def _call_rotary_emb(rope_fn, hidden: torch.Tensor, seq_len: int) -> tuple | None:
        """Try multiple rotary_emb call signatures; return (cos, sin) or None."""
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=hidden.device).unsqueeze(0)
        _attempts = [
            lambda: rope_fn(hidden, pos_ids),                    # (x, position_ids)        ← most common ≥ 4.40
            lambda: rope_fn(hidden, position_ids=pos_ids),       # keyword arg variant
            lambda: rope_fn(hidden, seq_len=seq_len),            # (x, seq_len)             ← old < 4.40
            lambda: rope_fn(hidden),                             # no extra args
        ]
        for _attempt in _attempts:
            try:
                result = _attempt()
                if isinstance(result, (tuple, list)) and len(result) == 2:
                    return result[0], result[1]
            except Exception:
                continue
        return None

    # ── Pre-flight layer probe ────────────────────────────────────────────────
    # Test a single fake input BEFORE running the full calibration loop so we
    # fail fast with a clear diagnostic instead of repeating the same warning
    # across all N calibration samples.
    _layer_cls = type(layers[0]).__name__
    logger.info(
        "  Pre-flight layer probe: %s (1 of %d blocks, type=%s)",
        "checking call compatibility", num_layers, _layer_cls,
    )
    try:
        _fake_ids  = torch.zeros((1, 16), dtype=torch.long, device=device)
        _fake_h    = _get_initial_hidden_states(model, _fake_ids)
        _probe_pe  = None
        if _rotary_emb is not None:
            _probe_pe = _call_rotary_emb(_rotary_emb, _fake_h, 16)
            if _probe_pe is not None:
                logger.debug("  Rotary probe OK — position_embeddings shapes: cos=%s sin=%s",
                             _probe_pe[0].shape, _probe_pe[1].shape)
            else:
                logger.warning(
                    "  ⚠️  rotary_emb probe: all call signatures failed — "
                    "position_embeddings will NOT be supplied to layer calls."
                )
        else:
            logger.debug("  No rotary_emb found — skipping position_embeddings variants.")
        _call_layer(layers[0], _fake_h, position_embeddings=_probe_pe)
        logger.info("  ✅ Layer probe passed — %s is callable, calibration will proceed.", _layer_cls)
    except Exception as _probe_exc:
        import traceback as _tb
        _trace = _tb.format_exc().strip()
        logger.warning(
            "  ⚠️  Direct layer probe FAILED for %s:\n"
            "     %s\n"
            "     Traceback:\n%s\n"
            "     Switching to hook-based calibration (model.forward + boundary hooks).",
            _layer_cls, _probe_exc, _trace,
        )
        return _compute_layer_importance_via_hooks(
            model, layers, tokenizer, cal_texts, max_cal_length
        )

    # For multimodal processors (Qwen2-VL, LLaVA …) unwrap to the inner
    # text tokenizer so plain string tokenisation works.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tok is not tokenizer:
        logger.info(
            "  Multimodal processor (%s) — using inner tokenizer (%s) for calibration.",
            type(tokenizer).__name__, type(text_tok).__name__,
        )

    _warn_count = 0
    _warn_limit = 3

    for text in tqdm(cal_texts, desc="Layer importance calibration"):
        # ── Tokenise ──────────────────────────────────────────────────────
        try:
            input_ids = text_tok(
                text,
                return_tensors = "pt",
                truncation     = True,
                max_length     = max_cal_length,
                padding        = False,
            )["input_ids"].to(device)
        except Exception as exc:
            if _warn_count < _warn_limit:
                logger.warning("Tokenisation failed (%d/%d): %s",
                               _warn_count + 1, len(cal_texts), exc)
                _warn_count += 1
            elif _warn_count == _warn_limit:
                logger.warning("(suppressing further tokenisation warnings)")
                _warn_count += 1
            continue

        # ── Layer-by-layer forward (bypasses patched model.forward) ───────
        try:
            hidden_states = _get_initial_hidden_states(model, input_ids)
        except Exception as exc:
            if _warn_count < _warn_limit:
                logger.warning("Embedding extraction failed (%d/%d): %s",
                               _warn_count + 1, len(cal_texts), exc)
                _warn_count += 1
            elif _warn_count == _warn_limit:
                logger.warning("(suppressing further embedding warnings)")
                _warn_count += 1
            continue

        # Compute (cos, sin) once per sample for architectures that need it.
        _position_embeddings = None
        if _rotary_emb is not None:
            _position_embeddings = _call_rotary_emb(
                _rotary_emb, hidden_states, hidden_states.shape[1]
            )  # None if all signatures fail — _call_layer will try without it

        for i, layer in enumerate(layers):
            h_in = hidden_states.detach().float()
            try:
                hidden_states = _call_layer(
                    layer, hidden_states,
                    position_embeddings=_position_embeddings,
                )
            except Exception as exc:
                if _warn_count < _warn_limit:
                    logger.warning("Layer %d call failed (%d/%d): %s",
                                   i, _warn_count + 1, len(cal_texts), exc)
                    _warn_count += 1
                elif _warn_count == _warn_limit:
                    logger.warning("(suppressing further layer warnings)")
                    _warn_count += 1
                break   # remaining layers in this sample are unscoreable

            h_out = hidden_states.detach().float()
            h_in_flat  = h_in.view(-1,  h_in.shape[-1])
            h_out_flat = h_out.view(-1, h_out.shape[-1])
            cos = F.cosine_similarity(h_in_flat, h_out_flat, dim=-1).clamp(-1.0, 1.0)
            importance[i] += (1.0 - cos.mean().item())
            counts[i]     += 1

    # ── Normalise ──────────────────────────────────────────────────────────
    active = counts > 0
    if not active.any():
        logger.error(
            "❌ All %d calibration samples failed — falling back to "
            "index-order pruning (drops last layers first). "
            "Check warnings above for root cause.",
            len(cal_texts),
        )
        return torch.arange(num_layers, dtype=torch.float).flip(0)

    importance[active] /= counts[active]

    logger.info("  Layer importance scores (higher = more important):")
    for i, (imp, cnt) in enumerate(zip(importance.tolist(), counts.tolist())):
        bar = "█" * int(imp * 20) + "░" * (20 - int(imp * 20))
        logger.info("    Layer %2d │ %s │ %.4f  (n=%d)", i, bar, imp, int(cnt))

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Layer removal
# ─────────────────────────────────────────────────────────────────────────────

def _remove_layers(
    model,
    layer_path: str,
    layers: torch.nn.ModuleList,
    keep_indices: list[int],
) -> None:
    """Replace the layer list with only the kept layers and update config."""
    kept = torch.nn.ModuleList([layers[i] for i in keep_indices])

    # Walk the attribute path and set the last component
    parts = layer_path.split(".")
    obj   = model
    for attr in parts[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], kept)

    # Update config
    cfg = getattr(model, "config", None)
    if cfg:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                old = getattr(cfg, attr)
                setattr(cfg, attr, len(keep_indices))
                logger.info("  config.%s: %d → %d", attr, old, len(keep_indices))
                break


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic quantization (score-guided 1-bit degradation)
# ─────────────────────────────────────────────────────────────────────────────

def _quantize_dense_layers(
    layers: torch.nn.ModuleList,
    quant_indices: list[int],
    importance: torch.Tensor | None = None,
) -> tuple[int, list[int]]:
    """Apply 1-bit quantization to all Linear layers in the specified blocks.

    Used by drop_dense_layers when --dynamicquant is active: instead of
    removing low-scored blocks, their Linear weights are binary-quantized
    in-place.  The ModuleList is not modified — layer count stays the same.

    Logs a per-layer verification table so the user can confirm quantization
    worked (binary_rows should be ≥ 99% for correctly quantized layers).

    Returns:
        (total_linear_count, quant_indices)
    """
    from hmlcore.quant import quantize_and_verify_module_1bit
    total = 0
    for i in quant_indices:
        score_str = f"  importance={importance[i].item():.4f}" if importance is not None else ""
        logger.info("  Layer %2d → 1-bit:%s", i, score_str)
        n, _ = quantize_and_verify_module_1bit(layers[i], prefix=f"layer[{i}]")
        total += n
    return total, quant_indices


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def drop_dense_layers(
    model,
    tokenizer,
    dataset,
    prune_ratio: float           = 0.5,
    num_samples: int             = 128,
    max_cal_length: int          = 2048,
    calibration_strategy: str   = "longest",
    dynamicquant: bool           = False,
) -> tuple[object, list[int] | None]:
    """
    Drop (or 1-bit quantize) the least-important transformer blocks.

    Uses ShortGPT's angular-distance importance metric.  Layers that barely
    transform their input (high cosine-similarity between h_in and h_out) are
    considered redundant and targeted first.

    Args:
        model:                  The merged, full-precision model to prune.
        tokenizer:              Tokenizer for encoding calibration prompts.
        dataset:                HF Dataset object.
        prune_ratio:            Fraction of layers to target (0.3 = 30%).
        num_samples:            Number of calibration samples.
        max_cal_length:         Max token length per calibration sample.
        calibration_strategy:   "longest" | "shortest" | "random" | "first".
                                "longest" (default) maximises hidden-state signal
                                per sample and produces more reliable scores.
        dynamicquant:           If True, quantize targeted layers to 1-bit
                                instead of removing them (--dynamicquant).
                                Layer count is preserved; Linear weights in
                                low-scored blocks are binary-quantized in-place.

    Returns:
        The pruned/quantized model (modified in-place).
    """
    layers, layer_path = find_decoder_layers(model)

    if layers is None:
        logger.warning(
            "⚠️  Could not locate transformer layer list in %s. "
            "Dense layer pruning skipped.  Recognised paths: %s",
            type(model).__name__,
            ", ".join(_LAYER_PATHS),
        )
        return model, None

    num_layers  = len(layers)
    num_to_drop = max(0, int(round(num_layers * prune_ratio)))
    num_to_keep = num_layers - num_to_drop

    if num_to_drop == 0:
        logger.info(
            "Dense pruning: prune_ratio=%.2f → 0 layers to drop from %d.",
            prune_ratio, num_layers,
        )
        return model, None

    params_before = _count_params(model)
    logger.info(
        "ShortGPT layer drop: %d → %d layers (dropping %d, ratio=%.2f)  "
        "[params before: %s]",
        num_layers, num_to_keep, num_to_drop, prune_ratio, f"{params_before:,}",
    )

    # ── Build calibration sample list ─────────────────────────────────────
    from hmlcore.calibration import build_calibration_samples
    cal_texts = build_calibration_samples(
        dataset,
        num_samples,
        strategy              = calibration_strategy,
        max_tokens_per_sample = max_cal_length,   # estimated tokens ≈ tokenizer tokens
        min_tokens_per_sample = 10,
    )
    if not cal_texts:
        logger.error(
            "❌ Calibration dataset produced no usable samples — "
            "ShortGPT scoring skipped, falling back to index-order pruning."
        )
        # Fallback: drop last layers (they tend to be least important)
        num_to_drop = max(0, int(round(len(layers) * prune_ratio)))
        keep_indices = list(range(len(layers) - num_to_drop))
        _remove_layers(model, layer_path, layers, keep_indices)
        return model, None

    # ── Score all layers ───────────────────────────────────────────────────
    importance = _compute_layer_importance(
        model, layers, tokenizer, cal_texts, max_cal_length
    )

    # ── Select which layers to keep ────────────────────────────────────────
    # Always preserve the first and last layers — they handle embedding
    # projection and final norm, and are disproportionately important.
    interior = list(range(1, num_layers - 1))

    if num_to_drop >= len(interior):
        # Extreme ratio: drop all interior layers, keep first+last only
        drop_indices  = interior
        keep_indices  = [0, num_layers - 1]
        logger.warning(
            "  prune_ratio=%.2f would remove all interior layers. "
            "Keeping only first and last block.",
            prune_ratio,
        )
    else:
        # Sort interior layers by importance (ascending = least important first)
        interior_imp  = importance[interior]
        worst_interior = sorted(
            interior,
            key=lambda i: importance[i].item(),
        )
        drop_indices  = sorted(worst_interior[:num_to_drop])
        keep_indices  = sorted(set(range(num_layers)) - set(drop_indices))

    logger.info(
        "  Targeting layers: %s  |  Action: %s",
        drop_indices,
        "1-bit quantize (--dynamicquant)" if dynamicquant else "remove",
    )

    if dynamicquant:
        # ── Quantize targeted layers to 1-bit (layer count unchanged) ─────
        n_linears, indices = _quantize_dense_layers(layers, drop_indices, importance=importance)
        logger.info(
            "✅ ShortGPT+DynQuant complete: %d layer(s) quantized to 1-bit  "
            "(%d Linear(s) total)  layer count: %d → %d (unchanged)  "
            "params: %s (count unchanged — layers retained at 1-bit precision)",
            len(drop_indices), n_linears, num_layers, num_layers, f"{params_before:,}",
        )
        return model, indices
    else:
        # ── Remove the layers ─────────────────────────────────────────────
        _remove_layers(model, layer_path, layers, keep_indices)

        params_after  = _count_params(model)
        reduction_pct = 100.0 * (params_before - params_after) / max(params_before, 1)
        logger.info(
            "✅ Dense layer drop complete: %d → %d layers  "
            "params: %s → %s  (%.1f%% reduction)",
            num_layers, len(keep_indices),
            f"{params_before:,}", f"{params_after:,}", reduction_pct,
        )

    return model, None
