# By dreamraster · dreaMSCend
"""
hmlcore/nodes/output_node.py
================================
OutputNode — final model save / merge / export stage.

Behaviour matrix (controlled by args.merge and args.quantize):

  No --merge:
    → Saves LoRA adapter weights only (small; base model needed at inference)

  --merge, quantize=bf16 or f16 (Unsloth):
    → save_pretrained_merged() → HuggingFace checkpoint

  --merge, quantize=f16/q8_0/q4_k (Unsloth, GGUF capable):
    → save_pretrained_gguf() → .gguf file for llama.cpp / Ollama

  --merge (standard PEFT fallback, no Unsloth):
    → merge_and_unload() → save_pretrained()

  Already merged (PrunerNode set args._already_merged = True):
    → Skip re-merge; direct save or GGUF export

Output is always written to <output_dir>/finale/.

Consumes:  model, tokenizer, args, use_unsloth
Produces:  finale_dir
"""

from __future__ import annotations

import glob
import logging
import os

import torch

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)

# Every value here triggers Unsloth's save_pretrained_gguf path instead of
# plain HF save.  "bf16" is intentionally excluded — it means "save as HF
# checkpoint".  Keep in sync with the choices list in config.py.
_GGUF_QUANTS = frozenset({
    "f16",
    "q8_0", "q6_k",
    "q5_k_m", "q5_k", "q4_k_m", "q4_k",
    "q3_k_m", "q2_k",
    "iq4_xs", "iq3_xxs",
    "iq2_xxs", "iq2_xs", "iq2_s",
    "iq1_s", "iq1_m",
})

# Weight-file patterns that can conflict when a directory is reused across runs
# (e.g. a 4-bit model.safetensors from a failed merge run sitting alongside a
# fresh pytorch_model.bin from a successful pruned run confuses GGUF converters).
_STALE_WEIGHT_PATTERNS = [
    "model*.safetensors",
    "pytorch_model*.bin",
    "model*.gguf",
]


def _purge_stale_weights(directory: str) -> None:
    """Best-effort deletion of old weight files before a fresh save.

    On Windows, safetensors files may be memory-mapped and undeletable while
    another process holds a handle.  We log a warning and continue — the
    save_pretrained call will overwrite safetensors in-place if it can.
    """
    for pattern in _STALE_WEIGHT_PATTERNS:
        for path in glob.glob(os.path.join(directory, pattern)):
            try:
                os.remove(path)
                logger.info("  Removed stale weight file: %s", os.path.basename(path))
            except OSError as exc:
                logger.warning(
                    "  Could not remove stale file %s (%s) — "
                    "save_pretrained will attempt to overwrite it.",
                    os.path.basename(path), exc,
                )


def _log_model_stats(model, tokenizer, save_dir: str) -> None:
    """Log a concise summary of the saved model: params, layers, dtype, MoE, disk size."""
    try:
        # ── Parameter counts ─────────────────────────────────────────────
        total_params    = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ── Memory estimate by dtype ─────────────────────────────────────
        # torch.uint8 is used by BnB 4-bit (2 values packed per byte).
        # For reporting we count bytes as-stored; note 4-bit models will
        # show ~half the "logical" parameter count.
        _BYTES = {
            torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
            torch.int8: 1, torch.uint8: 1,
        }
        dtype_counts: dict = {}
        total_bytes = 0
        for p in model.parameters():
            nb = _BYTES.get(p.dtype, p.element_size())
            total_bytes += p.numel() * nb
            key = str(p.dtype).replace("torch.", "")
            dtype_counts[key] = dtype_counts.get(key, 0) + p.numel()
        size_gb = total_bytes / 1e9
        dominant_dtype = max(dtype_counts, key=dtype_counts.get) if dtype_counts else "unknown"

        # ── Transformer layer count ───────────────────────────────────────
        num_layers = None
        cfg = getattr(model, "config", None)
        if cfg:
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(cfg, attr):
                    num_layers = getattr(cfg, attr)
                    break

        # ── MoE info ─────────────────────────────────────────────────────
        moe_line = ""
        try:
            from hmlcore.moe import find_moe_layers, get_num_experts
            moe_layers = find_moe_layers(model)
            if moe_layers:
                ne    = get_num_experts(moe_layers)
                top_k = "?"
                if cfg:
                    for attr in ("num_experts_per_tok", "top_k"):
                        if hasattr(cfg, attr):
                            top_k = getattr(cfg, attr)
                            break
                moe_line = (
                    f"  MoE:         {len(moe_layers)} layer(s)  |  "
                    f"{ne} experts/layer  |  top_k={top_k}"
                )
        except Exception:
            pass

        # ── Vocab / context ───────────────────────────────────────────────
        vocab_size   = getattr(cfg, "vocab_size",   None) if cfg else None
        max_pos      = getattr(cfg, "max_position_embeddings", None) if cfg else None
        hidden_size  = getattr(cfg, "hidden_size",  None) if cfg else None
        num_heads    = getattr(cfg, "num_attention_heads", None) if cfg else None

        # ── Disk size ─────────────────────────────────────────────────────
        disk_bytes = 0
        if os.path.isdir(save_dir):
            for entry in os.scandir(save_dir):
                if entry.is_file():
                    disk_bytes += entry.stat().st_size
        disk_gb = disk_bytes / 1e9

        # ── Tokenizer vocab ───────────────────────────────────────────────
        tok_vocab = getattr(tokenizer, "vocab_size", None)

        sep = "━" * 52
        logger.info(sep)
        logger.info("  Saved model summary")
        logger.info(sep)
        logger.info(
            "  Parameters:  %s total  |  %s trainable",
            f"{total_params:,}", f"{trainable_params:,}",
        )
        logger.info(
            "  Memory est:  %.2f GB  (%s dominant)",
            size_gb, dominant_dtype,
        )
        if num_layers is not None:
            extra = ""
            if hidden_size:
                extra += f"  hidden={hidden_size}"
            if num_heads:
                extra += f"  heads={num_heads}"
            logger.info("  Layers:      %d transformer block(s)%s", num_layers, extra)
        if vocab_size:
            logger.info(
                "  Vocab:       model=%s  tokenizer=%s  context=%s",
                f"{vocab_size:,}",
                f"{tok_vocab:,}" if tok_vocab else "n/a",
                f"{max_pos:,}" if max_pos else "n/a",
            )
        if moe_line:
            logger.info("%s", moe_line)
        if disk_bytes > 0:
            logger.info("  Disk:        %.2f GB  →  %s", disk_gb, save_dir)
        else:
            logger.info("  Output dir:  %s", save_dir)
        logger.info(sep)

    except Exception as exc:
        logger.warning("Could not compute model stats: %s", exc)


def _safe_save_pretrained(model, tokenizer, save_dir: str) -> None:
    """Save a merged model with automatic fallback chain.

    Some transformers versions track weight transformations applied at load time
    (fused QKV, rotated projections, etc.) and try to reverse them on save via
    revert_weight_conversion().  When those operations have no defined reverse_op
    the call raises NotImplementedError — with an empty message — before any file
    is written.  This affects both safe_serialization=True and False paths.

    Fallback chain:
      1. save_pretrained()                → safetensors (preferred for GGUF)
      2. save_pretrained(safe_serialization=False) → pytorch_model.bin
      3. torch.save(model.state_dict())   → pytorch_model.bin bypassing all
                                            transformers save hooks entirely
    """
    import torch as _torch

    # Safety net: strip any residual BnB config before writing config.json.
    # _dequantize_bnb_model() already does this, but models that bypass that
    # path (e.g. loaded in float from the start) may still carry the original
    # quantization_config from the HF hub — which would cause convert_hf_to_gguf
    # to raise NotImplementedError: "Quant method is not yet supported: bitsandbytes".
    _strip_bnb_config(model, save_dir)

    # ── Attempt 1: safetensors ────────────────────────────────────────────
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("  Saved as safetensors.")
        return
    except Exception as exc:
        logger.warning(
            "  safetensors save failed (%s: %s) — trying pytorch_model.bin.",
            type(exc).__name__, exc or "no message",
        )

    # ── Attempt 2: save_pretrained with pytorch_model.bin ─────────────────
    try:
        model.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir)
        logger.info("  Saved as pytorch_model.bin via save_pretrained.")
        return
    except Exception as exc:
        logger.warning(
            "  save_pretrained(safe_serialization=False) also failed (%s: %s) — "
            "falling back to direct torch.save (bypasses revert_weight_conversion).",
            type(exc).__name__, exc or "no message",
        )

    # ── Attempt 3: direct torch.save — bypasses all transformers save hooks ──
    # model.state_dict() returns the raw tensors without any weight-reversion
    # logic.  This is the same strategy used in _peft_merge_save.
    logger.info("  Building state dict for direct torch.save ...")
    state_dict = {
        k: v.detach().cpu().clone().contiguous()
        for k, v in model.state_dict().items()
    }
    bin_path = os.path.join(save_dir, "pytorch_model.bin")
    _torch.save(state_dict, bin_path)
    del state_dict

    # Save config (already updated with correct num_hidden_layers etc.)
    model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("  Saved as pytorch_model.bin via direct torch.save.")


def _strip_bnb_config(model, save_dir: str | None = None) -> None:
    """Remove BitsAndBytes quantization metadata from model.config and on-disk JSON.

    After dequantization all weights are plain floats, but model.config still
    carries quantization_config pointing to bitsandbytes.  When the model is
    saved and later passed to convert_hf_to_gguf.py, the converter reads
    config.json, sees quant_type='bitsandbytes', calls dequant_model() — and
    raises NotImplementedError because it doesn't know how to unpack BnB from
    a file whose weights are already float.

    This function performs both in-memory surgery and (if save_dir is provided)
    post-save file surgery to ensure the final directory is 'clean' for GGUF.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
        
    _bnb_attrs = (
        "quantization_config",
        "bitsandbytes",
        "_pre_quantization_dtype",
    )
    
    # ── In-Memory Surgery ──────────────────────────────────────────────────
    for attr in _bnb_attrs:
        # Try both direct attribute access and __dict__
        try:
            if hasattr(cfg, attr):
                setattr(cfg, attr, None)
                # For PretrainedConfig, None might still serialise. Try deleting.
                if hasattr(cfg, "__dict__"):
                    cfg.__dict__.pop(attr, None)
        except Exception:
            pass
            
    # Also look inside the attribute_map if it exists
    attribute_map = getattr(cfg, "attribute_map", {})
    if attribute_map:
        for k, v in list(attribute_map.items()):
            if v in _bnb_attrs:
                attribute_map.pop(k, None)

    # ── On-Disk Surgery (The Nuclear Option) ──────────────────────────────
    # If a save_dir is provided, we read the config.json back from disk and 
    # manually excise any residual BnB blocks. This is the only way to be 
    # 100% sure that GGUF converters won't see "null" or leftover fields.
    if save_dir and os.path.isdir(save_dir):
        config_path = os.path.join(save_dir, "config.json")
        if os.path.exists(config_path):
            try:
                import re
                with open(config_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Remove "quantization_config": { ... } logic
                # This regex handles nested braces for simple BnB configs
                content = re.sub(r'"quantization_config":\s*\{[^\}]+\},?', "", content)
                content = re.sub(r'"quantization_config":\s*null,?', "", content)
                content = re.sub(r'"_pre_quantization_dtype":\s*"[^"]*",?', "", content)
                content = re.sub(r'"_pre_quantization_dtype":\s*null,?', "", content)
                
                # Clean up any trailing commas before a closing brace
                content = re.sub(r',\s*\}', '}', content)
                
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.debug("  Cleaned config.json surgery complete (removed BnB metadata).")
            except Exception as exc:
                logger.warning("  Could not perform config.json surgery: %s", exc)


def _dequantize_bnb_model(model, dtype: "torch.dtype"):
    """Replace all BnB Linear4bit/Linear8bitLt layers with standard nn.Linear.

    After merge_and_unload() on a BnB-quantized PeftModel, the base layers are
    still Linear4bit — the LoRA delta was merged into the quantized weights but
    the storage format is unchanged.  GGUF converters and safetensors both
    require plain float tensors, so we must explicitly dequantize here.

    Also strips model.config.quantization_config so that convert_hf_to_gguf.py
    and other tools do not try to re-dequantize already-float weights.

    bitsandbytes.functional.dequantize_4bit dispatches to a C++ kernel on CPU
    and a CUDA kernel on CUDA, so this works in both device contexts.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.warning("  bitsandbytes not installed — skipping BnB dequantize.")
        _strip_bnb_config(model)
        return model

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            if "." in name:
                parent_path, attr = name.rsplit(".", 1)
                parent = model
                for part in parent_path.split("."):
                    parent = getattr(parent, part)
            else:
                parent, attr = model, name
            replacements.append((parent, attr, module))

    if not replacements:
        # No BnB layers, but config may still reference bitsandbytes from
        # the original load — strip it anyway.
        _strip_bnb_config(model)
        return model

    logger.info("  Dequantizing %d BnB layer(s) to %s ...", len(replacements), dtype)
    for parent, attr, module in replacements:
        device = next(iter(module.parameters())).device
        if isinstance(module, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
            ).to(dtype)
        else:
            w = module.weight.data.to(dtype)

        new_layer = torch.nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None,
            device=device, dtype=dtype,
        )
        new_layer.weight = torch.nn.Parameter(w)
        if module.bias is not None:
            new_layer.bias = torch.nn.Parameter(module.bias.data.to(dtype))
        setattr(parent, attr, new_layer)

    logger.info("  BnB dequantization complete (%d layers).", len(replacements))

    # Weights are now plain floats — strip the BnB config so downstream tools
    # (convert_hf_to_gguf.py, Ollama, llama.cpp loaders) don't try to
    # dequantize them again.
    _strip_bnb_config(model)
    return model


def _peft_merge_save(model, tokenizer, save_dir: str):
    """Windows-safe PEFT merge fallback.

    4-bit quantised models cannot be merged directly — transformers raises
    NotImplementedError on save_pretrained when still in bnb 4-bit form, and
    Unsloth's save_pretrained_merged can fail with RoPE tensor-size mismatches
    when max_seq_length < the model's original rope size.

    Strategy: reload the base model in bf16 on CPU (device_map="cpu" avoids
    Windows mmap locks), attach the adapter, merge, save as pytorch_model.bin
    (plain torch.save bypasses the safetensors mmap lock on Windows).

    Returns:
        The merged bf16 model (on CPU).  Caller should use it for stats then
        let it go out of scope for GC.  Returns the original model on failure.
    """
    import gc
    import shutil
    import tempfile

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info("🔀 PEFT fallback merge: reloading base model in bf16 to avoid "
                "4-bit dequantisation / RoPE-size errors ...")

    # Retrieve base model path from the embedded peft config
    try:
        base_model_name = model.peft_config[
            list(model.peft_config.keys())[0]
        ].base_model_name_or_path
    except Exception:
        base_model_name = model.base_model.model.config._name_or_path

    logger.info("  Base model: %s", base_model_name)

    tmp_adapter = tempfile.mkdtemp(prefix="hml_adapter_tmp_")
    try:
        model.save_pretrained(tmp_adapter)
        tokenizer.save_pretrained(tmp_adapter)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("  Loading fresh base model on CPU ...")

        # Load WITH the embedded quantization_config intact.
        # If we strip it, transformers builds a clean bf16 model (weights shaped
        # e.g. [1024, 3072]) but then tries to load BnB-packed tensors from disk
        # (shaped [1572864, 1]) → shape mismatch → RuntimeError.
        # With quantization_config present, BnB correctly unpacks the on-disk
        # weights.  We then call merge_and_unload() and explicitly dequantize the
        # remaining Linear4bit layers to clean bf16 via _dequantize_bnb_model().
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype       = dtype,
            device_map        = "cpu",
            trust_remote_code = True,
            low_cpu_mem_usage = True,
        )

        logger.info("  Attaching adapter and merging ...")
        merged = PeftModel.from_pretrained(base, tmp_adapter)
        merged = merged.merge_and_unload()
        del base
        gc.collect()

        # After merge_and_unload(), BnB Linear4bit layers still hold quantized
        # weights — the LoRA delta was merged into them but the storage format
        # is unchanged.  Dequantize to plain bf16 so state_dict() is clean.
        merged = _dequantize_bnb_model(merged, dtype)

        # Copy every tensor to a plain buffer — untie shared weights
        seen_ptrs: dict = {}
        state_dict = {}
        for k, v in merged.state_dict().items():
            ptr = v.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = k
            state_dict[k] = v.detach().cpu().clone().contiguous()

        os.makedirs(save_dir, exist_ok=True)
        logger.info("  Writing pytorch_model.bin → %s", save_dir)
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        del state_dict
        gc.collect()

        cfg_obj = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        cfg_obj.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("✅ PEFT merge complete (pytorch_model.bin).")

        # Return merged so the caller can report accurate post-merge stats.
        # Peak memory is unchanged (state_dict is already freed above).
        return merged

    finally:
        shutil.rmtree(tmp_adapter, ignore_errors=True)


# GGUF tensor-name suffixes for LLaMA-family models (Qwen2, Mistral, LLaMA,
# DeepSeek-dense, Phi-3 …) — used to build per-tensor llama-quantize overrides.
# Key: PyTorch attribute suffix inside a decoder block.
# Value: GGUF tensor name suffix (appended to "blk.{i}.").
_GGUF_TENSOR_MAP = {
    "self_attn.q_proj":   "attn_q",
    "self_attn.k_proj":   "attn_k",
    "self_attn.v_proj":   "attn_v",
    "self_attn.o_proj":   "attn_output",
    "mlp.gate_proj":      "ffn_gate",
    "mlp.up_proj":        "ffn_up",
    "mlp.down_proj":      "ffn_down",
    # Phi-3 / Phi-3.5 naming
    "self_attn.qkv_proj": "attn_qkv",
    "mlp.fc1":            "ffn_up",
    "mlp.fc2":            "ffn_down",
}


def _log_dynamic_gguf_guidance(
    save_dir: str,
    layer_indices: list[int] | None = None,
    expert_info: dict[str, list[int]] | None = None,
    base_quant: str = "q8_0",
    model: torch.nn.Module | None = None,
) -> None:
    """Log llama-quantize commands to produce a true mixed-precision GGUF.

    When --dynamicquant is active, the saved HF model has tiers of weights:
      • Important layers/experts (full bf16) — should be quantized to base_quant
      • Pre-quantized layers/experts (binary ±scale) — near-losslessly
        representable as iq1_s in GGUF.

    Standard GGUF converters apply one quant type to everything.  To get a
    true dynamic GGUF (different quant types per tensor), use llama.cpp's
    llama-quantize with --tensor-type overrides.
    """
    if not layer_indices and not expert_info:
        return

    f16_name   = os.path.join(save_dir, "model_f16.gguf")
    mixed_name = os.path.join(save_dir, f"model_dynamic_{base_quant}.gguf")

    # Build the --tensor-type flags
    tensor_flags = []
    
    # Recommendation: q2_k requires 256-column alignment and usually an imatrix.
    # If not aligned, llama.cpp falls back to iq4_nl (much larger).
    # q2_k is a safer fallback for non-aligned models or when no imatrix is available.
    target_dq_type = "q2_k"
    
    # Heuristic: check if we should suggest q2_k instead of iq1_s
    # (We don't have easy access to hidden_size here without the model,
    # but we can try to get it from ctx or just warn the user).
    is_aligned = True
    try:
        # qwen2 hidden_size is usually at model.config.hidden_size
        cfg = getattr(model, "config", None)
        hsz = getattr(cfg, "hidden_size", 4096)
        if hsz % 256 != 0:
            is_aligned = False
            target_dq_type = "q2_k" # Safer fallback
    except Exception:
        pass

    # Case 1: Dense layer dropping / DynamicQuant
    if layer_indices:
        for i in sorted(layer_indices):
            for _suffix in _GGUF_TENSOR_MAP.values():
                tensor_flags.append(f'--tensor-type "blk.{i}.{_suffix}.weight={target_dq_type}"')

    # Case 2: MoE Expert DynamicQuant
    if expert_info:
        # expert_info is { "layer.name": [expert_indices] }
        # Note: GGUF tensor naming for MoE experts varies by architecture.
        # This implementation assumes the common blk.N.ffn_gate/up/down pattern.
        # If experts are stacked in one tensor, per-expert quant is not possible
        # via llama-quantize; if separate, this works.
        for layer_path, eids in expert_info.items():
            # Extract block index from path (e.g. "model.layers.13" -> 13)
            import re
            match = re.search(r"\d+", layer_path)
            if match:
                idx = match.group()
                for eid in eids:
                    # Generic guess for expert tensor names
                    for _suffix in ("ffn_gate", "ffn_up", "ffn_down"):
                        tensor_flags.append(f'--tensor-type "blk.{idx}.{_suffix}.{eid}.weight={target_dq_type}"')

    if not tensor_flags:
        return

    flags_str = " ".join(tensor_flags)

    logger.info("")
    logger.info("━" * 68)
    logger.info("  ⚡ Dynamic GGUF export — mixed per-weight quantization")
    logger.info("━" * 68)
    logger.info("")
    if layer_indices:
        logger.info(
            "  %d layer(s) were 1-bit pre-quantized (indices: %s).",
            len(layer_indices), layer_indices,
        )
    if expert_info:
        logger.info(
            "  %d experts across %d layer(s) were 1-bit pre-quantized.",
            sum(len(v) for v in expert_info.values()), len(expert_info),
        )

    logger.info(
        "  Standard GGUF converters apply a single quant type to all tensors."
    )
    logger.info(
        "  For a true dynamic GGUF (pre-quantized → %s, others → %s), "
        "use llama-quantize:", target_dq_type, base_quant,
    )
    logger.info("")
    if not is_aligned:
        logger.info(
            "  ⚠️  Note: Model hidden_size is not divisible by 256. Using %s instead of iq1_s\n"
            "      to avoid llama.cpp alignment errors and iq4_nl fallbacks.", target_dq_type
        )
    logger.info("  Note: iq* quants (like iq1_s) REQUIRE an importance matrix (--imatrix).")
    logger.info("        If they fail, use --imatrix <file.dat> or use %s for those layers.", target_dq_type)
    logger.info("")
    logger.info("  Step 1 — Convert to F16 GGUF:")
    logger.info(
        "    python convert_hf_to_gguf.py %s --outtype f16 --outfile %s",
        save_dir, f16_name,
    )
    logger.info("")
    logger.info("  Step 2 — Re-quantize with per-layer/expert overrides (llama.cpp ≥ b3000):")
    logger.info(
        "    llama-quantize \\\n  %s \\\n  %s %s %s",
        flags_str, f16_name, mixed_name, base_quant,
    )
    logger.info("")
    logger.info(
        "  Result: blk.N.* (quantized) → %s  |  all other tensors → %s",
        target_dq_type, base_quant,
    )
    logger.info("━" * 68)
    logger.info("")


class OutputNode(BaseNode):
    NAME = "OutputNode"
    INPUT_KEYS = ("model", "tokenizer", "args", "use_unsloth")
    OUTPUT_KEYS = ("finale_dir",)

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "model", "tokenizer", "args", "use_unsloth")
        args        = ctx["args"]
        model       = ctx["model"]
        tokenizer   = ctx["tokenizer"]
        use_unsloth = ctx["use_unsloth"]

        finale_dir = os.path.join(args.output_dir, "finale")
        os.makedirs(finale_dir, exist_ok=True)

        # Remove stale weight files from previous runs so a leftover
        # model.safetensors (e.g. from a failed 4-bit run) doesn't sit
        # alongside the new weights and mislead GGUF converters.
        # We delete files individually rather than rmtree — Windows cannot
        # delete memory-mapped safetensors files held by another process,
        # and rmtree would raise a PermissionError that kills the pipeline.
        _purge_stale_weights(finale_dir)

        ctx["finale_dir"] = finale_dir

        do_merge       = getattr(args, "merge", False)
        quant          = getattr(args, "quantize", "bf16")
        already_merged = getattr(args, "_already_merged", False)

        logger.info(
            "💾 Saving model → %s  (merge=%s, quantize=%s, already_merged=%s)",
            finale_dir, do_merge, quant, already_merged,
        )

        # stats_model tracks what was actually saved — updated whenever we
        # produce a freshly merged bf16 model so _log_model_stats is accurate.
        stats_model = model

        # ── Already-merged path (REAP / ShortGPT pruner output) ─────────────
        if already_merged:
            if use_unsloth and quant in _GGUF_QUANTS and hasattr(model, "save_pretrained_gguf"):
                logger.info("🔀 Unsloth GGUF export (%s) → %s", quant, finale_dir)
                try:
                    model.save_pretrained_gguf(finale_dir, tokenizer,
                                               quantization_method=quant)
                except Exception as exc:
                    logger.warning(
                        "⚠️  Unsloth GGUF export failed (%s). "
                        "Falling back to HF format. "
                        "You can convert manually: "
                        "python convert_hf_to_gguf.py %s",
                        exc, finale_dir,
                    )
                    _safe_save_pretrained(model, tokenizer, finale_dir)
            else:
                logger.info("💾 HF save (already merged) → %s", finale_dir)
                _safe_save_pretrained(model, tokenizer, finale_dir)

        # ── Normal merge path ─────────────────────────────────────────────────
        elif do_merge and use_unsloth:
            if quant in _GGUF_QUANTS and hasattr(model, "save_pretrained_gguf"):
                logger.info("🔀 Unsloth GGUF export (%s) → %s", quant, finale_dir)
                try:
                    model.save_pretrained_gguf(finale_dir, tokenizer,
                                               quantization_method=quant)
                except Exception as exc:
                    logger.warning(
                        "⚠️  Unsloth GGUF export failed (%s). "
                        "Falling back to HF format. "
                        "You can convert manually: "
                        "python convert_hf_to_gguf.py %s",
                        exc, finale_dir,
                    )
                    stats_model = _peft_merge_save(model, tokenizer, finale_dir)
            else:
                logger.info("🔀 Unsloth merge (%s) → %s", quant, finale_dir)
                try:
                    model.save_pretrained_merged(finale_dir, tokenizer, save_method=quant)
                except Exception as exc:
                    logger.warning(
                        "⚠️  Unsloth merge failed (%s). "
                        "Falling back to standard PEFT merge → %s",
                        exc, finale_dir,
                    )
                    stats_model = _peft_merge_save(model, tokenizer, finale_dir)

        elif do_merge:
            logger.info("🔀 Standard PEFT merge → %s", finale_dir)
            stats_model = model.merge_and_unload()
            stats_model.save_pretrained(finale_dir)
            tokenizer.save_pretrained(finale_dir)

        else:
            logger.info("💾 Saving LoRA adapter only → %s", finale_dir)
            model.save_pretrained(finale_dir)
            tokenizer.save_pretrained(finale_dir)

        # ── Post-Save Surgery ────────────────────────────────────────────────
        # Force a clean-up of config.json on disk to remove any BnB metadata
        # that may have been re-inserted by transformers/peft save hooks.
        _strip_bnb_config(model, finale_dir)

        _log_model_stats(stats_model, tokenizer, finale_dir)

        # ── Dynamic GGUF guidance ─────────────────────────────────────────────
        # If --dynamicquant was used, provide the user with the llama-quantize
        # commands needed to produce a mixed-precision GGUF.
        if getattr(args, "dynamicquant", False):
            layer_indices = ctx.get("quantized_layers")
            expert_info   = ctx.get("quantized_experts")
            base_quant    = quant if quant in _GGUF_QUANTS else "q8_0"
            _log_dynamic_gguf_guidance(finale_dir, layer_indices, expert_info, base_quant, model=model)

        logger.info("🎉 Model saved → %s", finale_dir)
