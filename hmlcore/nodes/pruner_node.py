# By dreamraster · dreaMSCend
"""
hmlcore/nodes/pruner_node.py
================================
PrunerNode — REAP (Router-weighted Expert Activation Pruning) stage.

Skipped when:
  • args.prune_experts is False / not set

Pipeline:
  1. Dequantise + merge LoRA adapter (required before any pruning)
     - 4-bit / 8-bit BnB models: reload base in bf16 on CPU, re-attach
       adapter from a temp dir, then merge_and_unload().  This is the
       ONLY correct path — you cannot merge into packed uint8/int4 weights.
     - Already-float models: direct merge_and_unload().
  2. Detect model topology on the MERGED bf16 model (NOT the PeftModel).
     PeftModel's LoraModel wrapper hides attribute paths like model.layers;
     topology detection MUST run after merge so plain HF paths resolve.
  3. Route to pruning:
     - MoE model  → REAP calibration + expert pruning
     - Dense model → ShortGPT layer dropping

Consumes:  model, tokenizer, dataset, args, use_unsloth
Produces:  (model is modified in-place; _already_merged flag set on args)
"""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile

import torch

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)

# BitsAndBytes quantized dtypes — cannot be merged directly
_QUANT_DTYPES = {torch.uint8, torch.int8}


def _is_quantized(model) -> bool:
    """Return True if the model has BnB-quantized layers (4-bit or 8-bit)."""
    # Heuristic 1: dominant parameter dtype is uint8 / int8
    for p in model.parameters():
        if p.dtype in _QUANT_DTYPES:
            return True
    # Heuristic 2: module class name contains "4bit" or "8bit"
    for _, m in model.named_modules():
        cls = type(m).__name__
        if "4bit" in cls.lower() or "8bit" in cls.lower():
            return True
    return False


def _merge_lora_via_bf16_reload(model, tokenizer) -> object:
    """
    Dequantise + merge LoRA by reloading the base model in bf16.

    Required when the model is BnB 4-bit / 8-bit quantized: merge_and_unload()
    raises NotImplementedError on quantized weights.  Strategy:
      1. Save the LoRA adapter weights to a temp dir.
      2. Reload the base model from HF / local path in bf16 on CPU.
         (device_map="cpu" avoids Windows mmap locks.)
      3. Re-attach the adapter and call merge_and_unload().
      4. Move merged model to GPU if available.

    Returns the merged bf16 model (no LoRA, no quantization).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # ── Resolve base model path ───────────────────────────────────────────
    base_model_name: str | None = None
    try:
        base_model_name = model.peft_config[
            list(model.peft_config.keys())[0]
        ].base_model_name_or_path
    except Exception:
        pass
    if not base_model_name:
        try:
            base_model_name = model.base_model.model.config._name_or_path
        except Exception:
            pass
    if not base_model_name:
        raise NodeError(
            "Cannot resolve base model path for bf16 reload merge. "
            "Ensure the model has a valid peft_config or config._name_or_path."
        )

    logger.info(
        "  Base model for reload: %s", base_model_name,
    )

    tmp_adapter = tempfile.mkdtemp(prefix="hml_prune_adapter_")
    try:
        # Save LoRA adapter + tokenizer
        logger.info("  Saving LoRA adapter to temp dir ...")
        model.save_pretrained(tmp_adapter)
        tokenizer.save_pretrained(tmp_adapter)

        # Free the quantized model from GPU memory before loading bf16
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # IMPORTANT: load directly onto GPU (not CPU then .cuda()).
        # Unsloth patches attention with a custom CUDA kernel (apply_qkv) that
        # it only initialises when the model loads onto a CUDA device.  Loading
        # to CPU then calling .cuda() moves tensors but never triggers that init,
        # leaving apply_qkv unset and breaking the calibration forward pass.
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
            logger.info(
                "  Reloading base model in %s on GPU:%d ...",
                dtype, torch.cuda.current_device(),
            )
        else:
            device_map = "cpu"
            logger.info("  Reloading base model in %s on CPU ...", dtype)

        # Load WITH the embedded quantization_config intact.
        # If we strip it, transformers builds a clean bf16 model but then fails
        # to load the BnB-packed on-disk weights (shape mismatch).  With the
        # config intact, BnB correctly unpacks the weights.  We then explicitly
        # dequantize any remaining Linear4bit layers after merge_and_unload().
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype       = dtype,
            device_map        = device_map,
            trust_remote_code = True,
            low_cpu_mem_usage = True,
        )

        logger.info("  Attaching adapter and merging ...")
        merged = PeftModel.from_pretrained(base, tmp_adapter)
        merged = merged.merge_and_unload()

        # After merge_and_unload(), BnB Linear4bit layers still hold quantized
        # weights.  Dequantize to plain bf16 so calibration and state_dict() work.
        from hmlcore.nodes.output_node import _dequantize_bnb_model
        merged = _dequantize_bnb_model(merged, dtype)

        return merged

    finally:
        shutil.rmtree(tmp_adapter, ignore_errors=True)


class PrunerNode(BaseNode):
    NAME = "PrunerNode"
    INPUT_KEYS = ("model", "tokenizer", "dataset", "args", "use_unsloth")
    OUTPUT_KEYS = ()

    def should_run(self, ctx: NodeContext) -> bool:
        args = ctx.get("args")
        if args is None:
            return False
        return getattr(args, "prune_experts", False) or getattr(args, "prune_only", False)

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "model", "tokenizer", "dataset", "args", "use_unsloth")
        args        = ctx["args"]
        model       = ctx["model"]
        tokenizer   = ctx["tokenizer"]
        dataset     = ctx["dataset"]
        use_unsloth = ctx["use_unsloth"]

        # ── Step 1: Merge LoRA adapter FIRST ─────────────────────────────────
        # CRITICAL: topology detection (find_decoder_layers) MUST run on the
        # merged bf16 model, NOT on the PeftModel.  Inside a PeftModel the
        # LoraModel wrapper hides the base attribute paths (e.g. model.layers
        # is actually on LoraModel.model, which is NOT the same as model.model
        # when accessed through PEFT's __getattr__ proxy).  Running detection
        # on the PeftModel causes find_decoder_layers to return None, which
        # makes PrunerNode exit early — pruning never runs.
        logger.info("🔀 Merging LoRA adapter before pruning ...")

        quantized = _is_quantized(model)
        if quantized:
            logger.info(
                "  ⚠️  Model is quantized (uint8/int8 weights detected). "
                "Direct merge is not possible — reloading base model in bf16 "
                "to dequantise and merge.  This requires re-downloading / "
                "reloading from local cache."
            )
            try:
                model = _merge_lora_via_bf16_reload(model, tokenizer)
            except Exception as exc:
                raise NodeError(
                    f"bf16 reload merge failed: {exc}. "
                    f"Ensure '{getattr(args, 'student_model', '')}' is accessible."
                ) from exc
        else:
            # Float model — direct merge works
            try:
                model = model.merge_and_unload()
            except Exception as exc:
                logger.warning(
                    "  Direct merge failed (%s) — falling back to bf16 reload.", exc
                )
                try:
                    model = _merge_lora_via_bf16_reload(model, tokenizer)
                except Exception as exc2:
                    raise NodeError(f"All merge strategies failed: {exc2}") from exc2

        ctx["model"] = model
        # Signal to OutputNode that the adapter is already merged
        args._already_merged = True
        logger.info("✅ LoRA merge + dequantisation complete.")

        # Validate: the merged model must have 0 trainable LoRA parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dominant_dtype = _dominant_dtype(model)
        logger.info(
            "  Merged model: dtype=%s  trainable_params=%s",
            dominant_dtype, f"{trainable:,}",
        )
        if trainable > 0:
            logger.warning(
                "  ⚠️  %s trainable parameters remain after merge — "
                "the adapter may not have fully merged.  Pruning will "
                "proceed but results may be suboptimal.",
                f"{trainable:,}",
            )

        # ── Step 2: Detect topology on the MERGED bf16 model ─────────────────
        from hmlcore.moe import find_moe_layers
        from hmlcore.dense_pruner import find_decoder_layers
        moe_layers = find_moe_layers(model)
        is_moe     = len(moe_layers) > 0

        if is_moe:
            logger.info(
                "🎯 Topology: MoE model — %d expert layer(s) found. "
                "Using REAP expert pruning.",
                len(moe_layers),
            )
        else:
            # --prune_experts explicitly requests REAP (MoE expert pruning).
            # A dense model has no experts to prune — warn clearly and bail out.
            # Exception: --prune_only means "use whatever pruning fits this model"
            # (apply_args sets prune_experts=True for --prune_only, so we must
            # distinguish the two cases explicitly).
            explicit_reap = (
                getattr(args, "prune_experts", False)
                and not getattr(args, "prune_only", False)
            )
            if explicit_reap:
                logger.warning(
                    "⚠️  --prune_experts was passed but %s has no MoE expert "
                    "layers.  REAP requires a Mixture-of-Experts architecture "
                    "(e.g. Qwen3-30B-A3B, OLMoE-1B-7B, Mixtral-8x7B).  "
                    "Falling back to ShortGPT dense layer dropping.",
                    type(model).__name__,
                )

            dec_layers, _ = find_decoder_layers(model)
            if dec_layers is None:
                logger.error(
                    "❌ Cannot prune %s: no MoE expert layers and no "
                    "recognisable transformer block list found. "
                    "Pruning skipped (model already merged and saved).",
                    type(model).__name__,
                )
                return
            logger.info(
                "ℹ️  Topology: dense model (%d transformer blocks). "
                "Using ShortGPT layer dropping (REAP requires MoE).",
                len(dec_layers),
            )

        # ── Step 3: Prune or quantize ─────────────────────────────────────────
        calibration_samples  = getattr(args, "calibration_samples", 128)
        calibration_strategy = getattr(args, "calibration_strategy", "longest")
        prune_ratio          = getattr(args, "prune_ratio", 0.5)
        max_length           = getattr(args, "max_length", 2048)
        dynamicquant         = getattr(args, "dynamicquant", False)

        if dynamicquant:
            logger.info(
                "⚡ --dynamicquant: low-scored experts/layers will be 1-bit quantized "
                "instead of removed (Unsloth Dynamic-style score-guided precision)."
            )

        if is_moe:
            from hmlcore.moe import reap_prune_moe
            logger.info(
                "🔬 REAP: %d samples, strategy=%s, prune_ratio=%.2f%s",
                calibration_samples, calibration_strategy, prune_ratio,
                "  [dynamicquant=1-bit]" if dynamicquant else "",
            )
            try:
                model, quant_info = reap_prune_moe(
                    model                = model,
                    tokenizer            = tokenizer,
                    dataset              = dataset,
                    prune_ratio          = prune_ratio,
                    num_samples          = calibration_samples,
                    max_cal_length       = max_length,
                    calibration_strategy = calibration_strategy,
                    dynamicquant         = dynamicquant,
                )
                if dynamicquant and quant_info:
                    ctx["quantized_experts"] = quant_info
            except Exception as exc:
                raise NodeError(f"REAP pruning failed: {exc}") from exc

        else:
            from hmlcore.dense_pruner import drop_dense_layers
            logger.info(
                "✂️  ShortGPT: %d samples, strategy=%s, prune_ratio=%.2f%s",
                calibration_samples, calibration_strategy, prune_ratio,
                "  [dynamicquant=1-bit]" if dynamicquant else "",
            )
            try:
                model, quant_indices = drop_dense_layers(
                    model                = model,
                    tokenizer            = tokenizer,
                    dataset              = dataset,
                    prune_ratio          = prune_ratio,
                    num_samples          = calibration_samples,
                    max_cal_length       = max_length,
                    calibration_strategy = calibration_strategy,
                    dynamicquant         = dynamicquant,
                )
                if dynamicquant and quant_indices:
                    ctx["quantized_layers"] = quant_indices
            except Exception as exc:
                raise NodeError(f"Dense layer pruning failed: {exc}") from exc

        ctx["model"] = model


def _dominant_dtype(model) -> str:
    counts: dict = {}
    for p in model.parameters():
        k = str(p.dtype).replace("torch.", "")
        counts[k] = counts.get(k, 0) + p.numel()
    return max(counts, key=counts.get) if counts else "unknown"
