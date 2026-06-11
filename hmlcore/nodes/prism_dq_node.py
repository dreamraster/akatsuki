# By dreamraster · dreaMSCend
"""
hmlcore/nodes/prism_dq_node.py
================================
PRISMDQNode — Post-OutputNode PRISM Dynamic Quantization stage.

Runs AFTER OutputNode has saved the merged BF16 model to finale_dir.
Reads the saved weights back from disk into plain float, computes the
7 structural metrics from hmlcore.prism_dq, runs the Lagrangian allocator,
and emits a llama-quantize recipe + optional F16 GGUF → PRISM-DQ GGUF.

Activation condition:
    args.prism_dq == True  AND  ctx["finale_dir"] is set

Consumes:  finale_dir, args
Produces:  prism_dq_recipe  (dict: tensor_class → GGUF quant type)

Why after OutputNode?
    The PRISM-DQ spectral metrics (SVD, eigenvalue analysis) are only
    meaningful on unquantized BF16/FP32 weights.  The model inside the
    pipeline is still a 4-bit BnB PeftModel during training — running
    metrics on that would give meaningless results.  After OutputNode
    saves the merged BF16 checkpoint, we load JUST the weights (cpu,
    no CUDA needed) in float32 for the analysis pass.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional

import torch

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)


def _load_weights_for_analysis(
    finale_dir: str,
    dtype: torch.dtype = torch.float32,
) -> Optional["torch.nn.Module"]:
    """
    Load the saved BF16 HF model from finale_dir in float32 on CPU.

    We load in float32 (not bfloat16) because torch.linalg.svdvals
    has higher numerical precision on float32 — important for the
    Hill estimator and eigenvalue analysis.

    Strips any BitsAndBytes quantization_config from the loaded config
    so that from_pretrained creates a plain float model, not a 4-bit one.

    Returns the model or None on failure.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig

        logger.info(
            "  PRISM-DQ: Loading BF16 weights from %s for analysis (cpu, float32) ...",
            finale_dir,
        )

        # Load the config and strip BnB quantization settings so that
        # from_pretrained creates a clean float model (not Linear4bit).
        config = AutoConfig.from_pretrained(finale_dir, trust_remote_code=True)
        for _attr in ("quantization_config", "_pre_quantization_dtype", "bitsandbytes"):
            try:
                config.__dict__.pop(_attr, None)
            except Exception:
                pass

        model = AutoModelForCausalLM.from_pretrained(
            finale_dir,
            config            = config,   # overrides config.json quant settings
            torch_dtype       = dtype,
            device_map        = "cpu",
            trust_remote_code = True,
            low_cpu_mem_usage = True,
            # Allow BnB metadata keys that may remain in the state dict to be
            # silently skipped rather than raising an error.
            ignore_mismatched_sizes = True,
        )
        model.eval()
        return model
    except Exception as exc:
        logger.error(
            "  PRISM-DQ: Failed to load model from %s: %s.",
            finale_dir, exc,
        )
        return None


class PRISMDQNode(BaseNode):
    NAME = "PRISMDQNode"
    INPUT_KEYS  = ("finale_dir", "args")
    OUTPUT_KEYS = ("prism_dq_recipe",)

    def should_run(self, ctx: NodeContext) -> bool:
        args = ctx.get("args")
        if args is None:
            return False
        if not getattr(args, "prism_dq", False):
            return False
        if not ctx.get("finale_dir"):
            logger.warning(
                "PRISMDQNode: --prism_dq is set but finale_dir is not in context. "
                "This usually means OutputNode did not run or --merge was not used. "
                "PRISM-DQ requires a merged BF16 checkpoint."
            )
            return False
        return True

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "finale_dir", "args")
        args       = ctx["args"]
        finale_dir = ctx["finale_dir"]

        target_bpw        = getattr(args, "target_bpw", 4.0)
        dq_refinement     = getattr(args, "dq_refinement", False)
        llama_path        = getattr(args, "dq_llama_path", None)
        input_gguf        = getattr(args, "dq_input_gguf", None)

        # ── Prefer in-memory merged model (avoids BnB disk-save artifacts) ──────
        # OutputNode caches `stats_model` in ctx["analysis_model"] when --prism_dq
        # is set.  That model is already dequantized in memory (proper float weights),
        # so we skip the disk reload entirely.  Pop it to free ctx memory after use.
        analysis_model = ctx.pop("analysis_model", None)

        if analysis_model is not None:
            logger.info(
                "  PRISM-DQ: Using in-memory merged model for analysis "
                "(skipping disk reload)."
            )
            analysis_model.eval()
        else:
            # ── Fallback: reload from finale_dir ──────────────────────────────────
            # Validate weights exist on disk first
            finale_has_weights = any(
                os.path.isfile(os.path.join(finale_dir, f))
                for f in ("pytorch_model.bin", "model.safetensors")
            ) or any(
                f.endswith(".safetensors")
                for f in os.listdir(finale_dir)
                if os.path.isfile(os.path.join(finale_dir, f))
            )

            if not finale_has_weights:
                raise NodeError(
                    f"PRISMDQNode: No weight files found in {finale_dir}. "
                    "PRISM-DQ requires a merged BF16/FP16 HF checkpoint. "
                    "Re-run with --merge (and --quantize bf16)."
                )

            analysis_model = _load_weights_for_analysis(finale_dir)
            if analysis_model is None:
                raise NodeError(
                    "PRISMDQNode: Could not load model from finale_dir. "
                    "The checkpoint may contain BnB-packed tensors. "
                    "Re-run the pipeline — the dequant fix in output_node.py "
                    "should produce a clean checkpoint on the next run."
                )


        # ── Run PRISM-DQ ──────────────────────────────────────────────────────
        try:
            from hmlcore.prism_dq import run_prism_dq
            recipe = run_prism_dq(
                model               = analysis_model,
                target_bpw          = target_bpw,
                dq_refinement       = dq_refinement,
                output_dir          = finale_dir,
                llama_quantize_path = llama_path,
                input_gguf          = input_gguf,
                base_quant          = "Q3_K",
            )
        except Exception as exc:
            raise NodeError(f"PRISM-DQ analysis failed: {exc}") from exc
        finally:
            # Free the analysis model — it's a separate copy and not needed
            del analysis_model
            gc.collect()

        ctx["prism_dq_recipe"] = recipe
        logger.info("✅ PRISM-DQ complete — recipe written to %s/prism_dq_recipe.sh", finale_dir)
