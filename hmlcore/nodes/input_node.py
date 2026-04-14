# By dreamraster · dreaMSCend
"""
hmlcore/nodes/input_node.py
=============================
InputNode — first node in every pipeline.

Responsibilities:
  • Load model + tokenizer (Unsloth or standard PEFT via hmlcore.model)
  • Detect multimodal (VLM) models
  • Install chat template (hmlcore.data.setup_chat_template)
  • Load + preprocess dataset (hmlcore.data.load_and_preprocess_dataset)
  • Resolve resume checkpoint paths
  • Populate stage directories in context

Produces: model, tokenizer, use_unsloth, is_multimodal, dataset,
          sft_dir, grpo_dir, sft_checkpoint, grpo_checkpoint
"""

from __future__ import annotations

import logging
import os

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext

logger = logging.getLogger(__name__)


class InputNode(BaseNode):
    NAME = "InputNode"
    INPUT_KEYS = ("args",)
    OUTPUT_KEYS = (
        "model", "tokenizer", "use_unsloth", "is_multimodal",
        "dataset",
        "sft_dir", "grpo_dir",
        "sft_checkpoint", "grpo_checkpoint",
    )

    def run(self, ctx: NodeContext) -> None:
        self._require(ctx, "args")
        args = ctx["args"]

        # ── Stage directories ─────────────────────────────────────────────────
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        sft_dir  = os.path.join(output_dir, "sft")
        grpo_dir = os.path.join(output_dir, "grpo")
        ctx["sft_dir"]  = sft_dir
        ctx["grpo_dir"] = grpo_dir

        # ── Resume checkpoints ────────────────────────────────────────────────
        from hmlcore.trainer import find_last_checkpoint, is_sft_complete

        grpo_checkpoint: str | None = None
        sft_checkpoint:  str | None = None

        if getattr(args, "resume", False):
            grpo_checkpoint = find_last_checkpoint(grpo_dir)
            if grpo_checkpoint:
                logger.info("▶️  Resuming GRPO from: %s", grpo_checkpoint)
                logger.info("    SFT will be skipped (weights in GRPO checkpoint).")
            elif is_sft_complete(sft_dir) and not getattr(args, "disable_sft", False):
                logger.info("▶️  SFT already complete — adapter will be loaded.")
            else:
                sft_checkpoint = find_last_checkpoint(sft_dir)
                if sft_checkpoint:
                    logger.info("▶️  Partial SFT — resuming from %s", sft_checkpoint)
                else:
                    logger.info("▶️  --resume set but no checkpoints found. Starting fresh.")

        ctx["sft_checkpoint"]  = sft_checkpoint
        ctx["grpo_checkpoint"] = grpo_checkpoint

        # ── Model + tokenizer ─────────────────────────────────────────────────
        from hmlcore.model import load_model_and_tokenizer
        logger.info("�� Loading model: %s", args.student_model)
        try:
            model, tokenizer, use_unsloth = load_model_and_tokenizer(args)
        except Exception as exc:
            raise NodeError(f"Failed to load model '{args.student_model}': {exc}") from exc

        ctx["model"]       = model
        ctx["tokenizer"]   = tokenizer
        ctx["use_unsloth"] = use_unsloth

        # ── Multimodal detection ──────────────────────────────────────────────
        _cls_name = type(model).__name__
        is_multimodal = (
            "ConditionalGeneration" in _cls_name
            or hasattr(getattr(model, "config", None), "vision_config")
        )
        if is_multimodal:
            logger.info("�� Multimodal model detected (%s).", _cls_name)
        ctx["is_multimodal"] = is_multimodal

        # ── Pre-flight pipeline compatibility report ───────────────────────────
        from hmlcore.nodes.pipeline_check import run_pipeline_check
        run_pipeline_check(model, tokenizer, args, is_multimodal)

        # ── Chat template ─────────────────────────────────────────────────────
        from hmlcore.data import setup_chat_template
        tokenizer = setup_chat_template(tokenizer)
        ctx["tokenizer"] = tokenizer

        # ── Dataset ───────────────────────────────────────────────────────────
        paths = [p.strip() for p in args.datasets.split(",")]
        logger.info("�� Loading datasets: %s", paths)
        try:
            from hmlcore.data import load_and_preprocess_dataset
            dataset = load_and_preprocess_dataset(
                paths      = paths,
                tokenizer  = tokenizer,
                domain     = args.domain,
                max_length = args.max_length,
            )
        except Exception as exc:
            raise NodeError(f"Dataset loading failed: {exc}") from exc

        if len(dataset) == 0:
            raise NodeError(
                "Dataset is empty after preprocessing. "
                "Check --datasets path and --domain argument."
            )

        logger.info("✅ Dataset ready: %d examples.", len(dataset))

        # ── PRISM Data Selection ──────────────────────────────────────────────
        if getattr(args, "prism_select", False):
            from hmlcore.prism_selector import select_with_prism
            
            logger.info("�� PRISM: Starting data selection (curating diverse samples) ...")
            
            cache_path = args.prism_cache
            if cache_path is None:
                cache_path = os.path.join(args.output_dir, "prism_cache.pt")
                
            orig_size = len(dataset)
            dataset = select_with_prism(
                dataset    = dataset,
                model      = model,
                tokenizer  = tokenizer,
                tier       = args.prism_tier,
                layer      = args.prism_layer,
                batch_size = args.prism_batch,
                cache_path = cache_path,
                chunk_size = args.prism_chunk,
            )
            logger.info("✅ PRISM: Selection complete. Final dataset: %d samples (pruned from %d).", len(dataset), orig_size)
            
            if getattr(args, "prism_only", False):
                output_path = os.path.join(args.output_dir, f"prism_selected_{args.prism_tier}.jsonl")
                dataset.to_json(output_path, orient="records", lines=True)
                logger.info("⚡ PRISM_ONLY: Selected dataset saved to %s", output_path)
                import sys
                sys.exit(0) # Exit cleanly

        ctx["dataset"] = dataset
