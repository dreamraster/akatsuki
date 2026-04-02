# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
hmlcore/run.py  —  Entry point for GRPO-based distillation.

Usage:
    python -m hmlcore.run --student_model <model> --datasets <path> --domain math

    # With LLM judge (code / general):
    python -m hmlcore.run --student_model <model> --datasets <path> --domain code \\
        --judge_model "llama-3.1-8b-instruct"

    # Resume a previous run:
    python -m hmlcore.run ... --resume

    # Merge adapter into full model after training:
    python -m hmlcore.run ... --merge --merge_quantization bf16

    # Custom reasoning tags:
    python -m hmlcore.run ... --r_start "<think>" --r_end "</think>" \\
                            --s_start "<answer>" --s_end "</answer>"

    # Custom system prompt (use {r_start}/{r_end}/{s_start}/{s_end} as placeholders):
    python -m hmlcore.run ... --system_prompt "Solve the problem. Reasoning: {r_start}...{r_end} Answer: {s_start}...{s_end}"
"""

import os
import logging

import hmlcore.config as cfg
from hmlcore.config  import build_parser, apply_args
from hmlcore.model   import load_model_and_tokenizer, save_model
from hmlcore.data    import setup_chat_template, load_and_preprocess_dataset
from hmlcore.rewards import build_reward_functions
from hmlcore.trainer import (
    find_last_checkpoint, is_sft_complete, load_sft_adapter,
    run_sft, run_grpo,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Apply CLI values → update hmlcore.config globals (tags, system prompt)
    apply_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    sft_dir  = os.path.join(args.output_dir, "sft")
    grpo_dir = os.path.join(args.output_dir, "grpo")

    # ── Determine resume state ────────────────────────────────────────────────
    grpo_checkpoint = find_last_checkpoint(grpo_dir) if args.resume else None
    sft_done        = is_sft_complete(sft_dir)        if args.resume else False
    sft_checkpoint  = (find_last_checkpoint(sft_dir)
                       if (args.resume and not sft_done) else None)

    if grpo_checkpoint:
        logger.info(f"▶️  Resuming GRPO from: {grpo_checkpoint}")
        logger.info(    "    SFT will be skipped (weights already in GRPO checkpoint).")
    elif sft_done and not args.disable_sft:
        logger.info(f"▶️  SFT already complete — adapter will be loaded from {sft_dir}.")
    elif sft_checkpoint and not args.disable_sft:
        logger.info(f"▶️  Partial SFT found — resuming from {sft_checkpoint}.")
    elif args.resume:
        logger.info("▶️  --resume passed but no checkpoints found. Starting fresh.")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    model, tokenizer, use_unsloth = load_model_and_tokenizer(args)
    tokenizer = setup_chat_template(tokenizer)

    # ── Load + preprocess dataset ─────────────────────────────────────────────
    dataset = load_and_preprocess_dataset(
        paths      = args.datasets.split(","),
        tokenizer  = tokenizer,
        domain     = args.domain,
        max_length = args.max_length,
    )

    # ── Step 1: SFT warm-up ───────────────────────────────────────────────────
    if grpo_checkpoint:
        pass   # SFT weights are already baked into the GRPO checkpoint
    elif not args.disable_sft and not sft_done:
        run_sft(model, tokenizer, dataset, args, sft_dir, sft_checkpoint)
    elif sft_done and not args.disable_sft:
        load_sft_adapter(model, sft_dir)

    # ── Step 2: GRPO ──────────────────────────────────────────────────────────
    reward_funcs, judge = build_reward_functions(args, tokenizer)

    try:
        run_grpo(model, tokenizer, dataset, reward_funcs, args,
                 grpo_dir, grpo_checkpoint)
    finally:
        if judge is not None:
            logger.info(judge.cache_stats())
            judge.close()

    # ── Save final model ──────────────────────────────────────────────────────
    save_model(model, tokenizer, args, use_unsloth)


if __name__ == "__main__":
    main()
