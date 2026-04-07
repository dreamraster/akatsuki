# By dreamraster · dreaMSCend
"""
hmlcore/config.py
==============
Global tag constants, system-prompt builder, and CLI argument parser.
Everything else imports from here — no circular deps.
"""

import argparse

# ── Default reasoning/solution tags ──────────────────────────────────────────
REASONING_START = "<reasoning>"
REASONING_END   = "</reasoning>"
SOLUTION_START  = "<solution>"
SOLUTION_END    = "</solution>"
SYSTEM_PROMPT   = ""   # populated by apply_args() after CLI parsing
QWEN_JACK       = False


def get_system_prompt(r_start: str, r_end: str, s_start: str, s_end: str) -> str:
    return (
        "You are given a problem. "
        "Think about the problem and provide your working out. "
        f"Place it between {r_start} and {r_end}. "
        f"Then, provide your solution between {s_start} and {s_end}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO-based Distillation for Specialized Domains"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--student_model", type=str, required=True, help="Student model path or HuggingFace ID")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--disable_unsloth", action="store_true", help="Disable Unsloth acceleration")

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset paths / HF dataset IDs")
    parser.add_argument("--domain", type=str, choices=["math", "code", "general", "scene"], default="code")
    parser.add_argument("--max_length", type=int, default=2048)

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4, help="Completions generated per prompt for GRPO")
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--disable_sft", action="store_true", help="Skip the SFT formatting warm-up stage")
    parser.add_argument("--use_vllm", action="store_false", help="Enable vLLM for fast GRPO generation")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint")
    parser.add_argument("--qwen_jack", action="store_true",
                        help="Enable dataset preparation and training configuration aligned with the Qwopus notebook "
                             "(Qwen3-thinking templates, think-block normalization, and train_on_responses_only)")

    # ── Save / merge ─────────────────────────────────────────────────────────
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model before saving")
    parser.add_argument("--quantize", type=str, default="bf16",
                        choices=[
                            "bf16",                              # HF format (no GGUF)
                            "f16",  "q8_0",  "q6_k",            # high-quality GGUF
                            "q5_k_m", "q5_k", "q4_k_m", "q4_k",# medium GGUF
                            "q3_k_m", "q2_k",                   # low GGUF
                            "iq4_xs", "iq3_xxs",                 # imatrix GGUF
                            "iq2_xxs", "iq2_xs", "iq2_s",       # aggressive imatrix
                            "iq1_s",  "iq1_m",                  # extreme (pair with --dynamicquant)
                        ],
                        help=(
                            "Output format when saving. "
                            "'bf16' saves as HuggingFace checkpoint (default). "
                            "All other values save as GGUF via Unsloth. "
                            "IQ types (iq1_s, iq2_xxs …) pair well with --dynamicquant: "
                            "pre-quantized layers compress near-losslessly; "
                            "important layers retain quality."
                        ))
    # Legacy alias kept for backwards compatibility
    parser.add_argument("--merge_quantization", type=str, default=None,
                        choices=[
                            "bf16", "f16", "q8_0", "q6_k",
                            "q5_k_m", "q5_k", "q4_k_m", "q4_k",
                            "q3_k_m", "q2_k",
                            "iq4_xs", "iq3_xxs", "iq2_xxs", "iq2_xs", "iq2_s",
                            "iq1_s", "iq1_m",
                        ],
                        help=argparse.SUPPRESS)

    # ── REAP expert pruning ───────────────────────────────────────────────────
    parser.add_argument("--prune_experts", action="store_true",
                        help="Run REAP expert pruning after training (requires MoE model)")
    parser.add_argument("--prune_only", action="store_true",
                        help="Skip SFT + GRPO; run REAP pruning only")
    parser.add_argument("--prune_ratio", type=float, default=None,
                        help="Fraction of experts to remove per MoE layer (0.0–1.0). "
                             "Setting this automatically enables --prune_experts. "
                             "Default: 0.5 when pruning is active.")
    parser.add_argument("--calibration_samples", type=int, default=128,
                        help="Number of dataset examples used for REAP calibration (default: 128)")
    parser.add_argument("--calibration_strategy", type=str, default="longest",
                        choices=["longest", "shortest", "random", "first"],
                        help=(
                            "How to select calibration samples from the dataset. "
                            "'longest' (default) picks the longest examples first, "
                            "maximising hidden-state signal per sample for more reliable "
                            "pruning scores.  'shortest' favours diverse short-turn "
                            "coverage.  'random' shuffles with a fixed seed.  "
                            "'first' preserves natural dataset order."
                        ))
    parser.add_argument("--dynamicquant", action="store_true",
                        help=(
                            "Score-guided dynamic quantization: instead of removing "
                            "low-scored experts (REAP) or layers (ShortGPT), keep them "
                            "but apply 1-bit (binary) weight quantization to their "
                            "Linear layers in-place.  Inspired by Unsloth Dynamic 2.0 "
                            "GGUFs — important layers stay at full precision, redundant "
                            "ones are aggressively degraded.  Only valid when a pruning "
                            "option is active (--prune_experts / --prune_only / "
                            "--prune_ratio).  The resulting model has the same layer "
                            "count as the original; 1-bit weights compress to IQ1 "
                            "territory when exported to GGUF."
                        ))

    # ── LLM Judge (code + general rewards) ───────────────────────────────────
    parser.add_argument("--judge_model", type=str, default=None, help="lmStudio model name for LLM-as-judge scoring. " "Required to enable judge rewards for code/general domains.")
    parser.add_argument("--judge_url", type=str, default="http://localhost:1234", help="lmStudio base URL (default: http://localhost:1234)")
    parser.add_argument("--judge_timeout", type=int, default=60, help="HTTP timeout per judge call in seconds (default: 60)")
    parser.add_argument("--judge_cache_size", type=int, default=2048, help="Max cached (prompt, completion) judge scores (default: 2048)")
    parser.add_argument("--disable_judge", action="store_true", help="Disable LLM judge; fall back to heuristic rewards")

    # ── Prompt customisation ──────────────────────────────────────────────────
    parser.add_argument("--system_prompt", type=str, default=None, help="Override the default system prompt entirely (plain string). " "Use {r_start}/{r_end}/{s_start}/{s_end} as placeholders.")
    parser.add_argument("--r_start", type=str, default=REASONING_START, help="Reasoning-block open tag")
    parser.add_argument("--r_end",   type=str, default=REASONING_END, help="Reasoning-block close tag")
    parser.add_argument("--s_start", type=str, default=SOLUTION_START, help="Solution-block open tag")
    parser.add_argument("--s_end",   type=str, default=SOLUTION_END, help="Solution-block close tag")

    return parser


def apply_args(args: argparse.Namespace) -> None:
    """Write parsed CLI values back into this module's globals so every other
    module that does ``from hmlcore.config import REASONING_START`` etc. gets the
    user-overridden values after this call."""
    global REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END, SYSTEM_PROMPT, QWEN_JACK

    QWEN_JACK = getattr(args, "qwen_jack", False)

    REASONING_START = args.r_start
    REASONING_END   = args.r_end
    SOLUTION_START  = args.s_start
    SOLUTION_END    = args.s_end

    # If --qwen_jack is enabled, override default tags to match the notebook's <think> style
    # unless the user explicitly provided custom tags on the CLI.
    if getattr(args, "qwen_jack", False):
        if args.r_start == "<reasoning>": REASONING_START = "<think>"
        if args.r_end   == "</reasoning>": REASONING_END   = "</think>\n"
        if args.s_start == "<solution>":  SOLUTION_START  = ""
        if args.s_end   == "</solution>":  SOLUTION_END    = ""

    # Resolve legacy --merge_quantization alias → --quantize
    if getattr(args, "merge_quantization", None) is not None:
        args.quantize = args.merge_quantization

    # --prune_only implies --prune_experts
    if getattr(args, "prune_only", False):
        args.prune_experts = True

    # --prune_ratio N automatically enables pruning — no need to also pass --prune_experts
    if getattr(args, "prune_ratio", None) is not None:
        if not getattr(args, "prune_experts", False):
            import logging as _log
            _log.getLogger(__name__).info(
                "ℹ️  --prune_ratio %.2f detected → enabling --prune_experts automatically.",
                args.prune_ratio,
            )
        args.prune_experts = True

    # Default prune_ratio to 0.5 when pruning is active but ratio was not specified
    if getattr(args, "prune_experts", False) and getattr(args, "prune_ratio", None) is None:
        args.prune_ratio = 0.5

    # --dynamicquant is only meaningful when pruning is enabled — it redirects
    # the pruning action from "remove" to "quantize to 1-bit".  Error early so
    # the user doesn't run a full training pipeline only to find the flag was silently ignored.
    if getattr(args, "dynamicquant", False) and not getattr(args, "prune_experts", False):
        import sys
        _log.getLogger(__name__).error(
            "❌ --dynamicquant requires a pruning option to be active. "
            "Add --prune_experts, --prune_only, or --prune_ratio <float>."
        )
        sys.exit(1)

    if args.system_prompt:
        SYSTEM_PROMPT = args.system_prompt.format(
            r_start=REASONING_START, r_end=REASONING_END,
            s_start=SOLUTION_START,  s_end=SOLUTION_END,
        )
    else:
        SYSTEM_PROMPT = get_system_prompt(
            REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
        )
