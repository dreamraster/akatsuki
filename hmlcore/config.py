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
REASONING_END = "</reasoning>"
SOLUTION_START = "<solution>"
SOLUTION_END = "</solution>"
SYSTEM_PROMPT = ""  # populated by apply_args() after CLI parsing
QWEN_JACK = False


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
    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="Student model path or HuggingFace ID",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--disable_unsloth", action="store_true", help="Disable Unsloth acceleration"
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated dataset paths / HF dataset IDs",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["math", "code", "general", "scene"],
        default="code",
    )
    parser.add_argument("--max_length", type=int, default=2048)

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Completions generated per prompt for GRPO",
    )
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument(
        "--disable_sft",
        action="store_true",
        help="Skip the SFT formatting warm-up stage",
    )
    parser.add_argument(
        "--force_grpo",
        action="store_true",
        help="Force GRPO RL to run for VLM models, bypassing the skip guard.",
    )
    parser.add_argument("--use_vllm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the last checkpoint"
    )
    parser.add_argument(
        "--qwen_jack",
        action="store_true",
        help="Enable dataset preparation and training configuration aligned with the Qwopus notebook "
        "(Qwen3-thinking templates, think-block normalization, and train_on_responses_only)",
    )

    # ── Save / merge ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapter into base model before saving",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="bf16",
        choices=[
            "bf16",  # HF format (no GGUF)
            "f16",
            "q8_0",
            "q6_k",  # high-quality GGUF
            "q5_k_m",
            "q5_k",
            "q4_k_m",
            "q4_k",  # medium GGUF
            "q3_k_m",
            "q2_k",  # low GGUF
            "iq4_xs",
            "iq3_xxs",  # imatrix GGUF
            "iq2_xxs",
            "iq2_xs",
            "iq2_s",  # aggressive imatrix
            "iq1_s",
            "iq1_m",  # extreme (pair with --dynamicquant)
        ],
        help=(
            "Output format when saving. "
            "'bf16' saves as HuggingFace checkpoint (default). "
            "All other values save as GGUF via Unsloth. "
            "IQ types (iq1_s, iq2_xxs …) pair well with --dynamicquant: "
            "pre-quantized layers compress near-losslessly; "
            "important layers retain quality."
        ),
    )
    # ── Pruning (REAP / Bonsai / DLP) ─────────────────────────────────────────
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Prune the model after training. Auto-routes: REAP for MoE architectures, "
        "Bonsai/DLP for dense transformers.",
    )
    parser.add_argument(
        "--prune_only",
        action="store_true",
        help="Skip SFT + GRPO; merge, prune, and save only.",
    )
    parser.add_argument(
        "--prune_ratio",
        type=float,
        default=None,
        help="Fraction of experts (MoE) or layers (dense) to remove, e.g. 0.3. "
        "Automatically enables --prune. Default: 0.5 when pruning is active.",
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=128,
        help="Calibration samples used to score layer/expert importance (default: 128).",
    )
    parser.add_argument(
        "--dynamicquant",
        action="store_true",
        help="Instead of removing low-scored layers/experts, quantize them to 1-bit in-place.",
    )
    # Advanced calibration knobs — hidden from help, still functional
    parser.add_argument(
        "--calibration_strategy",
        type=str,
        default="longest",
        choices=["longest", "shortest", "random", "first"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bonsai_noise", type=float, default=1e-4, help=argparse.SUPPRESS
    )
    parser.add_argument("--dlp_scale", type=float, default=1.0, help=argparse.SUPPRESS)

    # ── LLM Judge (code + general rewards) ───────────────────────────────────
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="lmStudio model name for LLM-as-judge scoring. "
        "Required to enable judge rewards for code/general domains.",
    )
    parser.add_argument(
        "--judge_url",
        type=str,
        default="http://localhost:1234",
        help="lmStudio base URL (default: http://localhost:1234)",
    )
    parser.add_argument(
        "--judge_timeout",
        type=int,
        default=60,
        help="HTTP timeout per judge call in seconds (default: 60).",
    )
    parser.add_argument(
        "--judge_cache_size", type=int, default=2048, help=argparse.SUPPRESS
    )
    parser.add_argument("--disable_judge", action="store_true", help=argparse.SUPPRESS)

    # ── X-Token Distillation ────────────────────────────────────────────────
    xtoken_group = parser.add_argument_group("X-Token Distillation")
    xtoken_group.add_argument(
        "--xtoken_enabled",
        action="store_true",
        help="Enable X-Token projection-guided cross-tokenizer distillation",
    )
    xtoken_group.add_argument(
        "--teacher_model",
        type=str,
        default=None,
        help="Teacher model path or HuggingFace ID",
    )
    xtoken_group.add_argument(
        "--xtoken_projection",
        type=str,
        default="mlp",
        choices=["linear", "mlp", "identity"],
        help="Projection type ('linear', 'mlp', 'identity')",
    )
    xtoken_group.add_argument(
        "--xtoken_hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension for MLP projection",
    )
    xtoken_group.add_argument(
        "--xtoken_epochs",
        type=int,
        default=3,
        help="Number of distillation training epochs",
    )
    xtoken_group.add_argument(
        "--xtoken_save_steps",
        type=int,
        default=500,
        help="Checkpoint save interval",
    )

    # ── PRISM Data Selection ────────────────────────────────────────────────
    prism_group = parser.add_argument_group("PRISM")
    prism_group.add_argument(
        "--prism_select",
        action="store_true",
        help="Enable PRISM data selection before training.",
    )
    prism_group.add_argument(
        "--prism_tier",
        type=str,
        default="high",
        choices=["high", "mid", "low", "high+mid"],
        help="Quality tier to keep (default: 'high').",
    )
    prism_group.add_argument(
        "--prism_only",
        action="store_true",
        help="Run PRISM selection and save the result, then exit.",
    )
    # Internal PRISM tuning knobs — hidden from help
    prism_group.add_argument(
        "--prism_layer", type=int, default=-1, help=argparse.SUPPRESS
    )
    prism_group.add_argument(
        "--prism_batch", type=int, default=16, help=argparse.SUPPRESS
    )
    prism_group.add_argument(
        "--prism_chunk", type=int, default=2000, help=argparse.SUPPRESS
    )
    prism_group.add_argument(
        "--prism_cache", type=str, default=None, help=argparse.SUPPRESS
    )

    # ── Dataset Column Mapping ──────────────────────────────────────────────
    data_map_group = parser.add_argument_group("Dataset Column Mapping")
    data_map_group.add_argument(
        "--instruction_cols",
        type=str,
        default="instruction,prompt,question",
        help=argparse.SUPPRESS,
    )
    data_map_group.add_argument(
        "--response_cols",
        type=str,
        default="response,output,answer,solution",
        help=argparse.SUPPRESS,
    )

    # ── PRISM Dynamic Quantization ──────────────────────────────────────────
    dq_group = parser.add_argument_group("PRISM-DQ")
    dq_group.add_argument(
        "--prism_dq",
        action="store_true",
        help=(
            "Enable PRISM Dynamic Quantization after the model is saved. "
            "Analyzes each weight tensor with 7 structural metrics "
            "(PL-Alpha-Hill, Spectral Dominance, OSQE, Matrix Imbalance, "
            "Fragility, Boundary Density, Spectral Position Prior) and runs "
            "a Lagrangian allocator to assign per-tensor-class GGUF quant "
            "types that meet the --target_bpw budget. "
            "Requires --merge to produce a BF16 checkpoint for analysis. "
            "Emits a ready-to-run llama-quantize recipe to finale_dir/prism_dq_recipe.sh."
        ),
    )
    dq_group.add_argument(
        "--target_bpw",
        type=float,
        default=4.0,
        metavar="BPW",
        help=(
            "Target average bits-per-weight for PRISM-DQ allocation. "
            "Lower values = more aggressive quantization. "
            "Typical values: 2.5 (aggressive), 3.5 (balanced), 4.5 (high quality). "
            "Default: 4.0 (≈ Q4_K_M equivalent)"
        ),
    )
    dq_group.add_argument(
        "--dq_refinement",
        action="store_true",
        help=(
            "Enable per-block refinement pass in PRISM-DQ. "
            "Identifies individual transformer blocks whose quantization error "
            "significantly exceeds the class mean and bumps them up one quality level. "
            "Produces a more precise recipe at a small compute overhead."
        ),
    )
    dq_group.add_argument(
        "--dq_llama_path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the llama-quantize binary (from llama.cpp). "
            "If provided and --dq_input_gguf is set, PRISM-DQ will "
            "auto-invoke llama-quantize after generating the recipe. "
            "Example: D:/llama.cpp/build/bin/llama-quantize.exe"
        ),
    )
    dq_group.add_argument(
        "--dq_input_gguf",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Path to an F16 GGUF to quantize with the PRISM-DQ recipe. "
            "Required when --dq_llama_path is set for auto-invocation. "
            "Generate an F16 GGUF first with: "
            "python convert_hf_to_gguf.py <finale_dir> --outtype f16"
        ),
    )

    # ── Shortcut Heads (Qwen3-style) ─────────────────────────────────────────
    sc_group = parser.add_argument_group("Shortcut Heads")
    sc_group.add_argument(
        "--shortcut_heads",
        action="store_true",
        help="Enable Qwen3-style shortcut heads: shallow transformer blocks "
        "on intermediate layers that predict tokens further ahead (t+K). "
        "Auxiliary loss improves internal representations during training.",
    )
    sc_group.add_argument(
        "--shortcut_layers",
        type=str,
        default="-3,-2",
        help="Comma-separated decoder layer indices for shortcut heads. "
        "Negative = from end. Default: -3,-2",
    )
    sc_group.add_argument(
        "--shortcut_offsets",
        type=str,
        default="2,3",
        help="Comma-separated token offsets each shortcut predicts ahead. "
        "Must match --shortcut_layers count. Default: 2,3",
    )
    sc_group.add_argument(
        "--shortcut_depth",
        type=int,
        default=2,
        help="Number of shallow transformer layers per shortcut head. Default: 2",
    )
    sc_group.add_argument(
        "--shortcut_weight",
        type=float,
        default=0.1,
        help="Loss weight (lambda) for shortcut auxiliary loss. Default: 0.1",
    )
    sc_group.add_argument(
        "--shortcut_heads_freeze",
        action="store_true",
        help="Freeze shortcut heads after SFT warm-up (default behaviour). "
        "Pass --no_shortcut_freeze to keep them trainable during GRPO.",
    )
    sc_group.add_argument(
        "--no_shortcut_freeze",
        action="store_true",
        help="Keep shortcut heads trainable during GRPO (overrides default freeze).",
    )

    # ── Training Configuration ──────────────────────────────────────────────
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps for GRPO (default: 4).",
    )
    train_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0).",
    )
    train_group.add_argument(
        "--lora_scale",
        action="store_true",
        help="Auto-scale LoRA rank based on model size (64 for 7B-13B, 128 for 70B+). "
        "Requires --lora_rank to be set.",
    )
    train_group.add_argument(
        "--merge_max_seq_length",
        type=int,
        default=8192,
        help="max_seq_length used during merge (default: 8192). "
        "Increase for models with large RoPE sizes (e.g. Llama-405B).",
    )
    train_group.add_argument(
        "--sft_warmup_ratio",
        type=float,
        default=0.05,
        help="SFT warmup fraction of total steps (default: 0.05 = 5%%).",
    )
    train_group.add_argument(
        "--grpo_warmup_ratio",
        type=float,
        default=0.1,
        help="GRPO warmup fraction of total steps (default: 0.1 = 10%%). "
        "GRPO is sensitive to warmup due to KL divergence scaling.",
    )

    # ── Reasoning Tags ──────────────────────────────────────────────────────
    tag_group = parser.add_argument_group("Reasoning Tags")
    tag_group.add_argument(
        "--r_start", type=str, default="<reasoning>", help=argparse.SUPPRESS
    )
    tag_group.add_argument(
        "--r_end", type=str, default="</reasoning>", help=argparse.SUPPRESS
    )
    tag_group.add_argument(
        "--s_start", type=str, default="<solution>", help=argparse.SUPPRESS
    )
    tag_group.add_argument(
        "--s_end", type=str, default="</solution>", help=argparse.SUPPRESS
    )
    tag_group.add_argument(
        "--system_prompt", type=str, default=None, help=argparse.SUPPRESS
    )

    return parser


# ── Global state ─────────────────────────────────────────────────────────────
args: argparse.Namespace = None  # type: ignore


def apply_args(args_in: argparse.Namespace) -> None:
    """Write parsed CLI values back into this module's globals so every other
    module that does ``from hmlcore.config import REASONING_START`` etc. gets the
    user-overridden values after this call."""
    global \
        REASONING_START, \
        REASONING_END, \
        SOLUTION_START, \
        SOLUTION_END, \
        SYSTEM_PROMPT, \
        QWEN_JACK, \
        args
    args = args_in
    import logging as _log

    QWEN_JACK = getattr(args, "qwen_jack", False)

    REASONING_START = args.r_start
    REASONING_END = args.r_end
    SOLUTION_START = args.s_start
    SOLUTION_END = args.s_end

    # If --qwen_jack is enabled, override default tags to match the notebook's <think> style
    # unless the user explicitly provided custom tags on the CLI.
    if getattr(args, "qwen_jack", False):
        if args.r_start == "<reasoning>":
            REASONING_START = "<think>"
        if args.r_end == "</reasoning>":
            REASONING_END = "</think>\n"
        if args.s_start == "<solution>":
            SOLUTION_START = ""
        if args.s_end == "</solution>":
            SOLUTION_END = ""

    # --prune_only implies --prune
    if getattr(args, "prune_only", False):
        args.prune = True

    # --prune_ratio auto-enables --prune
    if getattr(args, "prune_ratio", None) is not None and not getattr(
        args, "prune", False
    ):
        _log.getLogger(__name__).info(
            "ℹ️  --prune_ratio %.2f detected → enabling --prune automatically.",
            args.prune_ratio,
        )
        args.prune = True

    # Default prune_ratio to 0.5 when --prune is active but ratio was not set
    if getattr(args, "prune", False) and getattr(args, "prune_ratio", None) is None:
        args.prune_ratio = 0.5

    # --dynamicquant requires --prune
    if getattr(args, "dynamicquant", False) and not getattr(args, "prune", False):
        import sys

        _log.getLogger(__name__).error(
            "❌ --dynamicquant requires pruning to be active. "
            "Add --prune, --prune_only, or --prune_ratio <float>."
        )
        sys.exit(1)

    # ── LoRA rank auto-scaling ─────────────────────────────────────────────
    if getattr(args, "lora_scale", False) and getattr(args, "lora_rank", 32) > 0:
        num_params = getattr(args, "_model_num_params", 7000000000)  # default 7B
        if num_params > 30e9:
            args.lora_rank = 128
        elif num_params > 5e9:
            args.lora_rank = 64
        _log.getLogger(__name__).info(
            "🔧 LoRA rank scaled to %d for model ~%.1fB params.",
            args.lora_rank,
            num_params / 1e9,
        )

    # ── Merge max_seq_length ──────────────────────────────────────────────
    if getattr(args, "merge_max_seq_length", 8192) < 4096:
        _log.getLogger(__name__).warning(
            "⚠️  merge_max_seq_length=%d is low for large models. "
            "Consider increasing to 4096+ for models with large RoPE.",
            args.merge_max_seq_length,
        )

    # --prism_only checks
    if getattr(args, "prism_only", False):
        if not getattr(args, "prism_select", False):
            _log.getLogger(__name__).info(
                "ℹ️ --prism_only detected → enabling --prism_select automatically."
            )
        args.prism_select = True

    # --prism_dq validation
    if getattr(args, "prism_dq", False):
        if not getattr(args, "merge", False):
            _log.getLogger(__name__).warning(
                "⚠️  --prism_dq is set but --merge is not.  PRISM-DQ requires a merged "
                "BF16 checkpoint in finale_dir to analyze.  Add --merge to your command."
            )
        if getattr(args, "dq_llama_path", None) and not getattr(
            args, "dq_input_gguf", None
        ):
            _log.getLogger(__name__).info(
                "ℹ️  --dq_llama_path is set but --dq_input_gguf is not.  "
                "Auto-invocation of llama-quantize will be skipped.  "
                "The recipe script will be saved to finale_dir/prism_dq_recipe.sh."
            )

    if args.system_prompt:
        SYSTEM_PROMPT = args.system_prompt.format(
            r_start=REASONING_START,
            r_end=REASONING_END,
            s_start=SOLUTION_START,
            s_end=SOLUTION_END,
        )
    else:
        SYSTEM_PROMPT = get_system_prompt(
            REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
        )
