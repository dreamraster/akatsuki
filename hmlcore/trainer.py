# By dreamraster · dreaMSCend
"""
hmlcore/trainer.py
===============
SFT warm-up stage and GRPO training stage, with full resume support.

Public API
----------
run_sft(model, tokenizer, dataset, args, sft_dir, sft_checkpoint)
run_grpo(model, tokenizer, dataset, reward_funcs, args, grpo_dir, grpo_checkpoint)
find_last_checkpoint(directory) -> str | None
is_sft_complete(sft_dir) -> bool
load_sft_adapter(model, sft_dir)
"""

import os
import glob
import logging
from contextlib import contextmanager

from transformers.trainer_utils import get_last_checkpoint
# from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig (moved inside functions)

import hmlcore.config as cfg

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resume helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_last_checkpoint(directory: str) -> str | None:
    """Return the most recent HF Trainer checkpoint path, or None.

    Uses HF's get_last_checkpoint() first (requires trainer_state.json).
    Falls back to a manual glob scan so checkpoints written by older TRL
    versions without trainer_state.json are still found.
    """
    if not os.path.isdir(directory):
        return None

    ckpt = get_last_checkpoint(directory)
    if ckpt:
        return ckpt

    # Manual fallback: find highest-numbered checkpoint-<int> dir
    candidates = sorted(
        glob.glob(os.path.join(directory, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
    )
    if candidates:
        logger.warning(
            f"get_last_checkpoint() found nothing in {directory} but manual scan "
            f"found {len(candidates)} checkpoint(s). Using: {candidates[-1]}"
        )
        return candidates[-1]

    return None


def is_sft_complete(sft_dir: str) -> bool:
    """True when SFT has fully completed and the adapter has been saved.

    We write a zero-byte sentinel file 'sft_complete' immediately after
    save_model() so we can distinguish a completed run from a crashed one
    that only left partial checkpoints.
    """
    return os.path.isfile(os.path.join(sft_dir, "sft_complete"))


def load_sft_adapter(model, sft_dir: str):
    """Load a previously saved SFT LoRA adapter into an existing PeftModel."""
    from peft import set_peft_model_state_dict, load_peft_weights
    logger.info(f"Loading SFT adapter from {sft_dir}")
    weights = load_peft_weights(sft_dir)
    set_peft_model_state_dict(model, weights)
    logger.info("✅ SFT adapter weights loaded.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SFT stage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_sft(model, tokenizer, dataset, args, sft_dir: str,
            sft_checkpoint: str | None):
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only
    """Run the SFT formatting warm-up on a small subset of the dataset.

    Builds a 'text' column via apply_chat_template and trains SFTTrainer.
    Saves the adapter + writes the sft_complete sentinel when done.

    If 'sft_complete' already exists in sft_dir, it loads the adapter and skips training.
    """
    if is_sft_complete(sft_dir):
        logger.info("✅ SFT already complete — loading adapter from %s", sft_dir)
        load_sft_adapter(model, sft_dir)
        return

    sft_dataset = dataset.select(range(min(len(dataset), 100)))

    if len(sft_dataset) == 0:
        logger.warning("SFT dataset slice is empty — nothing to train on.")
        return

    def tokenize_sft(x):
        # If --qwen_jack is active, use the raw response (already normalized in data.py)
        if cfg.QWEN_JACK:
            target = x.get("full_response", "")
        else:
            resp = x.get("full_response", "")
            if cfg.REASONING_START in str(resp) and cfg.SOLUTION_START in str(resp):
                target = resp
            elif "<think>" in str(resp) and "</think>" in str(resp):
                thought = str(resp).split("<think>")[1].split("</think>")[0]
                tail    = str(resp).split("</think>")[1].strip()
                target  = (
                    f"{cfg.REASONING_START}{thought}{cfg.REASONING_END}"
                    f"{cfg.SOLUTION_START}{tail}{cfg.SOLUTION_END}"
                )
            else:
                target = (
                    f"{cfg.REASONING_START}Thinking...{cfg.REASONING_END}"
                    f"{cfg.SOLUTION_START}{x['completion']}{cfg.SOLUTION_END}"
                )

            # The chat template prepends REASONING_START to every assistant turn —
            # strip it here to avoid doubling.
            if target.startswith(cfg.REASONING_START):
                target = target[len(cfg.REASONING_START):]

        # Normalise raw_messages: HF Arrow may return dict-of-lists
        raw = x["raw_messages"]
        if isinstance(raw, dict):
            raw_msgs = [
                {"role": r, "content": c}
                for r, c in zip(raw["role"], raw["content"])
            ]
        else:
            raw_msgs = list(raw)

        messages = raw_msgs + [{"role": "assistant", "content": target}]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Sanity-check: full text must start with the rendered prompt
        prompt_only = tokenizer.apply_chat_template(
            raw_msgs, tokenize=False, add_generation_prompt=True
        )
        if not text.startswith(prompt_only):
            logger.warning(
                "SFT text does not start with rendered prompt — check chat template.\n"
                "  prompt : %r\n  text   : %r",
                prompt_only[:120],
                text[:120],
            )

        return {"text": text}

    # load_from_cache_file=False prevents stale tokenisations when tags change
    sft_dataset = sft_dataset.map(tokenize_sft, load_from_cache_file=False)

    # Verify the "text" column was produced
    if "text" not in sft_dataset.column_names:
        # Surface the first tokenization error directly
        try:
            tokenize_sft(sft_dataset[0])
        except Exception as exc:
            raise ValueError(f"tokenize_sft() failed on first example: {exc}") from exc
        raise ValueError(
            f"'text' column missing after map(tokenize_sft). "
            f"Got columns: {sft_dataset.column_names}"
        )

    sft_dataset = sft_dataset.select_columns(["text"])

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("�� Step 1: SFT warm-up (%d examples) ...", len(sft_dataset))

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = sft_dataset,
        args             = SFTConfig(
            output_dir                  = sft_dir,
            dataset_text_field          = "text",
            per_device_train_batch_size = args.batch_size,
            learning_rate               = 2e-4,
            num_train_epochs            = 1,
            max_steps                   = getattr(args, "max_steps", -1),
            logging_steps               = 10,
            report_to                   = "none",
        ),
    )

    if sft_checkpoint:
        logger.info("▶️  Resuming SFT from %s", sft_checkpoint)

    if cfg.QWEN_JACK:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n<think>",
        )

    if cfg.QWEN_JACK:
        logger.info("Tokenization (QWEN_JACK):")
        logger.info(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
    # trainer.train(resume_from_checkpoint=sft_checkpoint)
    # trainer.save_model(sft_dir)

    # model.save_pretrained(sft_dir)  # Local saving
    # tokenizer.save_pretrained(sft_dir)

    trainer.train(resume_from_checkpoint=sft_checkpoint)
    trainer.save_model(sft_dir)

    # Write completion sentinel
    open(os.path.join(sft_dir, "sft_complete"), "w").close()
    logger.info("✅ SFT complete. Adapter → %s", sft_dir)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRPO stage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextmanager
def _grpo_config_compat():
    from trl import GRPOConfig
    """Patch GRPOConfig.__post_init__ to be idempotent during GRPOTrainer.__init__.

    In TRL ≥ 0.14, GRPOConfig.__post_init__ raises ValueError when both
    generation_batch_size and steps_per_generation are non-None on entry.
    prepare_peft_model() inside GRPOTrainer.__init__ calls
    dataclasses.replace(args, gradient_checkpointing=False), which copies the
    already-computed field values (both non-None after the first __post_init__)
    and re-runs __post_init__ — triggering the error.

    This context manager patches __post_init__ only for the duration of
    GRPOTrainer.__init__: when both fields arrive non-None (the dataclasses.replace
    re-run case), steps_per_generation is cleared first so the original logic can
    recompute it cleanly from generation_batch_size without raising.

    The patch is scoped to GRPOTrainer.__init__ only; training proceeds with
    a fully-computed steps_per_generation on trainer.args.
    """
    if not hasattr(GRPOConfig, "steps_per_generation"):
        yield
        return

    _orig = GRPOConfig.__post_init__

    def _compat_post_init(self):
        # Both non-None = dataclasses.replace copied previously-computed values.
        # Clear steps_per_generation so the original logic recomputes it from
        # generation_batch_size without raising the mutual-exclusion error.
        if (getattr(self, "steps_per_generation", None) is not None
                and getattr(self, "generation_batch_size", None) is not None):
            try:
                self.steps_per_generation = None
            except Exception:
                object.__setattr__(self, "steps_per_generation", None)
        _orig(self)

    GRPOConfig.__post_init__ = _compat_post_init
    try:
        yield
    finally:
        GRPOConfig.__post_init__ = _orig


def run_grpo(model, tokenizer, dataset, reward_funcs: list, args,
             grpo_dir: str, grpo_checkpoint: str | None):
    from trl import GRPOConfig, GRPOTrainer
    """Run GRPO reinforcement learning.

    resume_from_checkpoint is set in three places (belt-and-suspenders) because
    different TRL versions honour different paths:
      1. GRPOConfig attribute set after construction
      2. trainer.args patched after GRPOTrainer.__init__
      3. Passed directly to trainer.train()
    """
    logger.info(f"�� Step 2: GRPO RL ({args.domain}) ...")

    # TRL ≥ 0.14 added steps_per_generation (default non-None in some builds).
    # prepare_peft_model() internally calls dataclasses.replace(args,
    # gradient_checkpointing=False), which re-runs __post_init__ with all
    # current field values as kwargs.  Two issues arise in newer TRL:
    #
    #   1. "both configured at the same time" — if both generation_batch_size
    #      and steps_per_generation end up non-None after __post_init__, the
    #      re-run raises ValueError.  Fix: _grpo_config_compat() patches
    #      __post_init__ to be idempotent during GRPOTrainer.__init__ only.
    #
    #   2. "generation_batch_size (N) must be divisible by num_generations (M)"
    #      — when steps_per_generation=None, TRL auto-computes generation_batch_size
    #      as per_device_train_batch_size (e.g. 1), which fails the divisibility
    #      constraint.  Fix: explicitly set generation_batch_size to
    #      num_generations * batch_size, which is always divisible.
    _extra: dict = {}
    if hasattr(GRPOConfig, "steps_per_generation"):
        _extra["steps_per_generation"]  = None
        _extra["generation_batch_size"] = args.num_generations * args.batch_size

    grpo_config = GRPOConfig(
        output_dir                  = grpo_dir,
        gradient_checkpointing      = True,
        learning_rate               = 5e-6,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 4,
        num_generations             = args.num_generations,
        max_prompt_length           = args.max_length // 4,
        max_completion_length       = args.max_length - (args.max_length // 4),
        max_steps                   = args.max_steps,
        logging_steps               = 1,
        save_steps                  = 50,
        save_total_limit            = 3,
        report_to                   = "none",
        **_extra,
    )
    # After __post_init__, steps_per_generation is auto-computed from
    # generation_batch_size — both fields are now non-None on grpo_config.
    # _grpo_config_compat() handles the dataclasses.replace conflict inside
    # GRPOTrainer.__init__ without touching grpo_config itself, so
    # trainer.args.steps_per_generation remains valid for get_train_dataloader.

    if grpo_checkpoint:
        grpo_config.resume_from_checkpoint = grpo_checkpoint
        logger.info(f"�� GRPO resume checkpoint: {grpo_checkpoint}")

    with _grpo_config_compat():
        trainer = GRPOTrainer(
            model            = model,
            reward_funcs     = reward_funcs,
            args             = grpo_config,
            train_dataset    = dataset,
            processing_class = tokenizer,
        )

    if grpo_checkpoint:
        trainer.args.resume_from_checkpoint = grpo_checkpoint

    trainer.train(resume_from_checkpoint=grpo_checkpoint)
    logger.info("✅ GRPO training complete.")
