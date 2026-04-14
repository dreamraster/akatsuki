# By dreamraster · dreaMSCend
"""
hmlcore/model.py
=============
Model and tokenizer loading (Unsloth or standard PEFT), plus final save/merge.

Public API
----------
load_model_and_tokenizer(args) -> (model, tokenizer, use_unsloth: bool)
save_model(model, tokenizer, args, use_unsloth)
"""

import os
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional Unsloth import — `import unsloth` must come first per Unsloth's requirement
try:
    import unsloth  # noqa: F401  (side-effect: patches torch/transformers)
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL()
    HAS_UNSLOTH = True
    logger.info("�� Unsloth available and patched.")
except ImportError:
    HAS_UNSLOTH = False
    logger.error("�� Unsloth unavailable.")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Patch Transformers Logging Bug ──────────────────────────────────────────
# Some versions of transformers have a bug in `warning_once` where passing a
# message and an extra arg (like FutureWarning) causes a TypeError during
# string formatting if the message has no %s placeholders.
import transformers.utils.logging as hf_logging

try:
    # Dynamically obtain the logger class type to be compatible across versions
    _logger_instance = hf_logging.get_logger("transformers")
    _LoggerClass = type(_logger_instance)
    _original_warning_once = getattr(_LoggerClass, "warning_once", None)

    if _original_warning_once:
        def _patched_warning_once(self, *args, **kwargs):
            # If we have a message and extra args, but no % placeholders in the message,
            # the extra args will cause a TypeError in logging's getMessage().
            if len(args) > 1 and isinstance(args[0], str) and "%" not in args[0]:
                args = (args[0],)
            return _original_warning_once(self, *args, **kwargs)

        _LoggerClass.warning_once = _patched_warning_once
except Exception:
    pass # Guard against unexpected internal changes in transformers
# ─────────────────────────────────────────────────────────────────────────────

def use_unsloth_backend() -> bool:
    """Return True if the Unsloth backend is available and patched."""
    return HAS_UNSLOTH

def load_model_and_tokenizer(args):
    """Load student model + tokenizer with LoRA applied.

    Returns:
        model       — PeftModel (Unsloth or standard PEFT)
        tokenizer   — AutoTokenizer with pad_token set
        use_unsloth — bool, whether Unsloth is active
    """
    use_unsloth = HAS_UNSLOTH and not args.disable_unsloth

    if use_unsloth:
        logger.info("�� Loading model with Unsloth.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name          = args.student_model,
            max_seq_length      = args.max_length,
            load_in_4bit        = True,
            fast_inference      = False,
            gpu_memory_utilization = 0.9,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r                       = args.lora_rank,
            lora_alpha              = args.lora_rank * 2,
            target_modules          = ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing = "unsloth",
        )
    else:
        logger.info("�� Loading model with standard Transformers + PEFT.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit            = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype  = (torch.bfloat16
                                       if torch.cuda.is_bf16_supported()
                                       else torch.float16),
            bnb_4bit_quant_type     = "nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.student_model)
        model     = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model      = prepare_model_for_kbit_training(model)
        lora_cfg   = LoraConfig(
            r            = args.lora_rank,
            lora_alpha   = args.lora_rank * 2,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            task_type    = "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Ensure pad token exists (required by many tokenisers e.g. Qwen)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, use_unsloth


def save_model(model, tokenizer, args, use_unsloth: bool):
    """Save the final model.  Behaviour depends on --merge and --merge_quantization.

    No merge (default):
        Saves LoRA adapter weights only. Small files; base model needed at inference.

    --merge (standard PEFT, Unsloth bf16/f16):
        Merges adapter into base weights and saves a full HuggingFace model.

    --merge with q8_0 / q4_k_m / q5_k_m (Unsloth only):
        Exports directly to GGUF for use with llama.cpp / Ollama.
    """
    final_output = os.path.join(args.output_dir, "final_specialized_student")

    if args.merge and use_unsloth:
        quant    = args.merge_quantization
        is_gguf  = quant in ("q8_0", "q4_k_m", "q5_k_m")
        if is_gguf:
            logger.info(f"�� Unsloth: merging + exporting GGUF ({quant}) → {final_output}")
            model.save_pretrained_gguf(final_output, tokenizer,
                                       quantization_method=quant)
            logger.info(f"✅ GGUF export complete ({quant}).")
        else:
            logger.info(f"�� Unsloth: merging adapter → {quant} HF model → {final_output}")
            model.save_pretrained_merged(final_output, tokenizer, save_method=quant)
            logger.info(f"✅ Unsloth merge complete ({quant}).")

    elif args.merge:
        logger.info("�� Merging LoRA adapter (standard PEFT, bf16) ...")
        model = model.merge_and_unload()
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)
        logger.info("✅ Merge complete.")

    else:
        logger.info("�� Saving LoRA adapter only.")
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)

    logger.info(f"�� Model saved → {final_output}")
