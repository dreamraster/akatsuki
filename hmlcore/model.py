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

import importlib.util
import os
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Detect Unsloth availability WITHOUT importing it yet.
# Importing unsloth at module level monkey-patches the transformers model
# registry globally, replacing model classes with Unsloth-fused versions
# that require CUDA kernels set up by FastLanguageModel.from_pretrained.
# If --disable_unsloth is used, AutoModelForCausalLM.from_pretrained would
# still return the patched class → AttributeError on apply_qkv.
# Solution: only import + activate unsloth inside load_model_and_tokenizer
# when it is actually going to be used.
HAS_UNSLOTH    = importlib.util.find_spec("unsloth") is not None
_UNSLOTH_ACTIVE = False  # True only after we actually import + patch

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
    """Return True if Unsloth was actually imported and activated for this run.

    Distinct from HAS_UNSLOTH (installed) — this is only True after
    load_model_and_tokenizer ran with use_unsloth=True.
    """
    return _UNSLOTH_ACTIVE

def load_model_and_tokenizer(args):
    """Load student model + tokenizer with LoRA applied.

    Returns:
        model       — PeftModel (Unsloth or standard PEFT)
        tokenizer   — AutoTokenizer with pad_token set
        use_unsloth — bool, whether Unsloth is active
    """
    use_unsloth = HAS_UNSLOTH and not args.disable_unsloth

    if use_unsloth:
        global _UNSLOTH_ACTIVE
        # Import unsloth HERE (not at module level) so that the transformers
        # registry patch only fires when we're actually using Unsloth.
        # PatchFastRL() must also be called after the import.
        import unsloth  # noqa: F401  (side-effect: patches torch/transformers)
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL()
        _UNSLOTH_ACTIVE = True
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
        
        # Check if the model is a VLM by inspecting config
        is_vlm = False
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.student_model, trust_remote_code=True)
            arch = getattr(config, "architectures", [""])[0]
            is_vlm = (
                "ConditionalGeneration" in arch
                or "Vision" in arch
                or hasattr(config, "vision_config")
                or getattr(config, "model_type", "") in {"qwen2_vl", "llava", "mplug_owl2", "fuyu", "idefics", "idefics2", "paligemma", "pix2struct"}
            )
        except Exception:
            pass

        if is_vlm:
            logger.info("�� Detected VLM config. Loading processor & VLM model...")
            from transformers import AutoProcessor
            try:
                tokenizer = AutoProcessor.from_pretrained(args.student_model, trust_remote_code=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
            
            loaded = False
            for cls_name in ["AutoModelForVision2Seq", "AutoModelForConditionalGeneration", "AutoModel"]:
                try:
                    import transformers
                    model_cls = getattr(transformers, cls_name, None)
                    if model_cls is not None:
                        model = model_cls.from_pretrained(
                            args.student_model,
                            quantization_config = bnb_config,
                            device_map          = "auto",
                            trust_remote_code   = True,
                        )
                        loaded = True
                        break
                except Exception as e:
                    logger.warning(f"Failed to load VLM model with {cls_name}: {e}")
            if not loaded:
                raise RuntimeError(f"Could not load VLM model {args.student_model} with any AutoModel classes.")
        else:
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
            task_type    = None if is_vlm else "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Ensure pad token exists (required by many tokenisers e.g. Qwen)
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(actual_tokenizer, "pad_token", None) is None:
        actual_tokenizer.pad_token    = actual_tokenizer.eos_token
        actual_tokenizer.pad_token_id = actual_tokenizer.eos_token_id

    return model, tokenizer, use_unsloth


def save_model(model, tokenizer, args, use_unsloth: bool):
    """Save the final model.  Behaviour depends on --merge and --quantize.

    No merge (default):
        Saves LoRA adapter weights only. Small files; base model needed at inference.

    --merge (standard PEFT, Unsloth bf16/f16):
        Merges adapter into base weights and saves a full HuggingFace model.

    --merge with GGUF quant formats (Unsloth only):
        Exports directly to GGUF for use with llama.cpp / Ollama.
    """
    final_output = os.path.join(args.output_dir, "final_specialized_student")

    if args.merge and use_unsloth:
        quant    = getattr(args, "quantize", "bf16")
        # Define all GGUF quantization formats supported by Unsloth in config.py
        gguf_quants = {
            "f16", "q8_0", "q6_k", "q5_k_m", "q5_k", "q4_k_m", "q4_k",
            "q3_k_m", "q2_k", "iq4_xs", "iq3_xxs", "iq2_xxs", "iq2_xs",
            "iq2_s", "iq1_s", "iq1_m"
        }
        is_gguf  = quant in gguf_quants
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
