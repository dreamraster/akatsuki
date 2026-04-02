# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
Merge LoRA adapter weights into base model.

This script takes a trained LoRA checkpoint and merges it into the base Qwen3-0.6B model,
producing a standalone model that can be used for inference without PEFT library.

Usage:
    python merge_lora.py \
        --lora_model_name_or_path "./checkpoints/qwen3-0.6b-cuda-lora/final_checkpoint" \
        --output_dir "./merged_model"
"""

import argparse
import logging
import os

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def merge_lora_weights(
    lora_model_dir: str,
    output_dir: str,
    base_model_name_or_path: str = None,
    device_map: str = "auto",
    is_bfloat16: bool = True,
    new_model_name: str = None,
):
    """
    Merge LoRA adapter weights into the base model.

    Args:
        lora_model_dir: Path to directory containing LoRA checkpoint
        output_dir: Directory to save merged model
        base_model_name_or_path: Optional base model name/path. If None, reads from
            adapter_config.json or uses default "Qwen/Qwen3-0.6B"
        device_map: Device to load model on
        is_bfloat16: Use bfloat16 for inference

    Returns:
        Merged model saved to output_dir
    """
    logger.info(f"Loading LoRA model from {lora_model_dir}")

    # Check if adapter weights exist
    adapter_config_path = os.path.join(lora_model_dir, "adapter_config.json")
    adapter_weights_path = os.path.join(lora_model_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_config_path) and not os.path.exists(
        adapter_weights_path
    ):
        raise FileNotFoundError(
            f"Could not find LoRA adapter weights at {lora_model_dir}"
        )

    # Determine base model name from config or use default
    try:
        with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as f:
            import json

            adapter_config = json.load(f)
            if base_model_name_or_path is None:
                base_model_name_or_path = adapter_config.get(
                    "base_model_name_or_path", "Qwen/Qwen3-0.6B"
                )
    except Exception as e:
        if base_model_name_or_path is None:
            base_model_name_or_path = "Qwen/Qwen3-0.6B"
            logger.warning(
                f"Could not read adapter_config.json, using default: {base_model_name_or_path}"
            )

    logger.info(f"Base model: {base_model_name_or_path}")

    # Load base model
    logger.info("Loading base model...")
    torch_dtype = torch.bfloat16 if is_bfloat16 else torch.float16

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load base model {base_model_name_or_path}: {e}"
        )

    # Check if running on CPU and adjust device_map accordingly
    if device_map == "auto":
        if not torch.cuda.is_available():
            device_map = "cpu"
            logger.info("CUDA not available, using CPU")
        else:
            device_map = "cuda"

    # Load LoRA model (PEFT wrapper)
    logger.info(f"Loading PEFT model from {lora_model_dir}...")
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_model_dir,
    )

    # Merge LoRA weights into base model
    logger.info("Merging LoRA weights into base model...")
    try:
        merged_model = peft_model.merge_and_unload()
    except Exception as e:
        raise RuntimeError(f"Failed to merge LoRA weights: {e}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged model to {output_dir}...")

    # Save merged model
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    # Rename model in config if requested
    if new_model_name:
        try:
            import json
            config_path = os.path.join(output_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                config_data["_name_or_path"] = new_model_name
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"Set merged model name in config to: {new_model_name}")
        except Exception as e:
            logger.warning(f"Could not update new_model_name in config: {e}")

    # Copy tokenizer files
    tokenizer.save_pretrained(output_dir)

    # Copy chat template if it exists
    import shutil

    for filename in ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]:
        src_path = os.path.join(lora_model_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    logger.info(f"Model merged successfully! Saved to: {output_dir}")
    return output_dir


def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    """Helper function to generate text from a model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Greedy search for deterministic comparison
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def test_models(
    base_model_path: str,
    lora_model_dir: str,
    merged_model_dir: str,
    prompt: str = "<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n",
):
    """
    Test and compare inference across Base, LoRA, and Merged models.
    """
    logger.info("=" * 50)
    logger.info("RUNNING MODEL COMPARISON TESTS")
    logger.info("=" * 50)
    logger.info(f"Test Prompt:\n{prompt}")
    
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    # 1. Test Base Model
    logger.info("\n--- 1. Testing Base Model ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        device_map=device_map, 
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    base_result = generate_text(base_model, tokenizer, prompt)
    logger.info(f"Base Model Output:\n{base_result}")

    # 2. Test LoRA Model
    logger.info("\n--- 2. Testing Base + LoRA Model ---")
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(base_model, lora_model_dir)
    lora_result = generate_text(lora_model, tokenizer, prompt)
    logger.info(f"LoRA Model Output:\n{lora_result}")

    # Free memory before loading the third model
    del lora_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Test Merged Model
    logger.info("\n--- 3. Testing Merged Standalone Model ---")
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, trust_remote_code=True)
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_dir, 
        device_map=device_map, 
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    merged_result = generate_text(merged_model, merged_tokenizer, prompt)
    logger.info(f"Merged Model Output:\n{merged_result}")

    logger.info("\n--- Test Summary ---")
    if lora_result.strip() == merged_result.strip():
        logger.info("✅ SUCCESS: LoRA model and Merged model outputs MATCH perfectly!")
    else:
        logger.warning("❌ WARNING: LoRA model and Merged model outputs DO NOT match!")
    
    if lora_result.strip() != base_result.strip():
        logger.info("✅ SUCCESS: LoRA altered the base model's behavior (outputs differ).")
    else:
        logger.warning("⚠️ NOTICE: LoRA output is identical to base output. The adapter might not have learned enough, or the test prompt naturally elicits the same response.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into base model"
    )
    parser.add_argument(
        "--lora_model_dir",
        type=str,
        default="./checkpoints/qwen3-0.6b-cuda-lora/final_checkpoint",
        help="Directory containing LoRA checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/qwen3-0.6b-cuda-lora/merged_model",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="Base model name/path. If None, reads from adapter_config.json or uses Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to load model on (default: auto)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 precision (use float32 instead)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inference tests comparing base, LoRA, and merged models after merging",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="<|im_start|>user\nProvide a Python snippet for memory-efficient data loading.<|im_end|>\n<|im_start|>assistant\n",
        help="Prompt to use for the inference test",
    )
    parser.add_argument(
        "--new_model_name",
        type=str,
        default=None,
        help="Custom name to assign to the merged model in its config registry (e.g. 'MyOrg/Qwen3-FinTuned')",
    )

    args = parser.parse_args()

    # Determine base model name from adapter config if not provided
    base_model_name = args.base_model_name_or_path
    if base_model_name is None:
        try:
            import json
            with open(os.path.join(args.lora_model_dir, "adapter_config.json"), "r") as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen3-0.6B")
        except Exception:
            base_model_name = "Qwen/Qwen3-0.6B"

    # Run merge
    merge_lora_weights(
        lora_model_dir=args.lora_model_dir,
        output_dir=args.output_dir,
        base_model_name_or_path=base_model_name,
        device_map=args.device_map,
        is_bfloat16=not args.no_bf16,
        new_model_name=args.new_model_name,
    )

    print("\n" + "=" * 50)
    print("Merge complete!")
    print(f"Merged model saved to: {args.output_dir}")
    print("=" * 50)

    # Run tests if requested
    if args.test:
        test_models(
            base_model_path=base_model_name,
            lora_model_dir=args.lora_model_dir,
            merged_model_dir=args.output_dir,
            prompt=args.test_prompt
        )


if __name__ == "__main__":
    main()
