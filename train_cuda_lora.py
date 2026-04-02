# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
CUDA-based LoRA fine-tuning

This script provides GPU-accelerated training with LoRA (Low-Rank Adaptation)
for memory-efficient fine-tuning of Qwen3-0.6B.

Usage:
    # CUDA training (requires GPU with sufficient VRAM)
    python train_cuda_lora.py --model_name "Qwen/Qwen3-0.6B" --data_path "TeichAI/claude-4.5-opus-high-reasoning-250x"

    # With custom settings
    python train_cuda_lora.py ^
        --model_name "Qwen/Qwen3-0.6B" ^
        --data_path "TeichAI/claude-4.5-opus-high-reasoning-250x" ^
        --num_train_epochs 3 ^
        --per_device_train_batch_size 1 ^
        --gradient_accumulation_steps 4
"""

import argparse
import logging
import os
# Disable Habana plugins that might cause issues in some environments
os.environ['HABANA_VISIBLE_DEVICES'] = ''
os.environ['PT_HPU_LAZY_MODE'] = '0'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = '0'

import sys

# Configure logging first (before using logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Don't use CUDA for now - let's let the script decide
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

class InferenceCallback(TrainerCallback):
    """Custom callback to generate text at the end of each epoch."""
    def __init__(self, prompt, tokenizer):
        self.prompt = prompt
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        logger.info("\n" + "="*50)
        logger.info(f"Epoch {state.epoch} completed (Step {state.global_step}). Running quick evaluation...")
        
        # Prepare inputs
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
        
        # Generate with no_grad to save memory during training
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
        result = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        logger.info(f"Test Prompt: {self.prompt}")
        logger.info(f"Generated Response:\n{result}")
        logger.info("="*50 + "\n")



def tokenize_prompt(tokenizer, instruction, input_text=None, output_text=None, max_length=1024):
    """Tokenize a single training example."""
    # Build prompt
    if input_text:
        query = f"{instruction}\n{input_text}"
    else:
        query = instruction

    # Format with ChatML template (Qwen native style)
    prompt = (
        "<|im_start|>user\n"
        f"{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # For training, we include the output and the end token
    if output_text:
        full_response = f"{output_text}<|im_end|>"
    else:
        full_response = ""

    # Combine into final prompt
    final_prompt = f"{prompt}{full_response}"

    # Tokenize with truncation
    encoding = tokenizer(
        final_prompt,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False
    )

    input_ids = encoding["input_ids"]

    # Create labels - mask prompt tokens, keep only output tokens
    labels = [-100] * len(input_ids)

    # Only train on the response (assistant part)
    if output_text:
        # Find where the assistant response starts in the tokenized sequence
        prompt_token_ids = tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        prompt_length = len(prompt_token_ids)
        response_start = min(prompt_length, len(input_ids) - 1)

        # Copy actual input_ids to labels for response part
        for i in range(response_start, len(input_ids)):
            if input_ids[i] != tokenizer.pad_token_id:
                labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }


def preprocess_dataset(dataset, tokenizer, max_length=1024):
    """Process a dataset into training format.

    Supports both:
    1. Standard format: instruction/input/output columns
    2. Messages format: messages column (ShareGPT style)
    """
    def preprocess(example):
        # Check for messages format (ShareGPT style)
        if "messages" in example:
            messages = example.get("messages", [])
            # Convert messages to instruction/output format
            # Last message from assistant is what we want to generate
            instruction = ""
            output_text = ""

            system_msg = None
            user_content = ""
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        system_msg = content
                    elif role == "user":
                        user_content = content
                    elif role == "assistant":
                        output_text = content

            # Build instruction from system + user messages
            if system_msg:
                instruction = f"{system_msg}\n\n{user_content}"
            else:
                instruction = user_content

            return tokenize_prompt(tokenizer, instruction, None, output_text, max_length)

        # Fallback to standard format: instruction/input/output
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        return tokenize_prompt(tokenizer, instruction, input_text, output_text, max_length)

    try:
        # Handle HuggingFace datasets
        processed = dataset.map(
            preprocess,
            desc="Tokenizing dataset"
        )
        return processed
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


def find_available_cuda_device():
    """Check if CUDA is available and return device string."""
    if torch.cuda.is_available():
        logger.info(f"CUDA is available! Using GPU: {torch.cuda.get_device_name()}")
        return "cuda"
    else:
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"


def train_cuda_lora(
    model_name: str = "Qwen/Qwen3-0.6B",
    data_path: str = "TeichAI/claude-4.5-opus-high-reasoning-250x",
    output_dir: str = "./checkpoints/qwen3-0.6b-cuda-lora",
    num_train_epochs: int = 1,
    max_steps: int = -1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    load_in_4bit: bool = True,
    use_mixed_precision: bool = True
):
    """
    Train Qwen3-0.6B using LoRA on CUDA-compatible GPU.

    Args:
        model_name: Base model to fine-tune
        data_path: Dataset name or path (comma separated for multiple)
        output_dir: Directory to save checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate for optimizer
        lora_rank: LoRA rank (higher = more capacity, more VRAM)
        lora_alpha: LoRA alpha scaling factor
        load_in_4bit: Whether to use 4-bit quantization
        use_mixed_precision: Whether to use mixed precision training
    """
    logger.info("=" * 50)
    logger.info("GPU-based Qwen3-0.6B LoRA Fine-tuning")
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        logger.info(f"GPU available: True ({gpu_name})")
        # Explicit ROCm check
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            logger.info(f"ROCm / HIP version: {torch.version.hip}")
        else:
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        logger.warning("GPU not available - using CPU (slow!)")
    logger.info("=" * 50)

    # Check for Hugging Face authentication
    from huggingface_hub import login
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = os.environ.get("HUGGINGFACE_API_KEY")

    if token:
        logger.info("Authenticating with Hugging Face...")
        try:
            login(token, add_to_pip_compatible=True)
            logger.info("Authentication successful!")
        except Exception as e:
            logger.warning(f"Could not authenticate with Hugging Face: {e}")
            logger.warning("Continuing without authentication (may fail for private models)")

    # Load tokenizer
    logger.info("1/6 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.info("Make sure you have access to the Qwen3-0.6B model on Hugging Face")
        return None

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    logger.info("2/6 Loading dataset(s)...")
    data_paths = [p.strip() for p in data_path.split(',')]
    all_train_datasets = []
    
    for dp in data_paths:
        try:
            raw_dataset = load_dataset(dp, trust_remote_code=True)
            logger.info(f"Loaded dataset: {dp}")
            if isinstance(raw_dataset, dict):
                if not raw_dataset.get("train"):
                    logger.warning(f"Dataset {dp} missing 'train' split. Skipping.")
                    continue
                all_train_datasets.append(raw_dataset["train"])
            else:
                all_train_datasets.append(raw_dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset {dp}: {e}")
            
    if not all_train_datasets:
        logger.error("No valid datasets loaded. Exiting.")
        return None
        
    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets(all_train_datasets)

    logger.info(f"Training on {len(train_dataset)} examples")

    # Process dataset
    logger.info("3/6 Preprocessing dataset...")
    processed_dataset = preprocess_dataset(train_dataset, tokenizer)

    # Split into train and eval for evaluation & early stopping
    dataset_size = len(processed_dataset)
    split_idx = int(0.9 * dataset_size)  # 90/10 split
    processed_dataset = processed_dataset.shuffle(seed=42)
    train_dataset_processed = processed_dataset.select(range(split_idx))
    eval_dataset_processed = processed_dataset.select(range(split_idx, dataset_size))
    
    # Check if we should disable 4-bit (not supported on CPU and causes crashes)
    if not torch.cuda.is_available() and load_in_4bit:
        logger.warning("CUDA is not available. Disabling 4-bit quantization as it causes crashes on CPU.")
        load_in_4bit = False

    # Load model with 4-bit quantization for memory efficiency
    logger.info("4/6 Loading Qwen3-0.6B model...")
    try:
        # Use bfloat16 for modern GPUs (RTX series and newer)
        torch_dtype = torch.bfloat16 if use_mixed_precision else torch.float32

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                # Test explicit import to catch ROCm binary missing error early
                import bitsandbytes
            except Exception as e:
                logger.error(f"Could not load BitsAndBytes required for 4-bit quantization: {e}")
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    logger.warning("\n" + "!"*60)
                    logger.warning("ROCm DETECTED: Standard BitsAndBytes is missing ROCm wrappers!")
                    logger.warning("To fix this, you MUST run this script with the --no_4bit flag.")
                    logger.warning("Example: python train_cuda_lora.py --no_4bit")
                    logger.warning("!"*60 + "\n")
                return None
            
            bnb_config = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if use_mixed_precision else torch.float32,
                "bnb_4bit_quant_type": "nf4",
                "llm_int8_enable_fp32_cpu_offload": True,  # Allows partial RAM offloading without crashing
            }

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(**bnb_config),
                )
            except ValueError as e:
                logger.error(f"Quantization offload failed: {e}")
                logger.warning("Your GPU lacks the VRAM to load this model even in 4-bit. Try using --no_4bit along with deepspeed, or use a smaller base model.")
                return None
        else:
            # Full precision or mixed precision on GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto",
            )

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Apply LoRA
    logger.info("5/6 Applying LoRA...")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied successfully!")

    # Print trainable parameters
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"All parameters: {all_param:,}")
    logger.info(f"Trainable %: {100 * trainable_params / all_param:.4f}")
    logger.info(f"Model device: {model.device if hasattr(model, 'device') else 'N/A'}")

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_enable()

    # Training arguments (GPU-optimized)
    eval_save_steps = max(1, max_steps // 2) if max_steps > 0 else 50
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=learning_rate,
        max_grad_norm=0.3,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        seed=42,
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,  # Fix for Windows hanging

        dataloader_pin_memory=False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )

    # Data collator for causal language modeling - handles padding automatically
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Inference and Early Stopping callbacks
    eval_prompt = "<|im_start|>user\nProvide a Python snippet for memory-efficient data loading.<|im_end|>\n<|im_start|>assistant\n"
    inference_cb = InferenceCallback(eval_prompt, tokenizer)
    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=3)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        data_collator=data_collator,
        callbacks=[inference_cb, early_stopping_cb],
    )

    # Start training
    logger.info("=" * 50)
    device_name = "ROCm/HIP" if hasattr(torch.version, 'hip') and torch.version.hip is not None else "CUDA"
    logger.info(f"Starting {device_name} (GPU) LoRA fine-tuning...")
    logger.info(f"Using device: {trainer.args.device}")
    logger.info("=" * 50)

    trainer.train()

    # Save model
    final_output_dir = os.path.join(output_dir, "finale")
    trainer.save_model(final_output_dir)

    logger.info(f"Training complete. Model saved to {final_output_dir}")
    return trainer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GPU-based Qwen3-0.6B LoRA Fine-tuning"
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name or path")
    parser.add_argument("--data_path", type=str, default="TeichAI/claude-4.5-opus-high-reasoning-250x",
                        help="Dataset name or path")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=4,
                        help="Max training steps (overrides epochs if > 0)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")

    # GPU-specific settings
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision (requires Ampere+ GPU)")
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false", default=True,
                        help="Disable 4-bit quantization")

    args = parser.parse_args()

    # Set training precision based on arguments
    use_fp16 = args.fp16 or args.bf16

    train_cuda_lora(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir="./versions/qvenkify",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=use_fp16 or args.bf16,
        load_in_4bit=args.use_4bit
    )


if __name__ == "__main__":
    main()
