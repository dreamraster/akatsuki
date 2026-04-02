# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
CPU-based LoRA fine-tuning

This script provides an alternative training approach that works without CUDA.
It's slower but more compatible across different system configurations.

Usage:
    # CPU-only training (works on any machine with enough RAM)
    python train_cpu_lora.py --model_name "Qwen/Qwen3-0.6B" --data_path "TeichAI/claude-4.5-opus-high-reasoning-250x"
"""

import argparse
import logging
import os
import sys

# Disable Habana plugins that might cause issues in some environments
os.environ['HABANA_VISIBLE_DEVICES'] = ''
os.environ['PT_HPU_LAZY_MODE'] = '0'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = '0'

# Import PyTorch
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

class InferenceCallback(TrainerCallback):
    """Custom callback to generate text at the end of each epoch or eval step."""
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def train_cpu_lora(
    model_name: str = "Qwen/Qwen3-0.6B",
    data_path: str = "TeichAI/claude-4.5-opus-high-reasoning-250x",
    output_dir: str = "./checkpoints/qwen-0.6b-lora",
    num_train_epochs: int = 1,
    max_steps: int = -1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    use_gpu: bool = None
):
    """
    Train Qwen-0.6B using LoRA.
    Auto-detects GPU if use_gpu is None.
    """
    # Auto-detection
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    device = "cuda" if use_gpu else "cpu"
    
    logger.info("=" * 50)
    logger.info(f"Qwen-0.6B LoRA Fine-tuning on {device.upper()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if use_gpu and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        logger.info(f"GPU available: True ({gpu_name})")
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            logger.info(f"ROCm / HIP version: {torch.version.hip}")
        else:
            logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info("=" * 50)

    # Check for Hugging Face authentication (token from environment or HF_TOKEN)
    from huggingface_hub import login
    import os

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

    # Load dataset(s)
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

    # Load model
    logger.info(f"4/6 Loading Qwen-0.6B model ({device.upper()})...")

    try:
        # Use bfloat16 for GPU if supported, float32 for CPU
        torch_dtype = torch.bfloat16 if (use_gpu and torch.cuda.is_bf16_supported()) else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if use_gpu else None,
        )
        if not use_gpu:
            model = model.to("cpu")
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
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

    # Force CPU device for the model (already loaded, no move necessary)
    # Training arguments (CPU-specific)
    eval_save_steps = max(1, max_steps // 4) if max_steps > 0 else 50
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
        dataloader_pin_memory=False,
        use_cpu=not use_gpu,
        dataloader_num_workers=0,  # Fix for Windows hanging
        fp16=use_gpu and not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=use_gpu and torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
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
    logger.info("Starting CPU-based LoRA fine-tuning...")
    logger.info("=" * 50)

    trainer.train()

    # Save model
    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output_dir)

    logger.info(f"Training complete. Model saved to {final_output_dir}")
    return trainer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CPU-based Qwen-0.6B LoRA Fine-tuning"
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
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (overrides epochs if > 0)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")

    # Hardware-specific settings
    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU training")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision (requires Ampere+ GPU / ROCm)")
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false", default=True,
                        help="Disable 4-bit quantization")

    args = parser.parse_args()

    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False

    train_cpu_lora(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir="./checkpoints/qwen-0.6b-lora",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_gpu=use_gpu
    )


if __name__ == "__main__":
    main()
