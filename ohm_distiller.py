# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
Online Logit-Based Knowledge Distillation with QLoRA

This script implements high-quality, memory-efficient knowledge distillation
for large MoE/dense models (~128GB) on a 192GB VRAM machine.

Strategy: Online Logit KD (Approach 2 from distillation_strategy.md)
  - Teacher: loaded in 4-bit quantization (~32 GB VRAM), frozen
  - Student: loaded with QLoRA (~3.5 GB base + tiny LoRA adapters)
  - Loss: α × KL(teacher_soft, student_soft) + (1-α) × CE(student, labels)
  - Validation: multi-signal guardrails every epoch (perplexity, generation checks, early stopping)

Modes:
  1. "online"  — Teacher + student both loaded. Student learns from teacher's soft logits.
  2. "offline" — Teacher generates CoT data first, student trains on it. (Fallback mode)
  3. "cot_finetune" — Student fine-tunes on existing CoT datasets only. (No teacher needed)

Usage:
    # Online logit distillation (best quality, ~50 GB VRAM)
    python ohm_distiller.py --mode online \\
        --teacher_model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
        --student_model "Qwen/Qwen3-0.6B" \\
        --datasets "TeichAI/claude-4.5-opus-high-reasoning-250x"

    # Offline CoT generation + training (lower VRAM, lower quality)
    python ohm_distiller.py --mode offline \\
        --teacher_model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
        --student_model "Qwen/Qwen3-0.6B"

    # CoT fine-tuning only (no teacher needed, ~5 GB VRAM)
    python ohm_distiller.py --mode cot_finetune \\
        --student_model "Qwen/Qwen3-0.6B" \\
        --datasets "TeichAI/claude-4.5-opus-high-reasoning-250x"
"""

import os
import argparse
import json
import math
import logging

# Disable Habana plugins that might cause issues in some environments
os.environ['HABANA_VISIBLE_DEVICES'] = ''
os.environ['PT_HPU_LAZY_MODE'] = '0'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = '0'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── hmlcore: shared utilities ─────────────────────────────────────────────────
# ohm_distiller is an online/offline KD pipeline (KL divergence + CE loss) —
# it does not use the SFT/GRPO/REAP node graph.  The following hmlcore modules
# are used where they provide genuine overlap:
#   • hmlcore.model  — student model loading (QLoRA, same base as finetuner)
#   • hmlcore.moe    — REAP expert pruning (optional post-distillation step)
from hmlcore.model import load_model_and_tokenizer as _hml_load_model
from hmlcore.moe   import find_moe_layers, reap_prune_moe


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation Callbacks — Rock-Solid Per-Epoch Checks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CODING_EVAL_PROMPT = (
    "<|im_start|>user\n"
    "Write a Python function that takes a list of integers and returns the two "
    "numbers that add up to a given target. Include error handling.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

CODE_KEYWORDS = ['def ', 'class ', 'return ', 'import ', 'for ', 'if ', 'while ',
                 'print(', 'try:', 'except', 'with ', 'yield ', 'lambda ']


class DistillationGuardrails(TrainerCallback):
    """
    Multi-signal validation after every evaluation step.

    Checks:
      1. Perplexity explosion (stops training if > threshold)
      2. Loss divergence (warns if loss jumps > 50%)
      3. Generation sanity (warns if output doesn't contain code patterns)
      4. Early stopping with patience (stops if no improvement for N evals)
    """

    def __init__(self, tokenizer, eval_prompt=CODING_EVAL_PROMPT,
                 ppl_threshold=150.0, patience=5):
        self.tokenizer = tokenizer
        self.eval_prompt = eval_prompt
        self.ppl_threshold = ppl_threshold
        self.patience = patience
        self.prev_loss = float('inf')
        self.best_ppl = float('inf')
        self.patience_counter = 0
        self.epoch_history = []

    def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
        eval_loss = metrics.get("eval_loss", float('inf'))

        # Guard against NaN/Inf loss
        if math.isnan(eval_loss) or math.isinf(eval_loss):
            logger.error("⛔ eval_loss is NaN/Inf — STOPPING training")
            control.should_training_stop = True
            return

        ppl = math.exp(min(eval_loss, 20.0))  # Clamp to avoid overflow

        # ── Guardrail 1: Perplexity explosion ──
        if ppl > self.ppl_threshold:
            logger.error(
                f"⛔ Perplexity exploded to {ppl:.1f} (threshold={self.ppl_threshold}) "
                f"— STOPPING training. Check learning rate or data quality."
            )
            control.should_training_stop = True
            return

        # ── Guardrail 2: Loss divergence (sudden 50%+ jump) ──
        if state.epoch and state.epoch > 1 and eval_loss > self.prev_loss * 1.5:
            pct = ((eval_loss / self.prev_loss) - 1) * 100
            logger.warning(
                f"⚠️ Loss jumped {pct:.0f}% (from {self.prev_loss:.4f} to {eval_loss:.4f}) "
                f"— potential overfitting or data issue"
            )

        # ── Guardrail 3: Generation sanity check ──
        if model is not None and state.epoch and state.epoch >= 1:
            try:
                sample = self._generate_sample(model)
                has_code = any(kw in sample for kw in CODE_KEYWORDS)
                if not has_code:
                    logger.warning(
                        f"⚠️ Model output doesn't contain code patterns. "
                        f"Sample (first 300 chars):\n{sample[:300]}"
                    )
                else:
                    logger.info(f"✅ Generation check passed. Sample (first 200 chars):\n{sample[:200]}")
            except Exception as e:
                logger.warning(f"⚠️ Generation check failed: {e}")

        # ── Guardrail 4: Early stopping with patience ──
        if ppl < self.best_ppl:
            self.best_ppl = ppl
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(
                    f"⏹️ Early stopping: perplexity hasn't improved for "
                    f"{self.patience} evaluations (best={self.best_ppl:.1f})"
                )
                control.should_training_stop = True

        # ── Log epoch summary ──
        self.epoch_history.append({
            "epoch": state.epoch, "loss": eval_loss, "ppl": ppl
        })
        logger.info(
            f"📊 Epoch {state.epoch:.1f} | Step {state.global_step} | "
            f"Loss: {eval_loss:.4f} | PPL: {ppl:.1f} | "
            f"Best PPL: {self.best_ppl:.1f} | Patience: {self.patience_counter}/{self.patience}"
        )
        self.prev_loss = eval_loss

    def _generate_sample(self, model):
        """Generate a short code sample to verify the model is producing code."""
        model.eval()
        inputs = self.tokenizer(
            self.eval_prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        model.train()
        return generated

    def get_training_summary(self):
        """Return a summary of training history for logging."""
        if not self.epoch_history:
            return "No epochs completed."
        best = min(self.epoch_history, key=lambda x: x["ppl"])
        return (
            f"Training completed: {len(self.epoch_history)} evaluations | "
            f"Best PPL: {best['ppl']:.1f} at epoch {best['epoch']:.1f} | "
            f"Final PPL: {self.epoch_history[-1]['ppl']:.1f}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset Loading & Preprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_and_merge_datasets(dataset_paths: list) -> Dataset:
    """
    Load datasets from local JSONL files or HuggingFace repos and merge them.
    Supports multiple key formats: instruction/response, prompt/output, messages, etc.
    """
    all_datasets = []
    for dp in dataset_paths:
        logger.info(f"Loading dataset from: {dp}")
        try:
            if os.path.exists(dp) and dp.endswith('.jsonl'):
                data_list = []
                with open(dp, 'r') as f:
                    for line in f:
                        data_list.append(json.loads(line))
                all_datasets.append(Dataset.from_list(data_list))
            else:
                raw = load_dataset(dp, trust_remote_code=True)
                if isinstance(raw, dict):
                    for split_name in ["train", "cot", "default"]:
                        if split_name in raw:
                            all_datasets.append(raw[split_name])
                            break
                    else:
                        all_datasets.append(next(iter(raw.values())))
                else:
                    all_datasets.append(raw)
        except Exception as e:
            logger.error(f"Failed to load dataset '{dp}': {e}")

    if not all_datasets:
        raise ValueError("No valid datasets loaded. Aborting.")

    merged = concatenate_datasets(all_datasets)
    logger.info(f"Total merged examples: {len(merged)}")
    return merged


def preprocess_dataset(dataset: Dataset, tokenizer, max_length: int = 2048):
    """
    Tokenize dataset for causal LM training.
    Masks the prompt tokens so loss is only computed on the response.
    Supports: instruction/response, prompt/output, question/answer, messages formats.
    """
    def format_example(example):
        # Extract instruction and response from various formats
        if "messages" in example:
            messages = example["messages"]
            instruction = ""
            response = ""
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        instruction = content
                    elif role == "assistant":
                        response = content
        else:
            instruction = example.get("instruction",
                          example.get("prompt",
                          example.get("question", "")))
            response = example.get("response",
                       example.get("output",
                       example.get("answer", "")))

        # Build ChatML formatted text
        prompt = (
            "<|im_start|>user\n"
            f"{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full_text = prompt + response + "<|im_end|>"

        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=False,
        )

        # Create labels — mask prompt tokens with -100
        labels = encoding["input_ids"].copy()
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # Mask padding tokens
        for i in range(len(labels)):
            if labels[i] == tokenizer.pad_token_id:
                labels[i] = -100

        encoding["labels"] = labels
        return encoding

    processed = dataset.map(format_example, desc="Tokenizing dataset")
    return processed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Loading Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_hardware():
    """Detect available hardware and return configuration."""
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_name = torch.cuda.get_device_name()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        backend = "ROCm/HIP" if is_rocm else f"CUDA {torch.version.cuda}"
        logger.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB | Backend: {backend}")
        bf16_ok = torch.cuda.is_bf16_supported()
    else:
        vram_gb = 0
        bf16_ok = False
        logger.warning("No GPU detected — falling back to CPU (very slow)")

    return {
        "use_gpu": use_gpu,
        "vram_gb": vram_gb,
        "bf16": bf16_ok,
        "torch_dtype": torch.bfloat16 if (use_gpu and bf16_ok) else (torch.float16 if use_gpu else torch.float32),
    }


def load_teacher_model(model_name: str, hw: dict):
    """
    Load teacher model in 4-bit quantization for memory efficiency.
    Teacher is frozen (no gradients) — used only for forward passes.
    ~32 GB VRAM for a 128GB model.
    """
    logger.info(f"Loading Teacher Model (4-bit): {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # Double quantization saves ~0.4 bits/param
        bnb_4bit_compute_dtype=hw["torch_dtype"],
        bnb_4bit_quant_type="nf4",            # NormalFloat4 — optimal for pretrained weights
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=hw["torch_dtype"],
        trust_remote_code=True,
    )

    # Freeze teacher completely — no gradients needed
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    param_count = sum(p.numel() for p in model.parameters())
    vram_est = param_count * 0.5 / (1024**3)  # 4-bit ≈ 0.5 bytes/param
    logger.info(f"Teacher loaded: {param_count/1e9:.1f}B params | ~{vram_est:.1f} GB VRAM (4-bit)")
    return model


def load_student_model(model_name: str, hw: dict, lora_rank: int = 16,
                       lora_alpha: int = 32, load_in_4bit: bool = True):
    """
    Load student model with QLoRA for memory-efficient training.
    Base weights in 4-bit (~3.5 GB for 7B model), LoRA adapters in full precision.
    """
    logger.info(f"Loading Student Model (QLoRA): {model_name}")

    is_local = os.path.exists(model_name) and os.path.isdir(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=is_local
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with optional 4-bit quantization
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": hw["torch_dtype"],
        "local_files_only": is_local,
    }

    if load_in_4bit and hw["use_gpu"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=hw["torch_dtype"],
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    elif hw["use_gpu"]:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare model for k-bit training (handles frozen weights + gradient stuff)
    if load_in_4bit and hw["use_gpu"]:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA adapters
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Log trainable parameter stats
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Student loaded: {total/1e6:.1f}M total params | "
        f"{trainable/1e6:.1f}M trainable ({100*trainable/total:.2f}%) | "
        f"LoRA rank={lora_rank}, alpha={lora_alpha}"
    )

    return model, tokenizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Online Logit-Based Knowledge Distillation (Approach 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DistillationTrainer(Trainer):
    """
    Custom Trainer that implements online logit-based knowledge distillation.

    Instead of just training on hard labels (the correct next token), this trainer
    also makes the student match the teacher's full probability distribution over
    all tokens — the "soft labels" that carry dark knowledge about which tokens
    are related to each other.

    Loss = α × KL_divergence(teacher_soft, student_soft) + (1-α) × cross_entropy(student, labels)
    """

    def __init__(self, teacher_model=None, temperature=4.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        if teacher_model is not None:
            logger.info(
                f"🎓 Distillation config: temperature={temperature}, alpha={alpha} "
                f"(KD weight={alpha:.0%}, CE weight={1-alpha:.0%})"
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation that blends KL divergence with cross-entropy.

        If no teacher is loaded, falls back to standard cross-entropy (CoT fine-tuning mode).
        """
        # Standard student forward pass
        labels = inputs.pop("labels", None)
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # ── No teacher → standard cross-entropy loss ──
        if self.teacher_model is None:
            if labels is not None:
                # Shift logits and labels for causal LM
                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = student_outputs.loss
            return (loss, student_outputs) if return_outputs else loss

        # ── Teacher forward pass (frozen, no gradients) ──
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # ── Compute distillation loss ──
        # Shift for causal LM (predict next token)
        s_logits = student_logits[..., :-1, :].contiguous()
        t_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() if labels is not None else None

        # Handle vocabulary size mismatch between teacher and student
        min_vocab = min(s_logits.size(-1), t_logits.size(-1))
        s_logits = s_logits[..., :min_vocab]
        t_logits = t_logits[..., :min_vocab]

        # Temperature-scaled soft probabilities
        # Higher temperature "softens" the distribution, revealing dark knowledge
        T = self.temperature
        student_soft = F.log_softmax(s_logits / T, dim=-1)
        teacher_soft = F.softmax(t_logits / T, dim=-1)

        # KL divergence: how different is the student's distribution from the teacher's?
        # Multiply by T² to balance gradient magnitudes across temperatures
        kd_loss = F.kl_div(
            student_soft.view(-1, min_vocab),
            teacher_soft.view(-1, min_vocab),
            reduction='batchmean',
        ) * (T ** 2)

        # Standard cross-entropy with hard labels
        if shift_labels is not None:
            # Truncate vocab if student has fewer tokens
            ce_logits = student_logits[..., :-1, :].contiguous()
            ce_loss = F.cross_entropy(
                ce_logits.view(-1, ce_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            ce_loss = torch.tensor(0.0, device=s_logits.device)

        # Blend: α controls the mix of KD vs CE loss
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Offline CoT Generation (Fallback / Legacy Mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_cot_dataset(
    teacher_model_name: str,
    prompts: list,
    output_file: str,
    system_prompt: str = (
        "You are a coding expert. Before providing the final code, you must "
        "explain your step-by-step reasoning inside <thought>...</thought> tags."
    ),
    batch_size: int = 2,
):
    """
    Uses a teacher model to generate Chain-of-Thought reasoning for coding prompts.
    Teacher loads alone (full VRAM available), generates data, then exits.
    """
    logger.info(f"Loading Teacher for CoT generation: {teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)

    hw = detect_hardware()
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        device_map="auto",
        torch_dtype=hw["torch_dtype"],
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_records = []
    logger.info(f"Generating CoT responses for {len(prompts)} prompts...")

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]

        formatted_prompts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n<thought>\n"
            for p in batch_prompts
        ]

        inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_length = inputs.input_ids[j].shape[0]
            response_tokens = output[input_length:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            full_response = f"<thought>\n{response_text}"

            dataset_records.append({
                "instruction": batch_prompts[j],
                "response": full_response,
            })

    with open(output_file, 'w') as f:
        for record in dataset_records:
            f.write(json.dumps(record) + '\n')

    logger.info(f"CoT dataset saved to {output_file} ({len(dataset_records)} examples)")

    # Free VRAM for the next phase
    del model
    torch.cuda.empty_cache()
    return output_file


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Distillation Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_distillation(
    mode: str,
    teacher_model_name: str,
    student_model_name: str,
    dataset_paths: list,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    temperature: float = 4.0,
    alpha: float = 0.7,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    eval_split: float = 0.1,
    load_in_4bit: bool = True,
    patience: int = 5,
    # PRISM parameters
    prism_select: bool = False,
    prism_tier: str = "high",
    prism_layer: int = -1,
    prism_batch: int = 16,
    prism_chunk: int = 2000,
):
    """
    Main distillation pipeline supporting online KD, offline CoT, and CoT fine-tuning.
    """
    logger.info("=" * 60)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Teacher: {teacher_model_name}")
    logger.info(f"Student: {student_model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    hw = detect_hardware()
    teacher_model = None

    # ── Phase 1: Load Teacher (if needed) ──
    if mode == "online":
        teacher_model = load_teacher_model(teacher_model_name, hw)
    elif mode == "offline":
        # Generate CoT data, save to disk, then unload teacher
        generated_path = os.path.join(output_dir, "generated_cot_dataset.jsonl")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(generated_path):
            sample_prompts = [
                "Write a Python function to perform a binary search on a sorted array.",
                "How do you invert a binary tree in Python? Provide an elegant recursive solution.",
                "Write a highly optimized function to find the nth Fibonacci number without recursion.",
                "Implement a Python class for a thread-safe queue with timeout support.",
                "Write a function that detects cycles in a directed graph using DFS.",
                "Implement merge sort for a linked list in Python.",
                "Write a Python decorator that caches function results with a TTL (time-to-live).",
                "Implement a trie data structure with insert, search, and prefix matching.",
            ]
            generate_cot_dataset(
                teacher_model_name=teacher_model_name,
                prompts=sample_prompts,
                output_file=generated_path,
            )
        else:
            logger.info(f"Using existing CoT dataset: {generated_path}")

        dataset_paths = [generated_path] + (dataset_paths or [])

    # ── Phase 2: Load Student ──
    student_model, tokenizer = load_student_model(
        student_model_name, hw,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        load_in_4bit=load_in_4bit,
    )

    # ── Phase 3: Load and preprocess datasets ──
    if not dataset_paths:
        raise ValueError(
            "No datasets provided. Use --datasets to specify HuggingFace datasets "
            "or local .jsonl files, or use --mode offline to generate a CoT dataset."
        )

    dataset = load_and_merge_datasets(dataset_paths)
    
    # ── Phase 3.5: PRISM Data Selection ──
    if prism_select:
        logger.info(f"💎 PRISM: Selecting {prism_tier} tier samples...")
        from hmlcore.prism_selector import select_with_prism
        # Use a temporary cache path in the output directory
        cache_path = os.path.join(output_dir, "prism_cache.pt")
        dataset = select_with_prism(
            dataset=dataset,
            model=student_model,
            tokenizer=tokenizer,
            tier=prism_tier,
            layer=prism_layer,
            batch_size=prism_batch,
            cache_path=cache_path,
            chunk_size=prism_chunk,
        )
        logger.info(f"✅ PRISM selection complete: {len(dataset)} examples remaining.")

    processed = preprocess_dataset(dataset, tokenizer, max_length=max_length)

    # Train/eval split
    dataset_size = len(processed)
    split_idx = int((1 - eval_split) * dataset_size)
    processed = processed.shuffle(seed=42)
    train_dataset = processed.select(range(split_idx))
    eval_dataset = processed.select(range(split_idx, dataset_size))
    logger.info(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval examples")

    # ── Phase 4: Configure training ──
    use_gpu = hw["use_gpu"]
    eval_save_steps = max(1, len(train_dataset) // (batch_size * gradient_accumulation_steps * 2))

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        # Precision
        fp16=use_gpu and not hw["bf16"],
        bf16=use_gpu and hw["bf16"],
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        # Logging
        logging_strategy="steps",
        logging_steps=max(1, eval_save_steps // 5),
        # System
        use_cpu=not use_gpu,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=42,
    )

    # Validation callbacks
    guardrails = DistillationGuardrails(
        tokenizer=tokenizer,
        patience=patience,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Phase 5: Train ──
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=temperature,
        alpha=alpha,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[guardrails],
    )

    logger.info("=" * 60)
    mode_desc = {
        "online": "Online Logit KD (teacher + student, KL div + CE loss)",
        "offline": "Offline CoT (teacher-generated data, CE loss only)",
        "cot_finetune": "CoT Fine-tuning (existing datasets, CE loss only)",
    }
    logger.info(f"Starting: {mode_desc.get(mode, mode)}")
    logger.info(f"Device: {trainer.args.device}")
    logger.info("=" * 60)

    trainer.train()

    # ── Phase 6: Save final model ──
    final_dir = os.path.join(output_dir, "final_distilled_student")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Log training summary
    logger.info("=" * 60)
    logger.info("✅ Distillation Complete!")
    logger.info(guardrails.get_training_summary())
    logger.info(f"Final model saved to: {final_dir}")
    logger.info("=" * 60)

    # Cleanup teacher
    if teacher_model is not None:
        del teacher_model
        torch.cuda.empty_cache()

    return trainer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Pipeline (Online Logit KD / Offline CoT / CoT Fine-tuning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Online logit distillation (best quality, ~50 GB VRAM)
  python cot_distillation.py --mode online \\
      --teacher_model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
      --student_model "Qwen/Qwen3-0.6B" \\
      --datasets "TeichAI/claude-4.5-opus-high-reasoning-250x"

  # CoT fine-tuning only (no teacher, ~5 GB VRAM)
  python cot_distillation.py --mode cot_finetune \\
      --student_model "Qwen/Qwen3-0.6B" \\
      --datasets "dataset1,dataset2"

  # Offline CoT generation + training
  python cot_distillation.py --mode offline \\
      --teacher_model "Qwen/Qwen2.5-Coder-32B-Instruct"
        """,
    )

    # ── Mode ──
    parser.add_argument(
        "--mode", type=str, default="online",
        choices=["online", "offline", "cot_finetune"],
        help="Distillation mode: 'online' (teacher+student, best quality), "
             "'offline' (generate CoT then train), 'cot_finetune' (train on existing data only)"
    )

    # ── Models ──
    parser.add_argument(
        "--teacher_model", type=str,
        default="Qwen/Qwen3-1.7B-FP8",
        help="Teacher model (HuggingFace ID or local path). Loaded in 4-bit for online mode."
    )
    parser.add_argument(
        "--student_model", type=str,
        default="Qwen/Qwen3-0.6B",
        help="Student model to distill into. Loaded with QLoRA."
    )

    # ── Data ──
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated list of datasets (HuggingFace IDs or local .jsonl paths)."
    )

    # ── Output ──
    parser.add_argument(
        "--output_dir", type=str, default="./distilled",
        help="Directory to save checkpoints and the final distilled model."
    )

    # ── Training hyperparameters ──
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × this)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")

    # ── Distillation hyperparameters ──
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Temperature for softening logits. Higher = softer distributions. (online mode only)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="KD loss weight. 0.7 = 70%% KD + 30%% CE. (online mode only)")

    # ── LoRA configuration ──
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (higher = more capacity but more memory)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor, typically 2× rank)")

    # ── Validation ──
    parser.add_argument("--eval_split", type=float, default=0.1,
                        help="Fraction of data to hold out for validation (0.1 = 10%%)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (stop if no improvement for N evals)")

    # ── Memory optimization ──
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false", default=True,
                        help="Disable 4-bit quantization for student (uses bf16/fp16 instead)")

    # ── PRISM ──
    prism_group = parser.add_argument_group("PRISM Data Selection")
    prism_group.add_argument("--prism_select", action="store_true", help="Enable PRISM data selection before distillation")
    prism_group.add_argument("--prism_tier", type=str, default="high", choices=["high", "mid", "low", "high+mid"],
                              help="Quality tier to keep (default: 'high')")
    prism_group.add_argument("--prism_layer", type=int, default=-1, help="Hidden layer for embeddings")
    prism_group.add_argument("--prism_batch", type=int, default=16, help="Selection batch size")
    prism_group.add_argument("--prism_chunk", type=int, default=2000, help="Correlation chunk size")

    args = parser.parse_args()

    # Parse datasets
    dataset_list = None
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Validate mode requirements
    if args.mode == "cot_finetune" and not dataset_list:
        parser.error("--mode cot_finetune requires --datasets to be specified")

    run_distillation(
        mode=args.mode,
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        dataset_paths=dataset_list,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        temperature=args.temperature,
        alpha=args.alpha,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        eval_split=args.eval_split,
        load_in_4bit=args.load_in_4bit,
        patience=args.patience,
        # PRISM
        prism_select=args.prism_select,
        prism_tier=args.prism_tier,
        prism_layer=args.prism_layer,
        prism_batch=args.prism_batch,
        prism_chunk=args.prism_chunk,
    )


if __name__ == "__main__":
    main()
