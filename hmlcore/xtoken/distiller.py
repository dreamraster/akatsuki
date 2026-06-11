# By dreamraster · dreaMSCend
"""
X-Token Distiller — High-level knowledge distillation interface.

This module provides the main X-Token distiller class that orchestrates
projection-guided cross-tokenizer knowledge distillation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from hmlcore.xtoken.projection import (
    ProjectionAligner,
    CrossTokenizerMatcher,
    ProjectionGuidedDistiller,
)

logger = logging.getLogger(__name__)


@dataclass
class XTokenConfig:
    """Configuration for X-Token distillation."""

    # Projection settings
    projection_type: str = "mlp"
    hidden_dim: Optional[int] = None
    init_scale: float = 0.02

    # Distillation weights
    alignment_weight: float = 1.0
    matching_weight: float = 0.5
    ce_weight: float = 1.0

    # Training settings
    temperature: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Tokenizer settings
    max_length: int = 2048
    padding_side: str = "right"

    # Checkpoint settings
    checkpoint_dir: str = "./checkpoints/xtoken"
    save_steps: int = 500
    save_total_limit: int = 3

    # Multi-teacher settings
    num_teachers: int = 1
    teacher_weights: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_type": self.projection_type,
            "hidden_dim": self.hidden_dim,
            "init_scale": self.init_scale,
            "alignment_weight": self.alignment_weight,
            "matching_weight": self.matching_weight,
            "ce_weight": self.ce_weight,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "checkpoint_dir": self.checkpoint_dir,
            "num_teachers": self.num_teachers,
            "teacher_weights": self.teacher_weights,
        }


class XTokenDistiller:
    """
    X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation.
    
    This distiller uses projection-based alignment to transfer knowledge
    from a teacher model to a student model, even when they use different
    tokenizers.
    
    Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │                     X-Token Distiller                     │
        ├─────────────────────────────────────────────────────────────┤
        │  Input Text → Tokenizer → Tokens                          │
        │                ↓                                           │
        │  Teacher Model (Frozen)                                   │
        │    └──→ Embeddings (d_T)                                  │
        │                ↓ [Projection]                            │
        │  Student Model (Trainable)                                │
        │    └──→ Embeddings (d_S)                                  │
        │                ↓                                           │
        │  Alignment Loss + Matching Loss + CE Loss                 │
        └─────────────────────────────────────────────────────────────┘
    
    Key Features:
        - Cross-tokenizer support via projection alignment
        - Layer-wise embedding alignment
        - Token-level matching with learned similarity
        - Configurable loss weighting
        - Checkpointing and resume support
    """
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        teacher_tokenizer: PreTrainedTokenizerBase,
        student_tokenizer: PreTrainedTokenizerBase,
        config: Optional[XTokenConfig] = None,
    ):
        """
        Initialize X-Token distiller.
        
        Args:
            teacher_model: Teacher model (will be frozen)
            student_model: Student model (will be trained)
            teacher_tokenizer: Teacher tokenizer
            student_tokenizer: Student tokenizer
            config: Configuration (uses defaults if None)
        """
        self.config = config or XTokenConfig()
        
        # Models
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        # Tokenizers
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Move models to device
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        # Create distiller with projection
        self.distiller = ProjectionGuidedDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            projection_type=self.config.projection_type,
            hidden_dim=self.config.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.distiller.get_trainable_params(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,  # Will be updated during training
            eta_min=1e-6,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        logger.info(
            f"XTokenDistiller initialized with {self._count_trainable_params()} "
            f"trainable parameters"
        )
        
    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(
            p.numel() for p in self.distiller.get_trainable_params() 
            if p.requires_grad
        )
    
    def train(
        self,
        dataloader,
        num_epochs: int = 3,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the student model with X-Token distillation.
        
        Args:
            dataloader: PyTorch DataLoader with training data
            num_epochs: Number of training epochs
            callbacks: Optional list of callback functions
            
        Returns:
            Dictionary with training metrics
        """
        self.distiller.train()
        
        all_losses = {
            "total_loss": [],
            "alignment_loss": [],
            "matching_loss": [],
            "ce_loss": [],
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            epoch_losses = {k: [] for k in all_losses}
            
            for batch_idx, batch in enumerate(dataloader):
                self.global_step += 1
                
                # Prepare batch
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                teacher_input_ids = batch.get("teacher_input_ids")
                if teacher_input_ids is not None:
                    teacher_input_ids = teacher_input_ids.to(self.device)
                teacher_attention_mask = batch.get("teacher_attention_mask")
                if teacher_attention_mask is not None:
                    teacher_attention_mask = teacher_attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                
                # Forward pass
                outputs = self.distiller.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    teacher_input_ids=teacher_input_ids,
                    teacher_attention_mask=teacher_attention_mask,
                    labels=labels,
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                outputs["loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.distiller.get_trainable_params(),
                    max_norm=1.0,
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Record losses
                epoch_losses["total_loss"].append(outputs["loss"].item())
                epoch_losses["alignment_loss"].append(
                    outputs["alignment_loss"].item()
                )
                epoch_losses["matching_loss"].append(
                    outputs["matching_loss"].item()
                )
                if "ce_loss" in outputs:
                    epoch_losses["ce_loss"].append(
                        outputs["ce_loss"].item()
                    )
                
                # Progress update
                if batch_idx % 100 == 0:
                    avg_losses = {
                        k: sum(v) / len(v) for k, v in epoch_losses.items()
                    }
                    logger.info(
                        f"  Step {batch_idx}: "
                        f"Total={avg_losses['total_loss']:.4f}, "
                        f"Alignment={avg_losses['alignment_loss']:.4f}, "
                        f"Matching={avg_losses['matching_loss']:.4f}"
                    )
                
                # Save checkpoint
                if (
                    self.config.save_steps > 0 
                    and self.global_step % self.config.save_steps == 0
                ):
                    self.save_checkpoint(
                        os.path.join(
                            self.config.checkpoint_dir,
                            f"step_{self.global_step}"
                        )
                    )
                    
                    # Remove old checkpoints
                    self._cleanup_checkpoints()
                
                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, batch_idx, outputs)
            
            # Epoch metrics
            for k, v in epoch_losses.items():
                all_losses[k].extend(v)
            
            logger.info(f"Epoch {epoch + 1} complete.")
            
        return all_losses
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "student_state_dict": self.student_model.state_dict(),
            "projection_state_dict": self.distiller.projection_aligner.state_dict(),
            "matcher_state_dict": self.distiller.token_matcher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
        
        # Save model in HF format
        self.student_model.save_pretrained(path)
        self.student_tokenizer.save_pretrained(path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            [
                d for d in os.listdir(self.config.checkpoint_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_dir, d))
            ],
            key=lambda x: int(x.split("_")[-1]) if "_" in x else 0,
            reverse=True,
        )
        
        for old_checkpoint in checkpoints[self.config.save_total_limit:]:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, 
                old_checkpoint
            )
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        
        self.student_model.load_state_dict(checkpoint["student_state_dict"])
        self.distiller.projection_aligner.load_state_dict(
            checkpoint["projection_state_dict"]
        )
        self.distiller.token_matcher.load_state_dict(
            checkpoint["matcher_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def distill(
        self,
        source_text: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform distillation on a single text.
        
        Args:
            source_text: Input text to distill
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with distillation results
        """
        self.distiller.eval()
        
        max_len = max_length or self.config.max_length
        
        # Tokenize with teacher tokenizer
        teacher_encoded = self.teacher_tokenizer(
            source_text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        
        # Tokenize with student tokenizer
        student_encoded = self.student_tokenizer(
            source_text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            # Get teacher representation
            teacher_outputs = self.teacher_model(**teacher_encoded, output_hidden_states=True)
            teacher_embeds = teacher_outputs.hidden_states[-1]
            
            # Get student representation
            student_outputs = self.student_model(**student_encoded, output_hidden_states=True)
            student_embeds = student_outputs.hidden_states[-1]
            
            # Project teacher to student space
            projected = self.distiller.projection_aligner.forward(
                teacher_embeds,
                student_embeddings=student_embeds,
                attention_mask=teacher_encoded.get("attention_mask"),
            )
            
            # Compute alignment
            alignment_loss = self.distiller.projection_aligner.compute_alignment_loss(
                projected,
                student_embeds,
                mask=student_encoded.get("attention_mask"),
            )
            
            # Compute matching
            matching_loss = self.distiller.token_matcher.compute_matching_loss(
                teacher_tokens=teacher_encoded["input_ids"],
                student_tokens=student_encoded["input_ids"],
                teacher_mask=teacher_encoded.get("attention_mask"),
                student_mask=student_encoded.get("attention_mask"),
            )
            
        return {
            "teacher_embedding": teacher_embeds.cpu(),
            "student_embedding": student_embeds.cpu(),
            "projected_embedding": projected.cpu(),
            "alignment_loss": alignment_loss.item(),
            "matching_loss": matching_loss.item(),
            "total_loss": (alignment_loss + matching_loss).item(),
        }
    
    def export_model(self, output_path: str) -> None:
        """
        Export distilled model for inference.
        
        Merges projection weights into student model for deployment.
        """
        logger.info(f"Exporting model to {output_path}")
        
        # Save student model with projection info
        actual_teacher_tok = getattr(self.teacher_tokenizer, "tokenizer", self.teacher_tokenizer)
        actual_student_tok = getattr(self.student_tokenizer, "tokenizer", self.student_tokenizer)
        export_config = {
            "projection_type": self.config.projection_type,
            "hidden_dim": self.config.hidden_dim,
            "teacher_vocab_size": len(actual_teacher_tok),
            "student_vocab_size": len(actual_student_tok),
        }
        
        torch.save(
            {"model_state": self.student_model.state_dict(), "config": export_config},
            os.path.join(output_path, "xtoken_model.pt")
        )
        
        self.student_tokenizer.save_pretrained(output_path)
        
        logger.info(f"Model exported to {output_path}")


def create_xtoken_distiller_from_pretrained(
    teacher_model_name: str,
    student_model_name: str,
    config: Optional[XTokenConfig] = None,
    device: Optional[torch.device] = None,
) -> XTokenDistiller:
    """
    Create X-Token distiller from pretrained models.
    
    Args:
        teacher_model_name: HF model name or path for teacher
        student_model_name: HF model name or path for student
        config: Configuration (uses defaults if None)
        device: Device to load models on
        
    Returns:
        Initialized XTokenDistiller
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    logger.info(f"Loading teacher model: {teacher_model_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        device_map=str(device),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    logger.info(f"Loading student model: {student_model_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        device_map=str(device),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # Load tokenizers
    logger.info(f"Loading teacher tokenizer: {teacher_model_name}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    
    logger.info(f"Loading student tokenizer: {student_model_name}")
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    # Set padding token if needed
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    return XTokenDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        config=config,
    )
