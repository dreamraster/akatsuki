# By dreamraster · dreaMSCend
"""
XTokenNode — Pipeline node for X-Token distillation.

Integrates X-Token knowledge distillation into the hmlcore pipeline.
"""

from __future__ import annotations

import logging
import os
import torch
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, field

from hmlcore.nodes.base import BaseNode, NodeError
from hmlcore.nodes.context import NodeContext
from hmlcore.xtoken.distiller import XTokenDistiller, XTokenConfig
from hmlcore.xtoken.projection import ProjectionAligner

logger = logging.getLogger(__name__)


class XTokenCollateFn:
    """Pickleable collate function for X-Token dataloader (required for Windows/spawn)."""
    def __init__(self, teacher_tokenizer, student_tokenizer):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

    def __call__(self, batch):
        # Extract texts from batch
        texts = [example["text"] for example in batch]
        
        # Tokenize for teacher
        teacher_encoded = self.teacher_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=getattr(self.teacher_tokenizer, "model_max_length", 2048),
            return_tensors="pt",
        )
        
        # Tokenize for student
        student_encoded = self.student_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=getattr(self.student_tokenizer, "model_max_length", 2048),
            return_tensors="pt",
        )
        
        # Create labels
        labels = student_encoded["input_ids"].clone()
        
        return {
            "input_ids": student_encoded["input_ids"],
            "attention_mask": student_encoded["attention_mask"],
            "teacher_input_ids": teacher_encoded["input_ids"],
            "teacher_attention_mask": teacher_encoded["attention_mask"],
            "labels": labels,
            "text": texts,
        }


class XTokenNode(BaseNode):
    """
    Pipeline node that performs X-Token knowledge distillation.
    
    This node wraps an existing model and applies projection-guided
    cross-tokenizer distillation during training.
    
    INPUT_KEYS:
        - model: Base model to enhance with distillation
        - tokenizer: Tokenizer for the model
        - dataset: Training dataset
        - args: CLI arguments
        
    OUTPUT_KEYS:
        - model: Enhanced model with distillation capabilities
        - distiller: XTokenDistiller instance
        - distillation_config: Config used for distillation
    """
    
    NAME = "XTokenNode"
    INPUT_KEYS = ("model", "tokenizer", "dataset", "args")
    OUTPUT_KEYS = (
        "distiller", 
        "distillation_config",
        "xtoken_enabled",
    )
    
    def __init__(
        self,
        teacher_model_path: Optional[str] = None,
        projection_type: str = "mlp",
        hidden_dim: Optional[int] = None,
        enable_distillation: bool = True,
    ):
        """
        Initialize XTokenNode.
        
        Args:
            teacher_model_path: Path or HF ID for teacher model
            projection_type: Type of projection ("linear", "mlp", "identity")
            hidden_dim: Hidden dimension for MLP projection
            enable_distillation: Whether to enable distillation
        """
        self.teacher_model_path = teacher_model_path
        self.projection_type = projection_type
        self.hidden_dim = hidden_dim
        self.enable_distillation = enable_distillation
        
        self.distiller: Optional[XTokenDistiller] = None
        self.config: Optional[XTokenConfig] = None
        
    def should_run(self, ctx: NodeContext) -> bool:
        """Determine if this node should run."""
        if not self.enable_distillation:
            logger.info("XTokenNode: distillation disabled, skipping")
            return False
        
        # Check if already processed
        if ctx.get("xtoken_enabled", False):
            logger.info("XTokenNode: already enabled, skipping")
            return False
            
        return True
    
    def run(self, ctx: NodeContext) -> None:
        """Execute X-Token distillation setup."""
        self._require(ctx, "model", "tokenizer", "dataset", "args")
        
        args = ctx["args"]
        model = ctx["model"]
        tokenizer = ctx["tokenizer"]
        dataset = ctx["dataset"]
        
        # Determine teacher model
        teacher_path = self.teacher_model_path or getattr(
            args, "teacher_model", None
        )
        
        if teacher_path is None:
            # Use the same model as teacher (self-distillation)
            logger.info("XTokenNode: no teacher specified, using self-distillation")
            teacher_path = args.student_model
        
        # Create distillation config
        self.config = XTokenConfig(
            projection_type=self.projection_type,
            hidden_dim=self.hidden_dim,
            learning_rate=getattr(args, "learning_rate", 2e-4),
            batch_size=getattr(args, "batch_size", 1),
            gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 4),
            max_length=getattr(args, "max_length", 2048),
            checkpoint_dir=os.path.join(args.output_dir, "xtoken_checkpoints"),
            save_steps=getattr(args, "xtoken_save_steps", 500),
        )
        
        # Create distiller
        logger.info(f"XTokenNode: Initializing distiller with teacher={teacher_path}")
        
        try:
            from hmlcore.xtoken.distiller import XTokenDistiller
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading teacher model: {teacher_path}")
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_path,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
            if teacher_tokenizer.pad_token is None:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

            # student model and tokenizer are already loaded by InputNode (with active PEFT/Unsloth patches)
            # Do NOT reload student model to preserve Unsloth's patched class and parameters!
            self.distiller = XTokenDistiller(
                teacher_model=teacher_model,
                student_model=model,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=tokenizer,
                config=self.config,
            )
            
        except Exception as exc:
            raise NodeError(f"Failed to create XToken distiller: {exc}") from exc
        
        # Wrap the student model in distiller
        ctx["model"] = self.distiller.student_model
        ctx["distiller"] = self.distiller
        ctx["distillation_config"] = self.config.to_dict()
        ctx["xtoken_enabled"] = True
        
        logger.info("XTokenNode: Distillation pipeline ready")
        
    def get_distiller(self) -> XTokenDistiller:
        """Get the created distiller."""
        if self.distiller is None:
            raise NodeError("Distiller not initialized. Run the node first.")
        return self.distiller


class XTokenDistillNode(BaseNode):
    """
    Pipeline node that actively runs X-Token distillation training.
    
    This node performs the actual distillation training using the
    configured distiller.
    """
    
    NAME = "XTokenDistillNode"
    INPUT_KEYS = (
        "model", 
        "tokenizer", 
        "dataset", 
        "args",
        "distiller",
        "distillation_config",
    )
    OUTPUT_KEYS = (
        "distiller",
        "distillation_metrics",
    )
    
    def __init__(
        self,
        num_epochs: int = 3,
        batch_size: int = 1,
        max_steps: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        """
        Initialize distillation training node.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            max_steps: Maximum number of steps (overrides num_epochs if set)
            callbacks: Optional list of training callbacks
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.callbacks = callbacks or []
        self.metrics: dict = {}
        
    def should_run(self, ctx: NodeContext) -> bool:
        """Determine if this node should run."""
        if not ctx.get("xtoken_enabled", False):
            logger.info("XTokenDistillNode: xtoken not enabled, skipping")
            return False
        
        if ctx.get("distillation_completed", False):
            logger.info("XTokenDistillNode: distillation already completed, skipping")
            return False
            
        return True
    
    def run(self, ctx: NodeContext) -> None:
        """Run X-Token distillation training."""
        self._require(
            ctx, 
            "model", 
            "tokenizer", 
            "dataset", 
            "args",
            "distiller",
        )
        
        model = ctx["model"]
        tokenizer = ctx["tokenizer"]
        dataset = ctx["dataset"]
        args = ctx["args"]
        distiller = ctx["distiller"]
        
        # Create dataloader
        from torch.utils.data import DataLoader
        import platform
        # On Windows, force num_workers=0 to avoid multiprocessing pickling issues with Unsloth-patched tokenizers
        num_workers = 0 if platform.system() == "Windows" else getattr(args, "num_workers", 4)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._create_collate_fn(distiller.teacher_tokenizer, distiller.student_tokenizer),
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Determine training duration
        if self.max_steps is not None:
            total_steps = self.max_steps
            num_epochs = 1  # Will run until max_steps reached
        else:
            total_steps = len(dataloader) * self.num_epochs
            num_epochs = self.num_epochs
        
        # Update scheduler T_max
        distiller.scheduler.T_max = total_steps
        
        # Training loop
        logger.info(
            f"XTokenDistillNode: Starting training for {num_epochs} epochs "
            f"({total_steps} steps)"
        )
        
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                # Training step
                metrics = self._training_step(distiller, batch)
                
                total_loss += metrics["loss"]
                global_step += 1
                
                # Progress logging
                if batch_idx % 100 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    logger.info(
                        f"  Step {batch_idx}: Loss = {avg_loss:.4f}"
                    )
                
                # Check max steps
                if self.max_steps is not None and global_step >= self.max_steps:
                    logger.info(
                        f"Reached max steps ({self.max_steps}), stopping early"
                    )
                    break
            
            # Epoch summary
            epoch_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if epoch < num_epochs - 1:  # Not last epoch
                checkpoint_dir = os.path.join(
                    args.output_dir, 
                    f"xtoken_epoch_{epoch + 1}"
                )
                distiller.save_checkpoint(checkpoint_dir)
        
        # Final checkpoint
        final_dir = os.path.join(args.output_dir, "xtoken_final")
        distiller.save_checkpoint(final_dir)
        
        # Update context
        ctx["distillation_metrics"] = {
            "total_loss": total_loss,
            "num_steps": global_step,
            "num_epochs": num_epochs,
            "average_loss": total_loss / global_step,
        }
        ctx["distillation_completed"] = True
        
        logger.info("XTokenDistillNode: Training complete")
        
    def _create_collate_fn(self, teacher_tokenizer, student_tokenizer):
        """Create collate function for dataloader."""
        return XTokenCollateFn(teacher_tokenizer, student_tokenizer)
    
    def _training_step(
        self,
        distiller: XTokenDistiller,
        batch: dict,
    ) -> dict:
        """Perform a single training step."""
        model = distiller.student_model
        optimizer = distiller.optimizer
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        input_ids = batch["input_ids"].to(distiller.device)
        attention_mask = batch["attention_mask"].to(distiller.device)
        teacher_input_ids = batch["teacher_input_ids"].to(distiller.device)
        teacher_attention_mask = batch["teacher_attention_mask"].to(distiller.device)
        labels = batch["labels"].to(distiller.device)
        
        outputs = distiller.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            teacher_input_ids=teacher_input_ids,
            teacher_attention_mask=teacher_attention_mask,
            labels=labels,
        )
        
        # Backward pass
        outputs["loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            distiller.get_trainable_params(),
            max_norm=1.0,
        )
        
        optimizer.step()
        distiller.scheduler.step()
        
        return {
            "loss": outputs["loss"].item(),
            "alignment_loss": outputs["alignment_loss"].item(),
            "matching_loss": outputs["matching_loss"].item(),
        }


# Convenience function for creating XToken pipeline
def create_xtoken_pipeline(
    teacher_model_path: Optional[str] = None,
    projection_type: str = "mlp",
    hidden_dim: Optional[int] = None,
    num_epochs: int = 3,
    enable_distillation: bool = True,
) -> List[BaseNode]:
    """
    Create a list of nodes for X-Token distillation pipeline.
    
    Args:
        teacher_model_path: Path or HF ID for teacher model
        projection_type: Type of projection
        hidden_dim: Hidden dimension for MLP projection
        num_epochs: Number of training epochs
        enable_distillation: Whether to enable distillation
        
    Returns:
        List of pipeline nodes
    """
    nodes = []
    
    if enable_distillation:
        # Node 1: Setup distillation
        nodes.append(XTokenNode(
            teacher_model_path=teacher_model_path,
            projection_type=projection_type,
            hidden_dim=hidden_dim,
            enable_distillation=True,
        ))
        
        # Node 2: Run distillation training
        nodes.append(XTokenDistillNode(
            num_epochs=num_epochs,
            batch_size=1,
            callbacks=[],
        ))
    
    return nodes
