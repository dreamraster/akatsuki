# By dreamraster · dreaMSCend
"""
Projection Aligner — Core mechanism for cross-tokenizer knowledge transfer.

This module implements the projection-guided alignment between teacher and student
embedding spaces, enabling knowledge transfer across different tokenizer vocabularies.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional, Tuple, List, Dict
from dataclasses import dataclass
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for a single teacher model in multi-teacher distillation."""
    model: nn.Module
    tokenizer: Any
    weight: float = 1.0  # Weight for this teacher's contribution
    align_layers: Optional[List[int]] = None  # Specific layers to align (-1 for last)


class ProjectionAligner(nn.Module):
    """
    Projects teacher embeddings to student embedding space for knowledge transfer.

    Supports both single and multi-teacher distillation:
    
    Single Teacher:
        Teacher Embedding (d_model_T) --[Projection]--> Student Embedding (d_model_S)
    
    Multi-Teacher (weighted aggregation):
        Teacher 1 -> Projection -> Aligned 1 --|
        Teacher 2 -> Projection -> Aligned 2 --|-> Weighted Sum -> Student Space
        Teacher N -> Projection -> Aligned N --|
        
    The projection can be:
    - Linear: W @ x + b
    - MLP: x -> hidden -> output
    - Identity: direct pass-through (when dimensions match)
    """

    def __init__(
        self,
        teacher_embed_dim: int,
        student_embed_dim: int,
        hidden_dim: Optional[int] = None,
        projection_type: str = "linear",
        init_scale: float = 0.02,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_teachers: int = 1,
    ):
        super().__init__()
        """
        Initialize projection aligner.

        Args:
            teacher_embed_dim: Dimension of teacher model embeddings
            student_embed_dim: Dimension of student model embeddings
            hidden_dim: Hidden dimension for MLP projection (optional)
            projection_type: Type of projection ("linear", "mlp", "identity")
            init_scale: Weight initialization scale
            device: Device to place projections on
            dtype: Data type for projections
            num_teachers: Number of teacher models (for multi-teacher support)
        """
        self.teacher_embed_dim = teacher_embed_dim
        self.student_embed_dim = student_embed_dim
        self.hidden_dim = hidden_dim or student_embed_dim
        self.projection_type = projection_type
        self.num_teachers = num_teachers

        # Projection weights - shared across teachers
        if projection_type == "identity":
            assert teacher_embed_dim == student_embed_dim, \
                f"Identity projection requires matching dims: {teacher_embed_dim} vs {student_embed_dim}"
            self.proj_matrix = nn.Identity()
        elif projection_type == "linear":
            self.proj_matrix = nn.Linear(
                teacher_embed_dim,
                student_embed_dim,
                device=device,
                dtype=dtype,
            )
            self._init_weights(self.proj_matrix.weight, init_scale)
        elif projection_type == "mlp":
            self.proj_matrix = nn.Sequential(
                nn.Linear(teacher_embed_dim, self.hidden_dim, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(self.hidden_dim, student_embed_dim, device=device, dtype=dtype),
            )
            self._init_weights(self.proj_matrix[0].weight, init_scale)
            self._init_weights(self.proj_matrix[2].weight, init_scale)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        # Optional layer normalization
        self.norm = nn.LayerNorm(student_embed_dim, device=device, dtype=dtype)

        # Per-teacher weights for aggregation (learnable)
        if num_teachers > 1:
            self.teacher_weights = nn.Parameter(
                torch.ones(num_teachers, device=device, dtype=dtype) / num_teachers
            )
        else:
            self.teacher_weights = None

        # Cross-attention for alignment (optional)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=student_embed_dim,
            num_heads=min(8, student_embed_dim // 64),
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def _init_weights(self, weight: Tensor, scale: float) -> None:
        """Initialize weights with scaled normal distribution."""
        nn.init.trunc_normal_(weight, std=scale)
        
    def forward(
        self,
        teacher_embeddings: Tensor,
        student_embeddings: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        teacher_weight: Optional[float] = None,
    ) -> Tensor:
        """
        Project teacher embeddings to student space.
        
        Args:
            teacher_embeddings: Teacher model embeddings (batch, seq_len, d_model_T)
            student_embeddings: Optional student embeddings for cross-attention
            attention_mask: Attention mask for cross-attention
            teacher_weight: Optional weight for this teacher (for multi-teacher)
            
        Returns:
            Projected embeddings in student space (batch, seq_len, d_model_S)
        """
        # Project to student dimension
        projected = self.proj_matrix(teacher_embeddings)
        
        # Layer norm
        projected = self.norm(projected)
        
        # Apply teacher weight if provided (for multi-teacher aggregation)
        if teacher_weight is not None:
            projected = projected * teacher_weight
        
        # Cross-attention alignment (if student embeddings provided)
        if student_embeddings is not None:
            # Query: student embeddings, Key/Value: projected teacher embeddings
            # Invert attention_mask (HuggingFace: 1 for valid, 0 for pad. PyTorch: True for pad/ignore, False for valid)
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            aligned, _ = self.cross_attention(
                query=student_embeddings,
                key=projected,
                value=projected,
                key_padding_mask=key_padding_mask,
            )
            return aligned
        
        return projected
    
    def forward_multiple(
        self,
        teacher_embeddings_list: List[Tensor],
        student_embeddings: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Project multiple teacher embeddings and aggregate.
        
        Args:
            teacher_embeddings_list: List of teacher model embeddings
            student_embeddings: Optional student embeddings for cross-attention
            attention_mask: Attention mask for cross-attention
            
        Returns:
            Aggregated projected embeddings in student space
        """
        if len(teacher_embeddings_list) == 0:
            raise ValueError("teacher_embeddings_list cannot be empty")
        
        if len(teacher_embeddings_list) == 1:
            return self.forward(
                teacher_embeddings_list[0],
                student_embeddings,
                attention_mask,
            )
        
        # Forward pass for each teacher
        projections = []
        for i, teacher_embeds in enumerate(teacher_embeddings_list):
            # Get weight for this teacher
            weight = self.teacher_weights[i].item() if self.teacher_weights is not None else 1.0
            projected = self.forward(
                teacher_embeds,
                None,  # Don't use cross-attention yet for individual teachers
                attention_mask,
                teacher_weight=weight,
            )
            projections.append(projected)
        
        # Sum weighted projections
        aggregated = reduce(lambda x, y: x + y, projections)
        
        # Normalize by number of teachers if not using learned weights
        if self.teacher_weights is None:
            aggregated = aggregated / len(teacher_embeddings_list)
        
        # Final cross-attention with student
        if student_embeddings is not None:
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            aligned, _ = self.cross_attention(
                query=student_embeddings,
                key=aggregated,
                value=aggregated,
                key_padding_mask=key_padding_mask,
            )
            return aligned
        
        return aggregated
    
    def align_tokens(
        self,
        teacher_tokens: Tensor,
        student_tokens: Tensor,
        teacher_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Align teacher and student tokens through projection.
        
        Returns:
            Tuple of (aligned_teacher_tokens, aligned_student_tokens)
        """
        # Project teacher tokens
        aligned_teacher = self.forward(teacher_tokens)
        
        # Optionally align student tokens too (if different dim)
        if student_tokens.shape[-1] != self.student_embed_dim:
            aligned_student = self.forward(student_tokens)
        else:
            aligned_student = student_tokens
            
        return aligned_teacher, aligned_student
    
    def compute_alignment_loss(
        self,
        teacher_embeddings: Tensor,
        student_embeddings: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute alignment loss between teacher and student embeddings.
        
        Uses cosine similarity loss to align embedding spaces.
        
        Args:
            teacher_embeddings: Teacher embeddings (batch, seq_len, d_model)
            student_embeddings: Student embeddings (batch, seq_len, d_model)
            mask: Binary mask indicating valid positions
            
        Returns:
            Scalar alignment loss
        """
        # Normalize embeddings
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=-1)
        student_norm = F.normalize(student_embeddings, p=2, dim=-1)
        
        # Cosine similarity loss
        # We want: cos(teacher, student) -> 1 (maximum alignment)
        # Loss = 1 - cosine_similarity
        
        # Compute cosine similarity
        cosine_sim = (teacher_norm * student_norm).sum(dim=-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.to(cosine_sim.device)
            cosine_sim = cosine_sim * mask
            
            # Normalize by mask sum
            mask_sum = mask.sum()
            if mask_sum > 0:
                loss = 1.0 - (cosine_sim.sum() / mask_sum)
            else:
                loss = torch.tensor(0.0, device=cosine_sim.device)
        else:
            loss = 1.0 - cosine_sim.mean()
        
        return loss


class CrossTokenizerMatcher(nn.Module):
    """
    Matches tokens between teacher and student tokenizers via projection.
    
    Uses a learnable matching mechanism to find corresponding tokens across
    different tokenizers, enabling cross-tokenizer knowledge transfer.
    """
    
    def __init__(
        self,
        teacher_vocab_size: int,
        student_vocab_size: int,
        embed_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        """
        Initialize cross-tokenizer matcher.
        
        Args:
            teacher_vocab_size: Teacher tokenizer vocabulary size
            student_vocab_size: Student tokenizer vocabulary size
            embed_dim: Embedding dimension for token representations
            device: Device to place matchers on
            dtype: Data type for matchers
        """
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
        self.embed_dim = embed_dim
        
        # Token embedding tables
        self.teacher_token_embeds = nn.Embedding(
            teacher_vocab_size, embed_dim, device=device, dtype=dtype
        )
        self.student_token_embeds = nn.Embedding(
            student_vocab_size, embed_dim, device=device, dtype=dtype
        )
        
        # No massive match_score parameter to avoid CUDA OOM.
        
    def get_token_similarity(
        self,
        teacher_token_ids: Tensor,
        student_token_ids: Tensor,
    ) -> Tensor:
        """
        Get similarity scores between teacher and student token pairs.
        
        Args:
            teacher_token_ids: Teacher token IDs (batch, seq_len)
            student_token_ids: Student token IDs (batch, seq_len)
            
        Returns:
            Similarity scores (batch, seq_len)
        """
        # Get embeddings
        teacher_embeds = self.teacher_token_embeds(teacher_token_ids)
        student_embeds = self.student_token_embeds(student_token_ids)
        
        # Compute similarity via projection
        similarity = F.cosine_similarity(teacher_embeds, student_embeds, dim=-1)
        
        return similarity
    
    def match_tokens(
        self,
        teacher_tokens: Tensor,
        student_tokens: Tensor,
        teacher_mask: Optional[Tensor] = None,
        student_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Match teacher tokens to student tokens.
        
        Returns:
            Tuple of (matched_scores, matched_mask)
        """
        # Compute cross-token similarity matrix
        # teacher_embeds: (batch, T, embed_dim)
        # student_embeds: (batch, S, embed_dim)
        
        teacher_embeds = self.teacher_token_embeds(teacher_tokens)
        student_embeds = self.student_token_embeds(student_tokens)
        
        teacher_norm = F.normalize(teacher_embeds, p=2, dim=-1)
        student_norm = F.normalize(student_embeds, p=2, dim=-1)
        
        # Compute similarity matrix: (batch, T, S)
        similarity_matrix = torch.einsum(
            'bte,bse->bts', 
            teacher_norm, 
            student_norm
        )
        
        # Apply masks
        if teacher_mask is not None and student_mask is not None:
            # teacher_mask: (batch, T), student_mask: (batch, S)
            mask = torch.einsum('bt,bs->bts', teacher_mask.bool(), student_mask.bool())
            similarity_matrix = similarity_matrix.masked_fill(~mask, -1e9)
        
        # Find best matches
        # For each teacher token, find best student token
        max_scores, matched_indices = similarity_matrix.max(dim=-1)
        
        # Matched mask indicates valid matches
        matched_mask = max_scores > 0.0
        
        return max_scores, matched_mask
    
    def compute_matching_loss(
        self,
        teacher_tokens: Tensor,
        student_tokens: Tensor,
        teacher_mask: Optional[Tensor] = None,
        student_mask: Optional[Tensor] = None,
        alignment_scores: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute matching loss for cross-tokenizer alignment.
        
        Args:
            teacher_tokens: Teacher token IDs
            student_tokens: Student token IDs
            teacher_mask: Teacher attention mask
            student_mask: Student attention mask
            alignment_scores: Ground truth alignment scores (optional)
            
        Returns:
            Matching loss
        """
        # Get token embeddings
        teacher_embeds = self.teacher_token_embeds(teacher_tokens)
        student_embeds = self.student_token_embeds(student_tokens)
        
        teacher_norm = F.normalize(teacher_embeds, p=2, dim=-1)
        student_norm = F.normalize(student_embeds, p=2, dim=-1)
        
        # Compute similarity matrix: (batch, T, S)
        similarity_matrix = torch.einsum(
            'bte,bse->bts', 
            teacher_norm, 
            student_norm
        )
        
        # Apply masks
        if teacher_mask is not None and student_mask is not None:
            mask = torch.einsum('bt,bs->bts', teacher_mask.bool(), student_mask.bool())
            similarity_matrix = similarity_matrix.masked_fill(~mask, -1e9)
            
        # Find best matches
        max_scores, _ = similarity_matrix.max(dim=-1)
        
        if teacher_mask is not None:
            valid_mask = teacher_mask.to(max_scores.device)
            max_scores = max_scores * valid_mask
            mask_sum = valid_mask.sum()
            if mask_sum > 0:
                loss = 1.0 - (max_scores.sum() / mask_sum)
            else:
                loss = torch.tensor(0.0, device=max_scores.device)
        else:
            loss = 1.0 - max_scores.mean()
        
        return loss


class ProjectionGuidedDistiller:
    """
    Main distillation class using projection-guided cross-tokenizer alignment.
    
    Combines projection alignment with token matching to enable
    knowledge transfer between models with different tokenizer vocabularies.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_tokenizer: Any,
        student_tokenizer: Any,
        projection_type: str = "mlp",
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize projection-guided distiller.
        
        Args:
            teacher_model: Teacher model (frozen)
            student_model: Student model (trainable)
            teacher_tokenizer: Teacher tokenizer
            student_tokenizer: Student tokenizer
            projection_type: Projection type ("linear", "mlp", "identity")
            hidden_dim: Hidden dimension for MLP projection
            device: Device to run on
            dtype: Data type
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        
        # Get embedding dimensions
        teacher_embed_dim = self._get_embed_dim(teacher_model)
        student_embed_dim = self._get_embed_dim(student_model)
        
        logger.info(
            f"Projection-guided distiller initialized: "
            f"teacher_dim={teacher_embed_dim}, student_dim={student_embed_dim}"
        )
        
        # Projection aligner
        self.projection_aligner = ProjectionAligner(
            teacher_embed_dim=teacher_embed_dim,
            student_embed_dim=student_embed_dim,
            hidden_dim=hidden_dim,
            projection_type=projection_type,
            device=self.device,
            dtype=self.dtype,
        ).to(self.device)
        
        # Cross-tokenizer matcher
        actual_teacher_tok = getattr(teacher_tokenizer, "tokenizer", teacher_tokenizer)
        actual_student_tok = getattr(student_tokenizer, "tokenizer", student_tokenizer)
        self.token_matcher = CrossTokenizerMatcher(
            teacher_vocab_size=len(actual_teacher_tok),
            student_vocab_size=len(actual_student_tok),
            embed_dim=student_embed_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Store original models for forward passes
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
    def _get_embed_dim(self, model: nn.Module) -> int:
        """Extract embedding dimension from model."""
        config = getattr(model, "config", None)
        if config:
            for attr in ["hidden_size", "d_model", "n_embd", "dim", "hidden_dim"]:
                if hasattr(config, attr):
                    return getattr(config, attr)
        # Fallback to checking model embeddings directly
        for name, module in model.named_modules():
            if "embed_tokens" in name or "wte" in name:
                if hasattr(module, "embedding_dim"):
                    return module.embedding_dim
        return 768  # Default fallback
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        teacher_input_ids: Optional[Tensor] = None,
        teacher_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_logits: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with projection-guided distillation.
        
        Args:
            input_ids: Input token IDs (student vocabulary)
            attention_mask: Attention mask (student vocabulary)
            teacher_input_ids: Optional input token IDs for teacher (teacher vocabulary)
            teacher_attention_mask: Optional attention mask for teacher (teacher vocabulary)
            labels: Ground truth labels
            return_logits: Whether to return logits
            
        Returns:
            Dictionary with loss, logits, and alignment metrics
        """
        if teacher_input_ids is None:
            teacher_input_ids = input_ids
        if teacher_attention_mask is None:
            teacher_attention_mask = attention_mask
            
        # Get teacher embeddings (frozen)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                output_hidden_states=True,
            )
            teacher_hidden_states = teacher_outputs.hidden_states  # List of layer states
            
        # Get student embeddings with projection
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_hidden_states = student_outputs.hidden_states
        
        # Compute projection guidance loss
        alignment_loss = self._compute_alignment_loss(
            teacher_hidden_states,
            student_hidden_states,
            teacher_mask=teacher_attention_mask,
            student_mask=attention_mask,
        )
        
        # Compute matching loss
        matching_loss = self._compute_matching_loss(
            teacher_input_ids=teacher_input_ids,
            student_input_ids=input_ids,
            teacher_mask=teacher_attention_mask,
            student_mask=attention_mask,
        )
        
        # Total loss
        total_loss = alignment_loss + matching_loss
        
        # Add standard LM loss if labels provided
        if labels is not None and return_logits:
            logits = student_outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Weighted total loss
            total_loss = total_loss + 0.5 * ce_loss
            
        result = {
            "loss": total_loss,
            "alignment_loss": alignment_loss,
            "matching_loss": matching_loss,
        }
        
        if return_logits:
            result["logits"] = student_outputs.logits
            
        return result
    
    def _compute_alignment_loss(
        self,
        teacher_hidden_states: List[Tensor],
        student_hidden_states: List[Tensor],
        teacher_mask: Optional[Tensor] = None,
        student_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute alignment loss across layers.
        
        Aligns corresponding layers from teacher and student models.
        """
        layer_losses = []
        
        # Align each layer (or selected layers)
        num_layers = min(len(teacher_hidden_states), len(student_hidden_states))
        
        for layer_idx in range(num_layers):
            teacher_layer = teacher_hidden_states[layer_idx]
            student_layer = student_hidden_states[layer_idx]
            
            # Project teacher layer to student space
            # Query is student_layer (student sequence length). Key/Value is teacher_layer.
            # We use teacher_mask for the attention_mask arg because it corresponds to keys/values.
            projected = self.projection_aligner.forward(
                teacher_layer,
                student_embeddings=student_layer,
                attention_mask=teacher_mask,
            )
            
            # Alignment loss for this layer
            # Compare projected (which has student length now) to student_layer using student_mask.
            loss = self.projection_aligner.compute_alignment_loss(
                projected,
                student_layer,
                mask=student_mask,
            )
            layer_losses.append(loss)
        
        # Average across layers
        return torch.stack(layer_losses).mean()
    
    def _compute_matching_loss(
        self,
        teacher_input_ids: Tensor,
        student_input_ids: Tensor,
        teacher_mask: Optional[Tensor] = None,
        student_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute token matching loss."""
        return self.token_matcher.compute_matching_loss(
            teacher_tokens=teacher_input_ids,
            student_tokens=student_input_ids,
            teacher_mask=teacher_mask,
            student_mask=student_mask,
            alignment_scores=None,
        )
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters (student model + distiller modules)."""
        params = []
        for param in self.student_model.parameters():
            if param.requires_grad:
                params.append(param)
        for param in self.projection_aligner.parameters():
            params.append(param)
        for param in self.token_matcher.parameters():
            params.append(param)
        return params
    
    def train(self, mode: bool = True) -> "ProjectionGuidedDistiller":
        """Set training mode."""
        self.student_model.train(mode)
        self.projection_aligner.train(mode)
        self.token_matcher.train(mode)
        return self
    
    def eval(self) -> "ProjectionGuidedDistiller":
        """Set evaluation mode."""
        self.student_model.eval()
        self.projection_aligner.eval()
        self.token_matcher.eval()
        return self


def create_projection_aligner_from_config(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> ProjectionAligner:
    """
    Create projection aligner from configuration dictionary.
    
    Args:
        config: Configuration dictionary with projection settings
        device: Device to create on
        
    Returns:
        Initialized ProjectionAligner
    """
    return ProjectionAligner(
        teacher_embed_dim=config["teacher_embed_dim"],
        student_embed_dim=config["student_embed_dim"],
        hidden_dim=config.get("hidden_dim"),
        projection_type=config.get("projection_type", "linear"),
        init_scale=config.get("init_scale", 0.02),
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
