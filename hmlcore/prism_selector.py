> By dreamraster · dreaMSCend
"""
hmlcore/prism_selector.py
=========================
Implementation of PRISM (Self-Pruning Intrinsic Selection Method) for text.

Steps:
1. Extract embeddings: Use model's hidden states, pooled across sequence.
2. Re-centering: Subtract the global mean from all embeddings (drift correction).
3. Correlation: Compute pairwise similarity (dot-product).
4. Pruning: Sum the correlation rows and split into quality tiers.
"""

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from datasets import Dataset

logger = logging.getLogger(__name__)

@torch.no_grad()
def select_with_prism(
    dataset: Dataset,
    model,
    tokenizer,
    tier: str = "high",
    layer: int = -1,
    batch_size: int = 16,
    cache_path: str = None,
    chunk_size: int = 2000,
) -> Dataset:
    """
    Filters the dataset using the PRISM algorithm.
    Returns a new Dataset containing only samples from the requested tier.
    """
    if len(dataset) == 0:
        return dataset

    # 1. Extraction (or Load from cache)
    embeddings = None
    if cache_path and os.path.exists(cache_path):
        logger.info(f"PRISM: Loading cached embeddings from {cache_path}")
        try:
            cached_data = torch.load(cache_path, weights_only=True)
            if len(cached_data) == len(dataset):
                embeddings = cached_data
            else:
                logger.warning("PRISM: Cache size mismatch. Re-calculating.")
        except Exception as e:
            logger.warning(f"PRISM: Failed to load cache: {e}")

    if embeddings is None:
        embeddings = _extract_embeddings(dataset, model, tokenizer, layer, batch_size)
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(embeddings, cache_path)
                logger.info(f"PRISM: Saved embeddings to {cache_path}")
            except Exception as e:
                logger.warning(f"PRISM: Failed to save cache: {e}")

    # 2. Implicit Re-centering (Global Semantic Drift correction)
    # PRISM uses mean-subtraction to model intrinsic visual (textual) semantics.
    mean_vec = embeddings.mean(dim=0, keepdim=True)
    embeddings = embeddings - mean_vec
    
    # Normalize for cosine similarity (opt-in, PRISM paper often uses dot product on centered data)
    # We follow the paper's dot product on centered data approach.

    # 3. Correlation & Row-sum (Chunked for memory efficiency)
    num_samples = embeddings.size(0)
    row_sums = torch.zeros(num_samples, device=embeddings.device)
    
    logger.info(f"PRISM: Computing correlation matrix and row sums (N={num_samples})")
    for i in range(0, num_samples, chunk_size):
        end_i = min(i + chunk_size, num_samples)
        chunk_i = embeddings[i:end_i] # [chunk_size, hidden_dim]
        
        # Multiply chunk with the entire embedding matrix
        # Matrix multiply [chunk_size, hidden_dim] @ [hidden_dim, num_samples]
        corr_chunk = torch.matmul(chunk_i, embeddings.T) # [chunk_size, num_samples]
        
        row_sums[i:end_i] = corr_chunk.sum(dim=1)

    # 4. Tier Splitting (based on row sums)
    # High row sum = highly redundant = low quality (according to PRISM terminology)
    # Actually, the paper says "high-quality instruction data typically exhibit low redundancy"
    # So: Low row_sum -> High Quality Tier.
    
    # Normalize scores for quantile calculation
    min_val = row_sums.min()
    max_val = row_sums.max()
    norm_scores = (row_sums - min_val) / (max_val - min_val + 1e-9)

    # Calculate quantiles
    q1 = torch.quantile(norm_scores, 0.333).item()
    q2 = torch.quantile(norm_scores, 0.666).item()

    logger.info(f"PRISM: Score thresholds - Q1 (33%): {q1:.4f}, Q2 (66%): {q2:.4f}")
    
    # Determine indices for the requested tier
    keep_indices = []
    dropped_indices = []
    
    for idx, score in enumerate(norm_scores.tolist()):
        is_in_tier = False
        if tier == "high": # Most unique / Lowest redundancy
            if score <= q1: is_in_tier = True
        elif tier == "mid":
            if q1 < score <= q2: is_in_tier = True
        elif tier == "low": # Most redundant
            if score > q2: is_in_tier = True
        elif tier == "high+mid":
            if score <= q2: is_in_tier = True
        else:
            if score <= q1: is_in_tier = True

        if is_in_tier:
            keep_indices.append((idx, score))
        else:
            dropped_indices.append((idx, score))

    # Log examples of "most redundant" samples being excluded if we are in 'high' mode
    if tier == "high" and dropped_indices:
        # Sort dropped by score descending to get 'most redundant'
        dropped_indices.sort(key=lambda x: x[1], reverse=True)
        logger.info("PRISM: Examples of redundant/low-info samples being excluded:")
        for i in range(min(3, len(dropped_indices))):
            idx, score = dropped_indices[i]
            prompt_snip = str(dataset[idx]["prompt"])[:80].replace("\n", " ")
            logger.info(f"  - Dropped (score {score:.4f}): \"{prompt_snip}...\"")

    # Sort kept indices by score ascending (lowest redundancy first)
    keep_indices.sort(key=lambda x: x[1])
    indices = [x[0] for x in keep_indices]

    logger.info(f"PRISM: Selected {len(indices)} samples for tier '{tier}' (sorted by redundancy)")
    
    # Return filtered dataset
    return dataset.select(indices)

def _extract_embeddings(dataset, model, tokenizer, layer, batch_size):
    """Internal helper to extract pooled hidden states from the model."""
    model.eval()
    device = next(model.parameters()).device
    all_embeddings = []
    
    logger.info(
        f"PRISM: Extracting embeddings (layer={layer}, batch={batch_size}) "
        f"for {len(dataset)} samples on {device}..."
    )
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="PRISM Extraction"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # Use the 'prompt' column as per plan
        prompts = [x["prompt"] for x in batch]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states is a tuple of [batch, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states[layer]
            
            # Pool across sequence dimension (mean pooling, ignoring padding)
            # Create mask from attention_mask
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            all_embeddings.append(pooled.cpu()) # Keep on CPU until correlation step if needed
            
    return torch.cat(all_embeddings, dim=0)
