# By dreamraster · dreaMSCend
"""
hmlcore/prism_dq.py
===================
PRISM Dynamic Quantization (PRISM-DQ) engine.

Analyzes each weight tensor using 7 structural metrics and then runs a
Lagrangian allocator to assign per-tensor-class quantization types that
minimise total quantization distortion within a target bits-per-weight
(BPW) budget.

The output is a llama-quantize recipe: a mapping from GGUF tensor-name
patterns to GGUF quantization types (Q2_K, Q3_K, Q4_K, IQ4_XS, Q6_K).
This recipe is then written as a ready-to-run `llama-quantize` shell
command and optionally auto-invoked.

Metric Reference (Ex0bit / PRISM-DQ specification)
---------------------------------------------------
1. PL-Alpha-Hill     — Spectral heavy-tail index via Hill estimator on
                       eigenvalues of W^T W (NOT raw singular values).
2. Spectral Dominance— sigma_1 / sum(sigma_i): rank-1 approximation quality.
3. OSQE             — Optimal Scale Quantization Error at 2, 3, 4, 6 bits.
4. Matrix Imbalance  — max(CV_rows, CV_cols): coefficient of variation.
5. Fragility         — log(OSQE_2bit / OSQE_4bit): sensitivity to low bits.
6. Boundary Density  — Fraction of weights near quantization bin boundaries.
7. Spectral Position Prior — Bidirectional spectral norm product encoding
                             the layer's position within the model depth.

Public API
----------
run_prism_dq(model, args, finale_dir) -> dict[str, str]
    Returns the tensor-class → GGUF-quant-type recipe.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GGUF type → approximate bits-per-weight
# ─────────────────────────────────────────────────────────────────────────────

_QUANT_BPW: Dict[str, float] = {
    "Q2_K": 2.96,
    "Q3_K": 3.44,  # alias for Q3_K_M in llama-quantize
    "Q4_K": 4.58,  # alias for Q4_K_M
    "IQ4_XS": 4.25,
    "Q5_K": 5.33,
    "Q6_K": 6.14,
    "Q8_0": 8.00,
}

# Ordered candidate quantization types from lowest to highest quality
# Used by the Lagrangian allocator to pick the right level per tensor class.
_CANDIDATE_QUANTS: List[str] = ["Q2_K", "Q3_K", "Q4_K", "IQ4_XS", "Q5_K", "Q6_K"]

# ─────────────────────────────────────────────────────────────────────────────
# GGUF tensor name mapping: PyTorch attribute suffix → GGUF tensor-type key
# ─────────────────────────────────────────────────────────────────────────────

_GGUF_CLASS_MAP: Dict[str, str] = {
    # Standard LLaMA / Qwen / Mistral / Gemma naming
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    # Phi-3 / Phi-3.5
    "self_attn.qkv_proj": "attn_qkv",
    "mlp.fc1": "ffn_up",
    "mlp.fc2": "ffn_down",
    # Mamba / SSM hybrids (Qwen3.5)
    "mixer.in_proj": "ssm_alpha",
    "mixer.out_proj": "ssm_out",
    "mixer.x_proj": "ssm_beta",
    # Embedding
    "embed_tokens": "token_embd",
    "lm_head": "output",
}


def _tensor_class(name: str) -> str:
    """Map a PyTorch parameter name to a GGUF tensor class string."""
    for suffix, class_name in _GGUF_CLASS_MAP.items():
        if name.endswith(suffix + ".weight") or suffix in name:
            return class_name
    if "embed_tokens" in name:
        return "token_embd"
    if "lm_head" in name:
        return "output"
    return "other"


def _block_index(name: str) -> Optional[int]:
    """Extract the transformer block index from a parameter name."""
    m = re.search(r"(?:layers|blocks)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_float_cpu(w: torch.Tensor) -> torch.Tensor:
    """Upcast to float32 on CPU for reliable numeric computation."""
    return w.detach().float().cpu()


def _compute_singular_values(w: torch.Tensor) -> torch.Tensor:
    """Compute singular values of a 2-D weight matrix in float32."""
    w2 = w if w.ndim == 2 else w.flatten(1)
    # svdvals is faster than full SVD
    try:
        return torch.linalg.svdvals(w2)
    except Exception:
        # Fallback for older torch versions
        return torch.svd(w2, compute_uv=False).S


# ── Metric 1: PL-Alpha-Hill ───────────────────────────────────────────────────


def _pl_alpha_hill(w: torch.Tensor, tail_fraction: float = 0.15) -> float:
    """
    Spectral heavy-tail index via Hill estimator on eigenvalues of W^T W.

    Steps:
      1. Compute singular values sigma_i.
      2. Square them to get eigenvalues lambda_i = sigma_i^2 of W^T W.
      3. Sort descending, take the top `tail_fraction`.
      4. Apply Hill estimator: alpha_hat = (1/k * sum_i log(lambda_i / lambda_{k+1}))^-1

    Lower alpha → heavier tail → more training signal → higher sensitivity.
    """
    sigma = _compute_singular_values(w)
    lambdas = (sigma**2).sort(descending=True).values

    n = len(lambdas)
    k = max(2, int(n * tail_fraction))
    if k >= n:
        k = n - 1

    # Clamp to avoid log(0)
    eps = 1e-10
    tail = lambdas[:k].clamp(min=eps)
    pivot = lambdas[k].clamp(min=eps)

    log_ratios = torch.log(tail / pivot)
    alpha_hat = 1.0 / (log_ratios.mean().item() + eps)
    return float(alpha_hat)


# ── Metric 2: Spectral Dominance ─────────────────────────────────────────────


def _spectral_dominance(w: torch.Tensor) -> float:
    """
    Ratio of the top singular value to the sum of all singular values.
    Range: (0, 1]. Higher → more rank-1-like → potentially more sensitive.
    """
    sigma = _compute_singular_values(w)
    total = sigma.sum().item()
    if total < 1e-12:
        return 0.0
    return float(sigma[0].item() / total)


# ── Metric 3: OSQE — Optimal Scale Quantization Error ────────────────────────


def _optimal_scale_qe(w: torch.Tensor, bits: int) -> float:
    """
    Quantization MSE using the optimal (min-max) symmetric scale for `bits`.

    The optimal scale minimises MSE for uniform symmetric quantization:
        scale = max(|w|) / (2^(bits-1) - 1)
        w_q   = round(w / scale).clamp(-qmax, qmax) * scale
        OSQE  = mean((w - w_q)^2)
    """
    qmax = (2 ** (bits - 1)) - 1
    abs_max = w.abs().max()
    if abs_max < 1e-10:
        return 0.0
    scale = abs_max / qmax
    w_q = (w / scale).round().clamp(-qmax, qmax) * scale
    return float(((w - w_q) ** 2).mean().item())


def _osqe_vector(w: torch.Tensor) -> Dict[str, float]:
    """Compute OSQE at bit levels 2, 3, 4, 6 as per PRISM-DQ spec."""
    return {
        "osqe_2": _optimal_scale_qe(w, 2),
        "osqe_3": _optimal_scale_qe(w, 3),
        "osqe_4": _optimal_scale_qe(w, 4),
        "osqe_6": _optimal_scale_qe(w, 6),
    }


# ── Metric 4: Matrix Imbalance ───────────────────────────────────────────────


def _matrix_imbalance(w: torch.Tensor) -> float:
    """
    Max of the coefficient of variation (std/|mean|) across rows and columns.

    High imbalance → non-uniform weight distribution → sensitive to quant.
    """
    w2 = w if w.ndim == 2 else w.flatten(1)
    eps = 1e-10

    row_mean = w2.mean(dim=1).abs().clamp(min=eps)
    row_std = w2.std(dim=1)
    col_mean = w2.mean(dim=0).abs().clamp(min=eps)
    col_std = w2.std(dim=0)

    cv_rows = (row_std / row_mean).max().item()
    cv_cols = (col_std / col_mean).max().item()
    return float(max(cv_rows, cv_cols))


# ── Metric 5: Fragility ───────────────────────────────────────────────────────


def _fragility(osqe_2: float, osqe_4: float) -> float:
    """
    Log-ratio of 2-bit vs 4-bit quantization error.

    High fragility → quality degrades sharply as bits drop → needs more bits.
    """
    eps = 1e-10
    return float(math.log((osqe_2 + eps) / (osqe_4 + eps)))


# ── Metric 6: Boundary Density ───────────────────────────────────────────────


def _boundary_density(w: torch.Tensor, bits: int = 4, margin: float = 0.1) -> float:
    """
    Fraction of weights within `margin` of a quantization bin boundary.

    Weights near boundaries are most at risk of being quantized to the wrong
    bin → higher boundary density → higher sensitivity.

    `margin` is expressed as a fraction of the bin width (0.1 = 10%).
    """
    qmax = (2 ** (bits - 1)) - 1
    abs_max = w.abs().max()
    if abs_max < 1e-10:
        return 0.0
    scale = abs_max / qmax
    # Fractional positions within bins
    frac_pos = (w / scale) - (w / scale).round()
    # Values within margin of 0.5 (i.e., near a boundary)
    near_boundary = frac_pos.abs() > (0.5 - margin)
    return float(near_boundary.float().mean().item())


# ── Metric 7: Spectral Position Prior ────────────────────────────────────────


def _spectral_position_prior(
    layer_idx: int,
    total_layers: int,
    spectral_norms: List[float],
) -> float:
    """
    Bidirectional spectral norm product encoding the layer's depth position.

    Computes the product of the spectral norm of layer l and the spectral
    norm of the mirrored layer (total_layers - 1 - l), normalised by the
    max product in the model.  This creates a U-shaped prior that assigns
    higher sensitivity to early and late layers.

    Returns a value in [0, 1].
    """
    if total_layers == 0 or not spectral_norms:
        return 0.5

    mirror_idx = total_layers - 1 - layer_idx
    mirror_idx = max(0, min(mirror_idx, len(spectral_norms) - 1))
    own_idx = min(layer_idx, len(spectral_norms) - 1)

    product = spectral_norms[own_idx] * spectral_norms[mirror_idx]

    # Normalise by the max bidirectional product in the entire model
    max_product = max(
        spectral_norms[i] * spectral_norms[max(0, total_layers - 1 - i)]
        for i in range(min(total_layers, len(spectral_norms)))
    )
    if max_product < 1e-10:
        return 0.5
    return float(product / max_product)


# ─────────────────────────────────────────────────────────────────────────────
# Per-tensor metrics bundle
# ─────────────────────────────────────────────────────────────────────────────


def _compute_all_metrics(
    w: torch.Tensor,
    layer_idx: int,
    total_layers: int,
    spectral_norms: List[float],
) -> Dict[str, float]:
    """Compute all 7 PRISM-DQ structural metrics for one weight tensor."""
    w_f = _to_float_cpu(w)
    osqe = _osqe_vector(w_f)

    return {
        "pl_alpha_hill": _pl_alpha_hill(w_f),
        "spectral_dominance": _spectral_dominance(w_f),
        **osqe,
        "matrix_imbalance": _matrix_imbalance(w_f),
        "fragility": _fragility(osqe["osqe_2"], osqe["osqe_4"]),
        "boundary_density": _boundary_density(w_f),
        "spectral_pos_prior": _spectral_position_prior(
            layer_idx, total_layers, spectral_norms
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite sensitivity score
# ─────────────────────────────────────────────────────────────────────────────

# Weights for combining the 7 normalised metrics into a single sensitivity score.
# Spectral tail behaviour and quantisation error proxy are tied at maximum weight
# (empirical default — an ablation study would validate relative importance).
# Position prior adds context at minimum weight.
_METRIC_WEIGHTS = {
    "pl_alpha_hill": 0.20,  # Spectral tail behaviour (tied highest — empirical)
    "spectral_dominance": 0.15,
    "osqe_4": 0.20,  # Quantisation error proxy (tied highest — empirical)
    "matrix_imbalance": 0.15,
    "fragility": 0.15,
    "boundary_density": 0.10,
    "spectral_pos_prior": 0.05,
}


def _composite_sensitivity(metrics: Dict[str, float], model_stats: Dict) -> float:
    """
    Combine normalised metrics into a sensitivity score in [0, 1].

    Each metric is normalised by the model-wide [min, max] range so that they
    all contribute on the same scale regardless of absolute magnitude.
    """
    score = 0.0
    for key, weight in _METRIC_WEIGHTS.items():
        val = metrics.get(key, 0.0)
        mn = model_stats.get(f"{key}_min", 0.0)
        mx = model_stats.get(f"{key}_max", 1.0)
        rng = mx - mn
        norm = (val - mn) / rng if rng > 1e-10 else 0.0
        # PL-Alpha-Hill: lower is MORE sensitive → invert
        if key == "pl_alpha_hill":
            norm = 1.0 - norm
        score += weight * max(0.0, min(1.0, norm))
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Lagrangian bit allocator
# ─────────────────────────────────────────────────────────────────────────────


def _quant_for_sensitivity(sensitivity: float, lambda_: float) -> str:
    """
    Select the GGUF quantization type for a given sensitivity and λ.

    The Lagrangian formulation: for a tensor class with sensitivity s,
    choose the quant type q* that minimises:
        OSQE(q) * s + lambda * BPW(q)

    For implementation simplicity, we directly map sensitivity thresholds
    (adjusted by λ) to quant types.  Higher λ → tighter budget → lower bits.
    """
    # Sigmoid-based soft thresholds prevent hard zero-clipping when λ is large.
    # At λ=5 the sigmoid saturates (~0.993), thresholds compress to ~20 % of
    # their base values, but never reach zero — discriminative power is preserved
    # across all quantisation levels.
    base_thresholds = [
        ("Q6_K", 0.85),
        ("Q5_K", 0.70),
        ("Q4_K", 0.50),
        ("IQ4_XS", 0.35),
        ("Q3_K", 0.20),
    ]

    lambda_factor = torch.sigmoid(torch.tensor(lambda_))  # [0, 1]

    if sensitivity >= base_thresholds[0][1] * (1.0 - lambda_factor * 0.8):
        return "Q6_K"
    for qt, base in base_thresholds[1:]:
        if sensitivity >= base * (1.0 - lambda_factor * 0.8):
            return qt
    return "Q2_K"


def _lagrangian_allocate(
    sensitivities: Dict[str, float],
    num_params: Dict[str, int],
    target_bpw: float,
    lambda_lo: float = -2.0,
    lambda_hi: float = 5.0,
    max_iter: int = 40,
    tol: float = 0.01,
) -> Dict[str, str]:
    """
    Binary search on λ to find the allocation that meets target_bpw.

    Higher λ → tighter budget (lower bits assigned).
    Lower λ  → looser budget (higher bits assigned).

    Args:
        sensitivities: {tensor_class → sensitivity_score}
        num_params:    {tensor_class → total parameter count}
        target_bpw:    Target average bits per weight.

    Returns:
        {tensor_class → GGUF quant type}
    """
    total_params = sum(num_params.values())
    if total_params == 0:
        return {cls: "Q4_K" for cls in sensitivities}

    def compute_bpw(lambda_: float) -> Tuple[float, Dict[str, str]]:
        plan: Dict[str, str] = {}
        weighted_bits = 0.0
        for cls, sens in sensitivities.items():
            qt = _quant_for_sensitivity(sens, lambda_)
            plan[cls] = qt
            weighted_bits += _QUANT_BPW[qt] * num_params.get(cls, 0)
        bpw = weighted_bits / total_params
        return bpw, plan

    # Check if we can satisfy the budget at all
    bpw_lo, plan_lo = compute_bpw(lambda_hi)
    bpw_hi, plan_hi = compute_bpw(lambda_lo)

    if target_bpw <= bpw_lo:
        logger.warning(
            "PRISM-DQ: target_bpw=%.2f is lower than the minimum achievable "
            "BPW=%.2f at max compression. Returning most aggressive allocation.",
            target_bpw,
            bpw_lo,
        )
        return plan_lo

    if target_bpw >= bpw_hi:
        logger.info(
            "PRISM-DQ: target_bpw=%.2f is higher than the maximum BPW=%.2f. "
            "Returning highest quality allocation.",
            target_bpw,
            bpw_hi,
        )
        return plan_hi

    # Binary search
    for _ in range(max_iter):
        lambda_mid = (lambda_lo + lambda_hi) / 2.0
        bpw_mid, plan_mid = compute_bpw(lambda_mid)

        if abs(bpw_mid - target_bpw) < tol:
            break

        if bpw_mid > target_bpw:
            lambda_lo = lambda_mid  # Need more compression
        else:
            lambda_hi = lambda_mid  # Too aggressive, relax

    return plan_mid


# ─────────────────────────────────────────────────────────────────────────────
# Per-block refinement pass
# ─────────────────────────────────────────────────────────────────────────────


def _per_block_refinement(
    block_metrics: Dict[
        str,  # "blk.N.tensor_class" → metrics dict
        Dict[str, float],
    ],
    class_plan: Dict[str, str],
    class_sensitivities: Dict[str, float],
    refinement_threshold: float = 1.5,
) -> Dict[str, str]:
    """
    Identify individual blocks whose OSQE at the class-level assignment is
    significantly worse than the class mean, and bump them up one quant level.

    Args:
        block_metrics:          Metrics for each specific (block, tensor_class) pair.
        class_plan:             Global class → quant type plan from Lagrangian allocator.
        class_sensitivities:    Global sensitivity score per class.
        refinement_threshold:   Ratio of block OSQE to class-mean OSQE that triggers an upgrade.

    Returns:
        Overrides dict: {"blk.N.tensor_class" → upgraded_quant_type}
    """
    overrides: Dict[str, str] = {}

    # Compute class-level mean OSQE_4 for comparison baseline
    class_osqe_mean: Dict[str, List[float]] = {}
    for key, m in block_metrics.items():
        cls = key.rsplit(".", 1)[-1]
        class_osqe_mean.setdefault(cls, []).append(m.get("osqe_4", 0.0))

    class_mean_osqe: Dict[str, float] = {
        cls: (sum(vals) / len(vals)) if vals else 0.0
        for cls, vals in class_osqe_mean.items()
    }

    # Upgrade candidate quant index
    _quant_order = ["Q2_K", "Q3_K", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

    for key, m in block_metrics.items():
        cls = key.rsplit(".", 1)[-1]
        assigned_quant = class_plan.get(cls, "Q4_K")
        mean_osqe = class_mean_osqe.get(cls, 0.0)
        block_osqe = m.get("osqe_4", 0.0)

        # Trigger refinement if this block's error is significantly above the class mean
        if mean_osqe > 1e-10 and block_osqe > refinement_threshold * mean_osqe:
            # Upgrade one step
            if assigned_quant in _quant_order:
                idx = _quant_order.index(assigned_quant)
                if idx + 1 < len(_quant_order):
                    upgraded = _quant_order[idx + 1]
                    overrides[key] = upgraded
                    logger.debug(
                        "  PRISM-DQ refinement: %s upgraded %s → %s "
                        "(block_osqe=%.4f, class_mean=%.4f)",
                        key,
                        assigned_quant,
                        upgraded,
                        block_osqe,
                        mean_osqe,
                    )

    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_prism_dq(
    model: "torch.nn.Module",
    target_bpw: float = 4.0,
    dq_refinement: bool = False,
    output_dir: Optional[str] = None,
    llama_quantize_path: Optional[str] = None,
    input_gguf: Optional[str] = None,
    base_quant: str = "Q3_K",
) -> Dict[str, str]:
    """
    Run the full PRISM-DQ analysis and emit a llama-quantize recipe.

    Args:
        model:               A plain BF16 HF model (not PeftModel, not BnB).
        target_bpw:          Target average bits-per-weight budget.
        dq_refinement:       Enable per-block refinement pass.
        output_dir:          Directory to save the recipe script.
        llama_quantize_path: Path to the llama-quantize binary. If set, the
                             recipe is auto-invoked after generation.
        input_gguf:          Path to an F16 GGUF to be quantized. Required
                             when auto-invoking llama-quantize.
        base_quant:          Default quantization type for unclassified tensors.

    Returns:
        {tensor_class → GGUF quant type} mapping (the "recipe").
    """
    logger.info("━" * 60)
    logger.info("  PRISM Dynamic Quantization — Analysis")
    logger.info("━" * 60)
    logger.info("  Target BPW: %.2f", target_bpw)
    logger.info("  Refinement: %s", "enabled" if dq_refinement else "disabled")

    # ── Step 0: Collect all Linear weight tensors ─────────────────────────────
    named_linears: List[Tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None:
            named_linears.append((name, module.weight.data))

    if not named_linears:
        logger.error("PRISM-DQ: No nn.Linear layers found in model. Aborting.")
        return {}

    logger.info("  Found %d Linear layers to analyze.", len(named_linears))

    # ── Step 1: Collect spectral norms per block for Position Prior ───────────
    # Group by block index and take the max spectral norm within each block.
    block_norms_raw: Dict[int, List[float]] = {}
    for name, w in named_linears:
        bidx = _block_index(name)
        if bidx is not None:
            sigma = _compute_singular_values(_to_float_cpu(w))
            block_norms_raw.setdefault(bidx, []).append(float(sigma[0].item()))

    # One spectral norm per block (max across all tensors in that block)
    block_indices_sorted = sorted(block_norms_raw.keys())
    total_layers = max(block_indices_sorted) + 1 if block_indices_sorted else 1
    spectral_norms_by_block = [
        max(block_norms_raw.get(i, [0.0])) for i in range(total_layers)
    ]

    logger.info("  Detected %d transformer blocks.", total_layers)

    # ── Step 2: Compute all 7 metrics per layer ───────────────────────────────
    logger.info("  Computing structural metrics (this may take a moment) ...")

    all_metrics: Dict[str, Dict[str, float]] = {}  # param_name → metrics
    for name, w in named_linears:
        bidx = _block_index(name) or 0
        all_metrics[name] = _compute_all_metrics(
            w, bidx, total_layers, spectral_norms_by_block
        )

    # ── Step 3: Model-wide normalisation stats ────────────────────────────────
    model_stats: Dict[str, float] = {}
    for metric_key in list(_METRIC_WEIGHTS.keys()):
        vals = [m[metric_key] for m in all_metrics.values() if metric_key in m]
        if vals:
            model_stats[f"{metric_key}_min"] = min(vals)
            model_stats[f"{metric_key}_max"] = max(vals)

    # ── Step 4: Compute composite sensitivity per tensor + aggregate per class ─
    class_sensitivities_raw: Dict[str, List[float]] = {}
    class_param_counts: Dict[str, int] = {}
    block_metrics_flat: Dict[str, Dict[str, float]] = {}  # for refinement

    for name, m in all_metrics.items():
        cls = _tensor_class(name)
        sens = _composite_sensitivity(m, model_stats)
        class_sensitivities_raw.setdefault(cls, []).append(sens)

        # Count params for BPW weighting
        bidx = _block_index(name)
        w_shape = next(
            (
                mod.weight.shape
                for n2, mod in model.named_modules()
                if n2 == name and hasattr(mod, "weight")
            ),
            None,
        )
        # Fallback: try to get shape from the collected data
        for n2, w in named_linears:
            if n2 == name:
                class_param_counts[cls] = class_param_counts.get(cls, 0) + w.numel()
                break

        if bidx is not None:
            block_key = f"blk.{bidx}.{cls}"
            if block_key not in block_metrics_flat:
                block_metrics_flat[block_key] = m
            else:
                # Merge metrics: take mean for multi-tensor blocks
                for k, v in m.items():
                    block_metrics_flat[block_key][k] = (
                        block_metrics_flat[block_key][k] + v
                    ) / 2.0

    # Mean sensitivity per class
    class_sensitivities: Dict[str, float] = {
        cls: sum(vals) / len(vals) for cls, vals in class_sensitivities_raw.items()
    }

    # ── Step 5: Lagrangian allocator ──────────────────────────────────────────
    logger.info("  Running Lagrangian allocator (target_bpw=%.2f) ...", target_bpw)
    class_plan = _lagrangian_allocate(
        class_sensitivities, class_param_counts, target_bpw
    )

    # ── Step 6: Per-block refinement (optional) ───────────────────────────────
    block_overrides: Dict[str, str] = {}
    if dq_refinement:
        logger.info("  Running per-block refinement pass ...")
        block_overrides = _per_block_refinement(
            block_metrics_flat, class_plan, class_sensitivities
        )
        if block_overrides:
            logger.info(
                "  Refinement: %d block-level override(s) applied.",
                len(block_overrides),
            )

    # ── Step 7: Calculate achieved BPW and log results ────────────────────────
    total_p = sum(class_param_counts.values())
    achieved_bpw = 0.0
    if total_p > 0:
        for cls, qt in class_plan.items():
            achieved_bpw += _QUANT_BPW.get(qt, 4.0) * class_param_counts.get(cls, 0)
        achieved_bpw /= total_p

    logger.info("")
    logger.info("  ── PRISM-DQ Bit Allocation ──")
    logger.info(
        "  %-20s  %-12s  %-8s  %s", "Tensor Class", "Quant Type", "BPW", "Sensitivity"
    )
    logger.info("  %s", "─" * 58)
    for cls in sorted(class_plan.keys()):
        qt = class_plan[cls]
        sens = class_sensitivities.get(cls, 0.0)
        bpw = _QUANT_BPW.get(qt, 4.0)
        logger.info("  %-20s  %-12s  %-8.2f  %.3f", cls, qt, bpw, sens)

    if block_overrides:
        logger.info("")
        logger.info("  ── Per-Block Overrides ──")
        for key, qt in sorted(block_overrides.items()):
            logger.info("  %-40s → %s", key, qt)

    logger.info("")
    logger.info(
        "  Target BPW: %.2f  |  Achieved BPW: %.2f",
        target_bpw,
        achieved_bpw,
    )
    logger.info("━" * 60)

    # ── Step 8: Emit llama-quantize recipe ────────────────────────────────────
    recipe = _emit_recipe(
        class_plan=class_plan,
        block_overrides=block_overrides,
        base_quant=base_quant,
        output_dir=output_dir,
        input_gguf=input_gguf,
        llama_quantize_path=llama_quantize_path,
    )

    return recipe


# ─────────────────────────────────────────────────────────────────────────────
# Recipe emission + auto-invocation
# ─────────────────────────────────────────────────────────────────────────────


def _emit_recipe(
    class_plan: Dict[str, str],
    block_overrides: Dict[str, str],
    base_quant: str,
    output_dir: Optional[str],
    input_gguf: Optional[str],
    llama_quantize_path: Optional[str],
) -> Dict[str, str]:
    """
    Build and log the llama-quantize command, write it to a .sh script,
    and optionally invoke llama-quantize automatically.

    The output GGUF path is derived from input_gguf or output_dir.

    Returns the class_plan dict (not modified).
    """
    # Build --tensor-type flags from class plan
    tensor_flags: List[str] = []
    for cls, qt in sorted(class_plan.items()):
        tensor_flags.append(f'--tensor-type "{cls}={qt}"')

    # Per-block override flags (override the class-level assignment)
    # Format: --tensor-type "blk.(N).tensor_class.weight=TYPE"
    for block_key, qt in sorted(block_overrides.items()):
        # Convert "blk.18.attn_q" → regex "blk\.(18)\.attn_q"
        parts = block_key.split(".")
        if len(parts) >= 3:
            idx = parts[1]
            cls = ".".join(parts[2:])
            tensor_flags.append(f'--tensor-type "blk\\.({idx})\\.{cls}={qt}"')

    # Determine output GGUF path
    if input_gguf:
        in_dir = os.path.dirname(input_gguf)
        in_base = os.path.splitext(os.path.basename(input_gguf))[0]
        output_gguf = os.path.join(output_dir or in_dir, f"{in_base}-PRISM-DQ.gguf")
    else:
        output_gguf = os.path.join(output_dir or ".", "model_PRISM-DQ.gguf")
        input_gguf = os.path.join(output_dir or ".", "model_f16.gguf")

    lq_bin = llama_quantize_path or "llama-quantize"

    flags_str = " \\\n    ".join(tensor_flags)
    cmd = (
        f"{lq_bin} \\\n"
        f"    {flags_str} \\\n"
        f"    {input_gguf} \\\n"
        f"    {output_gguf} \\\n"
        f"    {base_quant}"
    )

    logger.info("")
    logger.info("━" * 60)
    logger.info("  ⚡ PRISM-DQ — llama-quantize Recipe")
    logger.info("━" * 60)
    logger.info("")
    logger.info(
        "  If you haven't already, convert your BF16 HF checkpoint to F16 GGUF:\n"
        "    python convert_hf_to_gguf.py <finale_dir> --outtype f16 --outfile %s",
        input_gguf,
    )
    logger.info("")
    logger.info("  Then run the following command to apply PRISM-DQ allocation:")
    logger.info("")
    logger.info("  %s", cmd)
    logger.info("")
    logger.info(
        "  Output: %s  (base type: %s for unclassified tensors)",
        output_gguf,
        base_quant,
    )
    logger.info("━" * 60)

    # Write to script file
    if output_dir:
        script_path = os.path.join(output_dir, "prism_dq_recipe.sh")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write("# PRISM-DQ recipe — generated by hmlcore/prism_dq.py\n")
                f.write(f"# Target BPW: auto-allocated\n")
                f.write(f"# Base quant: {base_quant}\n\n")
                f.write(cmd)
                f.write("\n")
            logger.info("  Recipe saved → %s", script_path)
        except Exception as exc:
            logger.warning("  Could not write recipe script: %s", exc)

    # Auto-invoke llama-quantize if binary is provided
    if llama_quantize_path and os.path.isfile(llama_quantize_path):
        if not input_gguf or not os.path.isfile(input_gguf):
            logger.warning(
                "  PRISM-DQ: --dq_llama_path is set but input GGUF not found at %s. "
                "Skipping auto-invocation. Run the recipe manually.",
                input_gguf,
            )
        else:
            logger.info("  Auto-invoking llama-quantize ...")
            try:
                # Build the actual subprocess args list
                lq_args = [llama_quantize_path]
                for cls, qt in sorted(class_plan.items()):
                    lq_args += [f"--tensor-type", f"{cls}={qt}"]
                for block_key, qt in sorted(block_overrides.items()):
                    parts = block_key.split(".")
                    if len(parts) >= 3:
                        idx = parts[1]
                        cls = ".".join(parts[2:])
                        lq_args += [f"--tensor-type", f"blk\\.({idx})\\.{cls}={qt}"]
                lq_args += [input_gguf, output_gguf, base_quant]

                result = subprocess.run(
                    lq_args,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info("  ✅ llama-quantize complete → %s", output_gguf)
                else:
                    logger.error(
                        "  ❌ llama-quantize failed (rc=%d):\n%s",
                        result.returncode,
                        result.stderr[:500],
                    )
            except Exception as exc:
                logger.error("  llama-quantize invocation failed: %s", exc)

    return class_plan
