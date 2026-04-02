# By dreamraster · dreaMSCend
"""
hmlcore/quant.py
================
Simulated weight quantization for score-guided dynamic precision assignment.

Inspired by Unsloth Dynamic 2.0 GGUFs: instead of applying a single uniform
quantization level to the whole model, layers are assigned precision based on
their importance score.  Low-scored layers are aggressively degraded to 1-bit
(binary) in-place; important layers remain at full precision (bf16/fp32).

Quantization is *simulated*: weights are replaced with their N-bit-quantized
float equivalents stored in the original dtype.  No packed integer kernels are
required at runtime — the model loads and runs normally.  When later exported
to GGUF via Unsloth, these already-degraded weights compress to extreme levels
(IQ1_S / IQ2_XXS territory) with minimal additional KL-divergence penalty.

Public API
----------
quantize_linear_1bit(linear)            -> None
quantize_module_1bit(module)            -> int   (number of Linear layers hit)
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def _binary_row_fraction(weight: torch.Tensor, atol: float = 0.02) -> float:
    """Return the fraction of rows that are binary-quantized.

    A 1-bit quantized row has all elements equal to ±s_i (the per-neuron scale).
    Equivalently: max(|w_row|) ≈ min(|w_row|) ≈ mean(|w_row|).

    We measure spread as (max - min) / mean for each row.  A spread < atol
    (default 2%) indicates a binary row.  Rows that are all-zero are skipped.
    """
    w          = weight.detach().float()
    abs_w      = w.abs()
    row_mean   = abs_w.mean(dim=-1)                             # (out,)
    row_spread = (abs_w.amax(dim=-1) - abs_w.amin(dim=-1))     # (out,)

    nonzero    = row_mean > 1e-10
    if not nonzero.any():
        return 1.0  # all-zero matrix counts as trivially binary

    relative_spread = row_spread[nonzero] / row_mean[nonzero]
    return (relative_spread < atol).float().mean().item()


def verify_linear_1bit(linear: torch.nn.Linear, name: str = "") -> float:
    """Log a one-line 1-bit verification report for a single Linear layer.

    Returns the binary row fraction (0.0 – 1.0).
    A correctly quantized layer should return > 0.99.
    """
    pct   = _binary_row_fraction(linear.weight.data)
    shape = "×".join(str(s) for s in linear.weight.shape)
    icon  = "✅" if pct > 0.95 else ("⚠️ " if pct > 0.50 else "❌")
    logger.info(
        "    %s %-38s  shape=%-14s  binary_rows=%5.1f%%",
        icon, (name or "Linear")[-38:], shape, pct * 100,
    )
    return pct


def verify_module_1bit(module: torch.nn.Module, prefix: str = "") -> dict[str, float]:
    """Log a verification table for every nn.Linear inside *module*.

    Returns a dict mapping layer name → binary row fraction.
    Call this after quantize_module_1bit() to confirm the quantization worked.
    """
    results: dict[str, float] = {}
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear) and m.weight is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            results[full_name] = verify_linear_1bit(m, full_name)

    if results:
        avg = sum(results.values()) / len(results)
        ok  = sum(1 for v in results.values() if v > 0.95)
        logger.info(
            "    Summary: %d/%d Linear(s) verified binary  (avg binary_rows=%.1f%%)",
            ok, len(results), avg * 100,
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Core quantization primitives
# ─────────────────────────────────────────────────────────────────────────────

def quantize_linear_1bit(linear: torch.nn.Linear) -> None:
    """Replace linear.weight with a 1-bit (binary) approximation in-place.

    Each weight w_{i,j} is replaced by ±s_i where s_i is the mean absolute
    value of output-neuron i's input weights:

        s_i  = mean_j |w_{i,j}|
        w̃_{i,j} = (+s_i  if w_{i,j} ≥ 0,  −s_i  if w_{i,j} < 0)

    The sign pattern carries the effective 1 bit; per-neuron scale s_i
    preserves magnitude so layer statistics remain finite.  Zero-scale rows
    (all-zero weights) stay all-zero.

    Implementation notes:
    - torch.where(w >= 0, 1, -1) is used instead of torch.sign(w) because
      sign(0) = 0, which would leave zero-valued weights at 0 after
      multiplication and produce rows with three distinct absolute values
      {0, −s, +s} rather than the required two {−s, +s}.
    - .data.copy_() is used for the final write rather than .data = new_tensor.
      Assigning to .data creates a new Python-level storage reference but does
      NOT modify the underlying memory that the model graph, optimizer, and
      state_dict() point to — so the weights appear unchanged.  copy_() writes
      directly into the existing parameter storage, which is guaranteed to
      persist through save/load and GGUF export.
    - All computation is done in float32 and cast back to avoid bfloat16
      precision loss during scale computation (mean of 896+ values).
    """
    with torch.no_grad():
        orig_dtype = linear.weight.dtype
        w    = linear.weight.data.float()                    # work in fp32
        scale = w.abs().mean(dim=-1, keepdim=True)           # (out, 1)
        signs = torch.where(w >= 0,
                            torch.ones_like(w),
                            torch.full_like(w, -1.0))        # ∈ {+1, −1}
        w_bin = (signs * scale).to(orig_dtype)               # cast back
        linear.weight.data.copy_(w_bin)                      # in-place write


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantize_module_1bit(module: torch.nn.Module) -> int:
    """Apply 1-bit quantization to every nn.Linear inside *module* (recursive).

    Returns:
        Number of nn.Linear layers quantized.
    """
    count = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and m.weight is not None:
            quantize_linear_1bit(m)
            count += 1
    return count


def quantize_and_verify_module_1bit(
    module: torch.nn.Module,
    prefix: str = "",
) -> tuple[int, dict[str, float]]:
    """Quantize every nn.Linear in *module* to 1-bit AND verify the result.

    Returns:
        (num_quantized, verification_results)
        verification_results maps layer name → binary_row_fraction.
    """
    count = quantize_module_1bit(module)
    results = verify_module_1bit(module, prefix=prefix)
    return count, results
