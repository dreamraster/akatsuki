# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
MoE Expert Pruning — REAP (Router-weighted Expert Activation Pruning).

One-shot, post-training expert pruning for Mixture-of-Experts models.
Scores each expert by how much useful work it does on a calibration set,
then removes the lowest-scoring experts per layer.

Reference: "REAP the Experts" (arXiv 2510.13999, Cerebras Research)

Usage as standalone:
    python -m hmlcore.moe --model <path> --calibration_dataset <path> --prune_ratio 0.5

Usage from ohm_hmlcore.py:
    python ohm_hmlcore.py ... --prune_experts --prune_ratio 0.5

Saliency score per expert j in a layer:
    S_j = (1/|X_j|) * sum_{x in X_j} [ g_j(x) * ||f_j(x)||_2 ]

Where:
    g_j(x)  = normalised router gate weight for expert j on token x
    f_j(x)  = expert j's output activation for token x
    X_j     = set of tokens routed to expert j by the top-K router
"""

import os
import logging
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MoE Layer Discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_moe_layers(model):
    """
    Find all MoE SparseMoeBlock layers in the model.

    Detects modules that have both a ``gate`` (nn.Linear router) and an
    ``experts`` sub-module with stacked 3D weight tensors.  Works for
    Qwen3-MoE, Mixtral, DeepSeek-MoE, and similar architectures.

    Returns:
        List of (name, module) tuples for each MoE block found.
    """
    moe_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and hasattr(module, 'experts'):
            if isinstance(module.gate, torch.nn.Linear):
                moe_layers.append((name, module))
    return moe_layers


def get_expert_weight(experts, param_name, expert_idx):
    """
    Get the weight tensor for a specific expert, handling PEFT wrappers.

    For PEFT-wrapped models, ``param.data`` gives the base weight tensor.
    For plain nn.Parameter or raw tensors, indexing works directly.
    """
    param = getattr(experts, param_name)
    weight = param.data if isinstance(param, torch.nn.Parameter) else param
    return weight[expert_idx]


def get_top_k(module):
    """Resolve the number of experts activated per token (top-K routing)."""
    for attr in ('top_k', 'num_experts_per_tok'):
        if hasattr(module, attr):
            return getattr(module, attr)
    cfg = getattr(module, 'config', None)
    if cfg and hasattr(cfg, 'num_experts_per_tok'):
        return cfg.num_experts_per_tok
    return 2  # safe default


def get_num_experts(moe_layers):
    """Get the number of experts from the first MoE layer's router."""
    if not moe_layers:
        return 0
    return moe_layers[0][1].gate.out_features


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REAP Scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_expert_output(experts, expert_input, eid):
    """
    Run a single expert's forward pass (gate_up → SiLU → down).

    Supports two weight layouts:
      - gate_up_proj / down_proj  (Qwen3-MoE, Qwen3-VL-MoE)
      - w1 / w3 / w2              (Mixtral, DeepSeek-MoE)

    Returns:
        Expert output tensor of shape (num_tokens, hidden_dim), or None if
        the experts module uses an unrecognised layout.
    """
    if hasattr(experts, 'gate_up_proj'):
        gu = F.linear(expert_input, get_expert_weight(experts, 'gate_up_proj', eid))
        gate_act, up = gu.chunk(2, dim=-1)
        return F.linear(F.silu(gate_act) * up,
                        get_expert_weight(experts, 'down_proj', eid))
    elif hasattr(experts, 'w1'):
        gate_act = F.linear(expert_input, get_expert_weight(experts, 'w1', eid))
        up = F.linear(expert_input, get_expert_weight(experts, 'w3', eid))
        return F.linear(F.silu(gate_act) * up,
                        get_expert_weight(experts, 'w2', eid))
    return None


@torch.no_grad()
def compute_reap_scores(model, tokenizer, dataset, num_samples=512,
                        max_cal_length=2048, calibration_strategy="longest"):
    """
    Compute REAP saliency scores for every expert in every MoE layer.

    Runs calibration data through the model, captures hidden states at each
    MoE layer input via hooks, then computes per-expert scores by replaying
    the router + individual expert forwards.

    **Domain-specific calibration:** The calibration dataset determines which
    experts score highly.  For a code-specialised model, pass code data so
    that code-relevant experts survive pruning.

    Args:
        model:                  The model (merged / full-precision recommended).
        tokenizer:              Tokenizer for encoding calibration prompts.
        dataset:                HF Dataset object.
        num_samples:            Number of calibration samples to use.
        max_cal_length:         Max token length per calibration sample.
        calibration_strategy:   "longest" | "shortest" | "random" | "first".
                                "longest" (default) maximises router-activation
                                signal per sample for more reliable REAP scores.

    Returns:
        Tuple of (reap_scores, token_counts, moe_layers) where:
          - reap_scores: dict mapping layer name → tensor of shape (num_experts,)
          - token_counts: dict mapping layer name → tensor of shape (num_experts,)
          - moe_layers: list of (name, module) tuples
        Returns (None, None, []) if the model has no MoE layers.
    """
    model.eval()
    device = next(model.parameters()).device

    moe_layers = find_moe_layers(model)
    if not moe_layers:
        logger.warning("No MoE layers found in the model.")
        return None, None, []

    num_experts = get_num_experts(moe_layers)
    logger.info(f"REAP: Found {len(moe_layers)} MoE layers with {num_experts} experts each.")

    # ── Build calibration sample list ──────────────────────────────────────
    from hmlcore.calibration import build_calibration_samples

    # For multimodal processors unwrap to inner text tokenizer for extraction.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    cal_texts = build_calibration_samples(
        dataset,
        num_samples,
        strategy              = calibration_strategy,
        max_tokens_per_sample = max_cal_length,
        min_tokens_per_sample = 10,
    )
    if not cal_texts:
        logger.error(
            "❌ Calibration dataset produced no usable samples — REAP scoring aborted."
        )
        return None, None, []

    # ── Register pre-hooks to capture hidden_states at each MoE input ──
    captured_inputs: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_pre_hook(layer_name):
        def hook_fn(module, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                captured_inputs[layer_name] = hs.detach()
        return hook_fn

    for name, module in moe_layers:
        hooks.append(
            module.register_forward_pre_hook(_make_pre_hook(name), with_kwargs=True)
        )

    # ── Score accumulators ──
    reap_scores  = {n: torch.zeros(num_experts) for n, _ in moe_layers}
    token_counts = {n: torch.zeros(num_experts) for n, _ in moe_layers}

    # ── Calibration forward passes ──
    for text in tqdm(cal_texts, desc="REAP calibration"):
        inputs = text_tok(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_cal_length,
            padding=False,
        ).to(device)

        model(**inputs)

        for name, module in moe_layers:
            if name not in captured_inputs:
                continue

            hs = captured_inputs[name]
            if hs.dim() == 3:
                hs = hs.view(-1, hs.shape[-1])

            # Router forward
            router_logits = module.gate(hs)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

            top_k = get_top_k(module)
            tk_weights, tk_indices = torch.topk(routing_weights, top_k, dim=-1)
            tk_weights = tk_weights / tk_weights.sum(dim=-1, keepdim=True)

            experts = module.experts

            for eid in range(num_experts):
                expert_mask = (tk_indices == eid)        # (N, top_k)
                token_mask = expert_mask.any(dim=-1)     # (N,)
                if not token_mask.any():
                    continue

                gate_w = (tk_weights * expert_mask.float())[token_mask].sum(dim=-1)
                expert_input = hs[token_mask]

                expert_out = _compute_expert_output(experts, expert_input, eid)
                if expert_out is None:
                    continue

                out_norms = expert_out.norm(dim=-1)
                reap_scores[name][eid] += (gate_w * out_norms).sum().cpu().item()
                token_counts[name][eid] += token_mask.sum().cpu().item()

        captured_inputs.clear()

    # ── Clean up hooks ──
    for h in hooks:
        h.remove()

    # ── Normalise (mean over routed tokens) ──
    for name in reap_scores:
        active = token_counts[name] > 0
        reap_scores[name][active] /= token_counts[name][active]
        s = reap_scores[name]
        logger.info(
            f"  {name}: score min={s.min():.4f}  max={s.max():.4f}  "
            f"mean={s.mean():.4f}  active={active.sum()}/{num_experts}"
        )

    return reap_scores, token_counts, moe_layers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REAP Pruning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _count_params(model) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def prune_moe_experts(model, reap_scores, moe_layers, prune_ratio=0.5):
    """
    Remove low-scoring experts from every MoE layer.

    Handles two expert storage layouts:
      - nn.ModuleList  (standard HF: Qwen3-MoE, Mixtral, DeepSeek-MoE):
          removes entire expert nn.Module objects from the list.
      - Stacked 3D tensors (Unsloth custom / fused kernel layout):
          slices gate_up_proj / down_proj / w1-w3 along expert dimension.

    For each layer:
      1. Selects the top (1 - prune_ratio) experts by REAP score.
      2. Removes / slices the expert weights to keep only those experts.
      3. Slices the router (gate) weight rows to match.
      4. Updates num_experts on the experts module and model config.

    Args:
        model:        The model to prune (modified in-place).
        reap_scores:  Dict of layer_name → score tensor from compute_reap_scores.
        moe_layers:   List of (name, module) from compute_reap_scores.
        prune_ratio:  Fraction of experts to remove (0.5 = keep half).

    Returns:
        The pruned model.
    """
    num_experts = get_num_experts(moe_layers)
    num_to_keep = max(1, int(num_experts * (1 - prune_ratio)))

    params_before = _count_params(model)
    logger.info(
        "REAP: Pruning %d → %d experts per layer (ratio=%.2f)  "
        "[params before: %s]",
        num_experts, num_to_keep, prune_ratio, f"{params_before:,}",
    )

    for name, module in moe_layers:
        layer_scores = reap_scores[name]
        keep = layer_scores.topk(num_to_keep).indices.sort().values
        keep_list = keep.tolist()
        pruned = sorted(set(range(num_experts)) - set(keep_list))
        logger.info(
            "  %s: keeping %d experts %s  |  dropping %d %s",
            name, num_to_keep, keep_list[:8],
            len(pruned), pruned[:8] if len(pruned) <= 8 else str(pruned[:8]) + "...",
        )

        experts = module.experts

        if isinstance(experts, torch.nn.ModuleList):
            # ── Standard HF layout: remove entire expert nn.Module objects ──
            # This is the layout used by Qwen3-MoE, Mixtral, DeepSeek-MoE in
            # HuggingFace transformers.  Replacing the ModuleList is the ONLY
            # way to actually reduce the parameter count on disk.
            module.experts = torch.nn.ModuleList([experts[i] for i in keep_list])
        else:
            # ── Stacked 3D tensor layout (Unsloth fused / custom kernels) ──
            # Expert weights are stored as 3D tensors: (num_experts, out, in).
            for pname in ('gate_up_proj', 'down_proj', 'w1', 'w2', 'w3'):
                if not hasattr(experts, pname):
                    continue
                old = getattr(experts, pname)
                data = old.data if isinstance(old, torch.nn.Parameter) else old
                setattr(experts, pname,
                        torch.nn.Parameter(data[keep].contiguous()))

            # Slice optional bias tensors
            for bname in ('gate_up_proj_bias', 'down_proj_bias'):
                if hasattr(experts, bname) and getattr(experts, bname) is not None:
                    old_b = getattr(experts, bname)
                    data = old_b.data if isinstance(old_b, torch.nn.Parameter) else old_b
                    setattr(experts, bname,
                            torch.nn.Parameter(data[keep].contiguous()))

        # ── Slice router weight rows (one row per expert) ────────────────
        gate = module.gate
        gate.weight = torch.nn.Parameter(gate.weight.data[keep].contiguous())
        if gate.bias is not None:
            gate.bias = torch.nn.Parameter(gate.bias.data[keep].contiguous())
        gate.out_features = num_to_keep

        # ── Update num_experts bookkeeping on the MoE block ─────────────
        for obj in (experts, module):
            if hasattr(obj, 'num_experts'):
                obj.num_experts = num_to_keep

    # ── Update model config ───────────────────────────────────────────────
    cfg = getattr(model, 'config', None)
    if cfg:
        for attr in ('num_experts', 'num_local_experts'):
            if hasattr(cfg, attr):
                old_val = getattr(cfg, attr)
                setattr(cfg, attr, num_to_keep)
                logger.info("  config.%s: %d → %d", attr, old_val, num_to_keep)

        # Clamp top_k if it now exceeds the surviving expert count
        for attr in ('num_experts_per_tok', 'top_k'):
            if hasattr(cfg, attr):
                old_topk = getattr(cfg, attr)
                if old_topk > num_to_keep:
                    setattr(cfg, attr, num_to_keep)
                    logger.info(
                        "  config.%s clamped: %d → %d (cannot exceed num_to_keep)",
                        attr, old_topk, num_to_keep,
                    )

    params_after = _count_params(model)
    reduction_pct = 100.0 * (params_before - params_after) / max(params_before, 1)
    logger.info(
        "✅ REAP pruning complete: %d → %d experts/layer  "
        "params: %s → %s  (%.1f%% reduction)",
        num_experts, num_to_keep,
        f"{params_before:,}", f"{params_after:,}", reduction_pct,
    )
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dynamic quantization (score-guided 1-bit degradation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def quantize_moe_experts(model, reap_scores, moe_layers, quant_ratio=0.5):
    """
    Score-guided dynamic quantization for MoE experts (--dynamicquant).

    Instead of removing the lowest-scoring experts, keeps them but applies
    1-bit (binary) weight quantization to all their Linear layers.  High-
    scoring experts are left at full precision.

    Inspired by Unsloth Dynamic 2.0: important layers keep Q8/Q6, redundant
    ones are aggressively degraded to IQ1 territory.  Here the REAP saliency
    score is the importance signal instead of a separate calibration imatrix.

    Handles both storage layouts:
      - nn.ModuleList (standard HF): quantize each low-scoring expert module.
      - Stacked 3D tensors (Unsloth fused): warn + fall back to removal for
        that layer (per-expert in-place slicing not possible for stacked layout).

    Args:
        model:        The model to modify in-place.
        reap_scores:  Dict of layer_name → score tensor (num_experts,).
        moe_layers:   List of (name, module) from compute_reap_scores.
        quant_ratio:  Fraction of experts to quantize per layer (0.5 = bottom half).

    Returns:
        The model with low-scoring experts quantized to 1-bit, and a dict
        mapping layer_name -> list of quantized expert indices.
    """
    from hmlcore.quant import quantize_and_verify_module_1bit

    num_experts  = get_num_experts(moe_layers)
    num_to_quant = max(1, int(num_experts * quant_ratio))

    logger.info(
        "REAP+DynQuant: quantizing bottom %d/%d experts per layer to 1-bit  "
        "(ratio=%.2f, %d kept at full precision)",
        num_to_quant, num_experts, quant_ratio, num_experts - num_to_quant,
    )

    total_linears = 0
    quant_info = {}

    for name, module in moe_layers:
        scores = reap_scores[name]

        # Bottom quant_ratio fraction: lowest REAP scores → 1-bit
        quant_idx = scores.topk(num_to_quant, largest=False).indices.sort().values
        quant_list = quant_idx.tolist()
        keep_list  = sorted(set(range(num_experts)) - set(quant_list))

        quant_info[name] = quant_list

        logger.info(
            "  %s: quantizing experts %s  |  full-precision: %s",
            name,
            quant_list if len(quant_list) <= 8 else str(quant_list[:8]) + "...",
            keep_list  if len(keep_list)  <= 8 else str(keep_list[:8])  + "...",
        )

        experts = module.experts

        if isinstance(experts, torch.nn.ModuleList):
            for eid in quant_list:
                score = scores[eid].item()
                logger.info("    expert %d (score=%.4f) → 1-bit:", eid, score)
                n, _ = quantize_and_verify_module_1bit(
                    experts[eid], prefix=f"{name}.experts[{eid}]"
                )
                total_linears += n
        else:
            # Stacked 3D tensor layout: can't slice individual experts for in-place
            # quantization.  Fall back to removing the low-scoring experts instead.
            logger.warning(
                "  %s: stacked-tensor layout — per-expert 1-bit quantization not "
                "supported; falling back to REAP removal of bottom %d experts.",
                name, num_to_quant,
            )
            keep = scores.topk(num_experts - num_to_quant).indices.sort().values
            for pname in ("gate_up_proj", "down_proj", "w1", "w2", "w3"):
                if not hasattr(experts, pname):
                    continue
                old  = getattr(experts, pname)
                data = old.data if isinstance(old, torch.nn.Parameter) else old
                setattr(experts, pname, torch.nn.Parameter(data[keep].contiguous()))
            # Slice router rows to match
            gate = module.gate
            gate.weight       = torch.nn.Parameter(gate.weight.data[keep].contiguous())
            gate.out_features = num_experts - num_to_quant

    logger.info(
        "✅ REAP+DynQuant complete: %d Linear layer(s) quantized to 1-bit  "
        "  Model retains all %d experts — layer count unchanged.",
        total_linears, num_experts,
    )
    return model, quant_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience wrapper (score + prune/quantize in one call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def reap_prune_moe(model, tokenizer, dataset, prune_ratio=0.5,
                   num_samples=512, max_cal_length=2048,
                   calibration_strategy="longest", dynamicquant=False) -> tuple[object, dict | None]:
    """
    Full REAP pipeline: calibrate → score → prune or quantize.

    This is the main entry point used by PrunerNode.

    Args:
        model:                  The (merged, full-precision) model to prune.
        tokenizer:              Tokenizer for encoding calibration prompts.
        dataset:                HF Dataset object.
        prune_ratio:            Fraction of experts to remove/quantize (0.5 = half).
        num_samples:            Number of calibration samples.
        max_cal_length:         Max token length per calibration sample.
        calibration_strategy:   "longest" | "shortest" | "random" | "first".
        dynamicquant:           If True, quantize low-scoring experts to 1-bit
                                instead of removing them (--dynamicquant).

    Returns:
        The pruned/quantized model (modified in-place).
    """
    reap_scores, token_counts, moe_layers = compute_reap_scores(
        model, tokenizer, dataset,
        num_samples           = num_samples,
        max_cal_length        = max_cal_length,
        calibration_strategy  = calibration_strategy,
    )

    if reap_scores is None:
        logger.warning("No MoE layers found. Skipping REAP pruning.")
        return model, None

    if dynamicquant:
        return quantize_moe_experts(model, reap_scores, moe_layers, quant_ratio=prune_ratio)
    return prune_moe_experts(model, reap_scores, moe_layers, prune_ratio), None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """
    Standalone REAP pruning: load a model, prune its MoE experts, save.

    Example:
        python ohm_moe.py \\
            --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
            --calibration_dataset code_samples.jsonl \\
            --prune_ratio 0.5 \\
            --output_dir ./pruned_model
    """
    parser = argparse.ArgumentParser(
        description="REAP: One-shot MoE Expert Pruning"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HF ID")
    parser.add_argument("--calibration_dataset", type=str, required=True,
                        help="Path to calibration dataset (JSONL with 'prompt' field, or HF dataset ID)")
    parser.add_argument("--prune_ratio", type=float, default=0.5,
                        help="Fraction of experts to remove (default: 0.5)")
    parser.add_argument("--calibration_samples", type=int, default=512,
                        help="Number of calibration samples (default: 512)")
    parser.add_argument("--max_cal_length", type=int, default=2048,
                        help="Max token length per sample (default: 2048)")
    parser.add_argument("--output_dir", type=str, default="./pruned_model",
                        help="Directory to save the pruned model")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load calibration dataset
    cal_path = args.calibration_dataset
    if os.path.exists(cal_path) and cal_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=cal_path, split="train")
    else:
        dataset = load_dataset(cal_path, split="train", trust_remote_code=True)

    # Ensure 'prompt' column exists — try common alternatives
    if 'prompt' not in dataset.column_names:
        for alt in ('instruction', 'question', 'input', 'text'):
            if alt in dataset.column_names:
                dataset = dataset.rename_column(alt, 'prompt')
                break
        else:
            raise ValueError(
                f"Dataset must have a 'prompt' column. "
                f"Found: {dataset.column_names}"
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = reap_prune_moe(
        model, tokenizer, dataset,
        prune_ratio=args.prune_ratio,
        num_samples=args.calibration_samples,
        max_cal_length=args.max_cal_length,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving pruned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
