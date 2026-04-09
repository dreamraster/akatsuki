# By dreamraster · dreaMSCend
"""
hmlcore/calibration.py
======================
Shared calibration-sample builder for REAP (moe.py) and ShortGPT (dense_pruner.py).

Inspired by the two-lane calibration pattern in moe-compress
(github.com/0xSero/moe-compress): choose samples by a strategy that matches
the workload you care about, filter out tokens that are too short to provide
signal or too long to fit in VRAM.

Public API
----------
build_calibration_samples(dataset, num_samples, *, strategy, max_tokens_per_sample,
                          min_tokens_per_sample, seed) -> List[str]

Selection strategies
--------------------
"longest"  Descending token count.  More hidden-state transitions per
           sample → more reliable ShortGPT/REAP scores.  Recommended default.
"shortest" Ascending token count.  Diverse short-turn coverage.
"random"   Reproducible shuffle (seed-controlled).
"first"    Natural dataset order — previous default behaviour.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

logger = logging.getLogger(__name__)

# Ordered field names to try when extracting text from a sample dict.
_TEXT_FIELDS = (
    "prompt",
    "text",
    "instruction",
    "question",
    "input",
    "content",
    "output",
    "response",
    "answer",
)


def _extract_text(sample: dict) -> Optional[str]:
    """Extract a plain text string from a dataset row.

    Fallback chain:
      1. Known field names (prompt, text, instruction, …)
      2. OpenAI-style messages / conversations list → rendered as "role: content"
      3. Concatenation of all string values in the row
    """
    # 1. Known fields
    for field in _TEXT_FIELDS:
        v = sample.get(field)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2. Chat messages format
    for key in ("messages", "conversations"):
        msgs = sample.get(key)
        if not isinstance(msgs, list):
            continue
        parts: List[str] = []
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role    = msg.get("role") or msg.get("from") or ""
            content = msg.get("content") or msg.get("value") or ""
            if isinstance(content, str) and content.strip():
                line = f"{role}: {content.strip()}" if role else content.strip()
                parts.append(line)
        if parts:
            return "\n".join(parts)

    # 3. Concatenate every string value as last resort
    fallback = " ".join(
        v for v in sample.values()
        if isinstance(v, str) and v.strip()
    )
    return fallback.strip() or None


def _estimate_tokens(text: str) -> int:
    """Fast heuristic: one token ≈ 4 characters (Latin script)."""
    return max(1, len(text) // 4)


def build_calibration_samples(
    dataset,
    num_samples: int,
    *,
    strategy: str = "longest",
    max_tokens_per_sample: Optional[int] = None,
    min_tokens_per_sample: int = 10,
    seed: int = 42,
    tokenizer: Optional[object] = None,
    chat_template: bool = False,
) -> List[str]:
    """Select calibration texts from a HuggingFace Dataset.

    Args:
        dataset:               HF Dataset object.
        num_samples:           Maximum number of texts to return.
        strategy:              "longest" | "shortest" | "random" | "first"
        max_tokens_per_sample: Drop samples whose estimated token count exceeds
                               this value.  None = no upper limit.
        min_tokens_per_sample: Drop samples shorter than this (default 10).
                               Very short samples contribute no useful signal.
        seed:                  RNG seed for "random" strategy.
        tokenizer:             Optional HF tokenizer for chat-template rendering.
        chat_template:         If True, uses tokenizer.apply_chat_template to
                               format the input.

    Returns:
        List of up to num_samples plain-text strings, ordered by strategy.
    """
    pairs: List[tuple[str, int]] = []  # (text, estimated_tokens)
    n_no_text    = 0
    n_too_short  = 0
    n_too_long   = 0

    for sample in dataset:
        text = _extract_text(sample)
        if text is None:
            n_no_text += 1
            continue

        if chat_template and tokenizer is not None:
            # Wrap in actual chat template if requested.
            # This aligns with how the model sees data during SFT/GRPO.
            messages = [{"role": "user", "content": text}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif chat_template:
            # Fallback to manual ChatML if no tokenizer provided
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        est = _estimate_tokens(text)

        if est < min_tokens_per_sample:
            n_too_short += 1
            continue

        if max_tokens_per_sample is not None and est > max_tokens_per_sample:
            n_too_long += 1
            continue

        pairs.append((text, est))

    if n_no_text or n_too_short or n_too_long:
        logger.info(
            "  Calibration filter: %d usable  |  %d no-text  "
            "|  %d too-short (<=%d tok)  |  %d too-long (>%s tok)",
            len(pairs), n_no_text,
            n_too_short, min_tokens_per_sample,
            n_too_long, str(max_tokens_per_sample) if max_tokens_per_sample else "∞",
        )

    if not pairs:
        logger.warning(
            "  ⚠️  No usable calibration samples after filtering — "
            "pruning scores may be unreliable."
        )
        return []

    # ── Apply selection strategy ───────────────────────────────────────────
    if strategy == "longest":
        pairs.sort(key=lambda x: x[1], reverse=True)
    elif strategy == "shortest":
        pairs.sort(key=lambda x: x[1])
    elif strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(pairs)
    # "first" → natural dataset order, no sort

    selected = pairs[:num_samples]

    if selected:
        tok_min  = min(t for _, t in selected)
        tok_max  = max(t for _, t in selected)
        tok_total = sum(t for _, t in selected)
        logger.info(
            "  Calibration samples: strategy=%s  n=%d/%d  "
            "est_tokens total=%s  per-sample min=%d max=%d",
            strategy, len(selected), len(pairs),
            f"{tok_total:,}", tok_min, tok_max,
        )

    return [text for text, _ in selected]
