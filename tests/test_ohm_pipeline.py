#!/usr/bin/env python3
"""
Smoke tests for ohm_finetuner.py — pipeline correctness, not model quality.

Each test verifies that a given stage / combination reaches completion and
writes the expected artefacts.  Training is capped at 2 steps so the suite
runs on any GPU that can load the test models.

Models under test
-----------------
BNB   : models/qwen-bnb      Qwen3ForCausalLM (BnB 4-bit, dense, 28 layers)
INST  : models/qwen-instruct  Qwen2ForCausalLM (BnB 4-bit, dense, 24 layers)
VLM   : models/qwen-vlm      Qwen3_5ForConditionalGeneration (multimodal, SSM-hybrid)

Expected pipeline behaviour per model
--------------------------------------
BNB   : SFT ✓  GRPO ✓  ShortGPT(--prune_only) ✓  REAP(--prune_experts) → no-MoE skip
INST  : SFT ✓  GRPO ✓  ShortGPT(--prune_only) ✓  REAP(--prune_experts) → no-MoE skip
VLM   : SFT ✓  GRPO → skipped (multimodal)     ShortGPT → skipped (SSM hybrid)
        REAP(--prune_experts) → no-MoE skip

Run
---
    pytest tests/test_ohm_pipeline.py -v --tb=short               # all GPU tests
    pytest tests/test_ohm_pipeline.py -v --tb=short -m "not slow" # dry structure only
    pytest tests/test_ohm_pipeline.py -v --tb=short -k "bnb"      # single model
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
SCRIPT  = ROOT / "ohm_finetuner.py"
BNB     = ROOT / "models" / "qwen-bnb"        # Qwen3ForCausalLM,  BnB 4-bit, dense
INST    = ROOT / "models" / "qwen-instruct"   # Qwen2ForCausalLM,  BnB 4-bit, dense
VLM     = ROOT / "models" / "qwen-vlm"        # Qwen3_5ForConditionalGeneration, VLM
DATASET = ROOT / "datasets" / "teichiai-claude-4.5-high-reasoning-250x.jsonl"

# The entry-point must exist for any test to be meaningful.
if not SCRIPT.exists():
    pytest.skip(f"Entry-point missing: {SCRIPT}", allow_module_level=True)
if not DATASET.exists():
    pytest.skip(f"Dataset missing: {DATASET}", allow_module_level=True)

# Per-model skip markers — individual tests skip when their model folder is absent,
# rather than killing the entire module.
_SKIP_BNB  = pytest.mark.skipif(not BNB.exists(),  reason=f"Model not found: {BNB}")
_SKIP_INST = pytest.mark.skipif(not INST.exists(), reason=f"Model not found: {INST}")
_SKIP_VLM  = pytest.mark.skipif(not VLM.exists(),  reason=f"Model not found: {VLM}")

# ── Shared minimal-cost args ────────────────────────────────────────────────────
_BASE = [
    "--datasets",            str(DATASET),
    "--domain",              "code",
    "--max_steps",           "2",
    "--batch_size",          "1",
    "--num_generations",     "2",
    "--lora_rank",           "8",
    "--max_length",          "512",
    "--calibration_samples", "4",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(extra: List[str], out_dir: Path) -> subprocess.CompletedProcess:
    """Invoke ohm_finetuner.py with _BASE + extra, capturing all output."""
    cmd = [sys.executable, str(SCRIPT)] + _BASE + extra + ["--output_dir", str(out_dir)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        env=env,
    )


def _logs(r: subprocess.CompletedProcess) -> str:
    """Combined stdout + stderr (logging goes to stderr by default)."""
    return r.stdout + r.stderr


def _diagnose(r: subprocess.CompletedProcess, label: str = "") -> str:
    """Build a structured diagnostic block for readable failure messages.

    Includes:
    - Exit code
    - All WARNING / ERROR / CRITICAL lines from the combined log
    - All lines containing 'Traceback' or '  File "' (exception context)
    - Last 60 lines of combined output
    """
    combined = _logs(r)
    lines    = combined.splitlines()

    important = [
        ln for ln in lines
        if any(tag in ln for tag in ("WARNING", "ERROR", "CRITICAL",
                                     "Traceback", '  File "', "NodeError",
                                     "NotImplementedError", "AttributeError",
                                     "RuntimeError"))
    ]

    tail = lines[-60:] if len(lines) > 60 else lines

    hdr = f"{'─'*60}\nDIAGNOSTIC REPORT{' — ' + label if label else ''}\n{'─'*60}"
    sections = [
        hdr,
        f"Exit code : {r.returncode}",
        "",
        "── Key log lines (WARNING / ERROR / exceptions) ──",
        "\n".join(important) if important else "  (none found)",
        "",
        "── Last 60 lines of combined output ──",
        "\n".join(tail),
        "─"*60,
    ]
    return "\n".join(sections)


def _assert_ok(r: subprocess.CompletedProcess, label: str = "") -> None:
    """Fail with a structured diagnostic when exit-code != 0."""
    assert r.returncode == 0, _diagnose(r, label or "expected exit 0")


def _assert_contains(r: subprocess.CompletedProcess,
                     *phrases: str, label: str = "") -> None:
    """Assert all phrases appear somewhere in the combined log output."""
    combined = _logs(r)
    missing  = [p for p in phrases if p not in combined]
    if missing:
        diag = _diagnose(r, label)
        pytest.fail(
            f"Expected phrases not found in logs:\n"
            + "\n".join(f"  • {p!r}" for p in missing)
            + f"\n\n{diag}"
        )


def _assert_any(r: subprocess.CompletedProcess,
                *phrases: str, label: str = "") -> None:
    """Assert at least one phrase appears in the combined log output."""
    combined = _logs(r)
    if not any(p in combined for p in phrases):
        diag = _diagnose(r, label)
        pytest.fail(
            f"None of the expected phrases found in logs:\n"
            + "\n".join(f"  • {p!r}" for p in phrases)
            + f"\n\n{diag}"
        )


# ── Artefact helpers ───────────────────────────────────────────────────────────

def _sft_complete(out_dir: Path) -> bool:
    return (out_dir / "sft" / "sft_complete").exists()


def _finale_has_files(out_dir: Path) -> bool:
    finale = out_dir / "finale"
    return finale.is_dir() and any(finale.iterdir())


def _has_hf_model(out_dir: Path) -> bool:
    return (out_dir / "finale" / "config.json").exists()


def _has_gguf(out_dir: Path) -> bool:
    finale = out_dir / "finale"
    return any(finale.glob("*.gguf")) if finale.is_dir() else False


# ══════════════════════════════════════════════════════════════════════════════
# Stage: SFT
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestSFT:
    """SFT stage runs and writes the sft_complete sentinel on every model."""

    @_SKIP_BNB
    def test_sft_bnb(self, tmp_path):
        """qwen-bnb (Qwen3, BnB 4-bit, dense): SFT completes, sentinel written."""
        r = _run(["--student_model", str(BNB)], tmp_path)
        _assert_ok(r, "sft_bnb")
        assert _sft_complete(tmp_path), _diagnose(r, "sft_complete sentinel missing")

    @_SKIP_INST
    def test_sft_instruct(self, tmp_path):
        """qwen-instruct (Qwen2, BnB 4-bit, dense): SFT completes, sentinel written."""
        r = _run(["--student_model", str(INST)], tmp_path)
        _assert_ok(r, "sft_instruct")
        assert _sft_complete(tmp_path), _diagnose(r, "sft_complete sentinel missing")

    @_SKIP_VLM
    def test_sft_vlm(self, tmp_path):
        """qwen-vlm (Qwen3.5 VLM, SSM-hybrid): SFT completes; GRPO auto-skipped."""
        r = _run(["--student_model", str(VLM)], tmp_path)
        _assert_ok(r, "sft_vlm")
        assert _sft_complete(tmp_path), _diagnose(r, "sft_complete sentinel missing")
        _assert_contains(r, "Multimodal model detected", label="vlm multimodal skip log")


# ══════════════════════════════════════════════════════════════════════════════
# Stage: GRPO (isolated via --disable_sft)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestGRPO:
    """GRPO stage in isolation (--disable_sft)."""

    @_SKIP_BNB
    def test_grpo_bnb(self, tmp_path):
        """qwen-bnb: GRPO runs and logs Step 2."""
        r = _run(["--student_model", str(BNB), "--disable_sft"], tmp_path)
        _assert_ok(r, "grpo_bnb")
        _assert_contains(r, "Starting Step 2", label="grpo step log")

    @_SKIP_INST
    def test_grpo_instruct(self, tmp_path):
        """qwen-instruct: GRPO runs and logs Step 2."""
        r = _run(["--student_model", str(INST), "--disable_sft"], tmp_path)
        _assert_ok(r, "grpo_instruct")
        _assert_contains(r, "Starting Step 2", label="grpo step log")

    @_SKIP_VLM
    def test_grpo_skipped_vlm(self, tmp_path):
        """qwen-vlm: GRPO is skipped with multimodal-skip warning."""
        r = _run(["--student_model", str(VLM), "--disable_sft"], tmp_path)
        _assert_ok(r, "grpo_vlm_skip")
        _assert_contains(r, "Multimodal model detected", label="multimodal skip log")


# ══════════════════════════════════════════════════════════════════════════════
# Stage: ShortGPT dense layer dropping (--prune_only)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestShortGPT:
    """Dense layer dropping via --prune_only.

    Dense models (qwen-bnb, qwen-instruct): ShortGPT should run and write
    finale/.  VLM (SSM-hybrid): no recognisable transformer block list →
    pruning skipped gracefully, pipeline exits 0.
    """

    @_SKIP_BNB
    def test_shortgpt_bnb(self, tmp_path):
        """qwen-bnb: --prune_only triggers ShortGPT; finale/ written."""
        r = _run(["--student_model", str(BNB), "--prune_only",
                  "--calibration_samples", "4"], tmp_path)
        _assert_ok(r, "shortgpt_bnb")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after ShortGPT")

    @_SKIP_INST
    def test_shortgpt_instruct(self, tmp_path):
        """qwen-instruct: --prune_only triggers ShortGPT; finale/ written."""
        r = _run(["--student_model", str(INST), "--prune_only",
                  "--calibration_samples", "4"], tmp_path)
        _assert_ok(r, "shortgpt_instruct")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after ShortGPT")

    @_SKIP_VLM
    def test_shortgpt_skipped_vlm(self, tmp_path):
        """qwen-vlm (SSM-hybrid): --prune_only skips gracefully; pipeline exits 0."""
        r = _run(["--student_model", str(VLM), "--prune_only",
                  "--calibration_samples", "4"], tmp_path)
        _assert_ok(r, "shortgpt_vlm_skip")
        # SSM/linear-attention hybrid → no recognisable block list → error logged
        _assert_any(
            r,
            "no recognisable transformer block",
            "Pruning skipped",
            "SSM",
            "linear_attention",
            label="expected pruning-skip log for VLM",
        )

    @_SKIP_BNB
    def test_shortgpt_custom_ratio_bnb(self, tmp_path):
        """qwen-bnb: --prune_ratio accepted and ShortGPT uses it."""
        r = _run(["--student_model", str(BNB), "--prune_only",
                  "--prune_ratio", "0.25", "--calibration_samples", "4"], tmp_path)
        _assert_ok(r, "shortgpt_ratio_bnb")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty with prune_ratio=0.25")


# ══════════════════════════════════════════════════════════════════════════════
# Stage: REAP expert pruning (--prune_experts on non-MoE models)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestREAPNoMoEGuard:
    """REAP guard: --prune_experts on a dense/VLM model warns and falls back to ShortGPT.

    When --prune_experts is passed but the model has no MoE expert layers, the
    pipeline logs a warning and falls through to ShortGPT layer dropping (dense)
    or skips pruning entirely (SSM/VLM with no recognisable block list).
    All three models should exit 0 with the no-MoE warning logged.
    """

    _WARN = "--prune_experts was passed but"

    @_SKIP_BNB
    def test_reap_no_moe_bnb(self, tmp_path):
        """qwen-bnb (dense): --prune_experts warns about missing MoE, falls back to ShortGPT."""
        r = _run(["--student_model", str(BNB), "--prune_experts"], tmp_path)
        _assert_ok(r, "reap_guard_bnb")
        _assert_contains(r, self._WARN, label="expected no-MoE REAP warning")

    @_SKIP_INST
    def test_reap_no_moe_instruct(self, tmp_path):
        """qwen-instruct (dense): --prune_experts warns about missing MoE, falls back to ShortGPT."""
        r = _run(["--student_model", str(INST), "--prune_experts"], tmp_path)
        _assert_ok(r, "reap_guard_instruct")
        _assert_contains(r, self._WARN, label="expected no-MoE REAP warning")

    @_SKIP_VLM
    def test_reap_no_moe_vlm(self, tmp_path):
        """qwen-vlm (VLM): --prune_experts warns about missing MoE, exits 0."""
        r = _run(["--student_model", str(VLM), "--prune_experts"], tmp_path)
        _assert_ok(r, "reap_guard_vlm")
        _assert_contains(r, self._WARN, label="expected no-MoE REAP warning")


# ══════════════════════════════════════════════════════════════════════════════
# Combination: SFT + GRPO
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestSFTAndGRPO:
    """Full SFT → GRPO pipeline (default flags, no pruning, no merge)."""

    @_SKIP_BNB
    def test_sft_grpo_bnb(self, tmp_path):
        """qwen-bnb: SFT sentinel written and GRPO reaches Step 2."""
        r = _run(["--student_model", str(BNB)], tmp_path)
        _assert_ok(r, "sft_grpo_bnb")
        assert _sft_complete(tmp_path), _diagnose(r, "SFT sentinel missing")
        _assert_contains(r, "Starting Step 2", label="GRPO step log missing")

    @_SKIP_INST
    def test_sft_grpo_instruct(self, tmp_path):
        """qwen-instruct: SFT sentinel written and GRPO reaches Step 2."""
        r = _run(["--student_model", str(INST)], tmp_path)
        _assert_ok(r, "sft_grpo_instruct")
        assert _sft_complete(tmp_path), _diagnose(r, "SFT sentinel missing")
        _assert_contains(r, "Starting Step 2", label="GRPO step log missing")

    @_SKIP_VLM
    def test_sft_grpo_vlm(self, tmp_path):
        """qwen-vlm: SFT sentinel written; GRPO skipped (multimodal)."""
        r = _run(["--student_model", str(VLM)], tmp_path)
        _assert_ok(r, "sft_grpo_vlm")
        assert _sft_complete(tmp_path), _diagnose(r, "SFT sentinel missing for VLM")
        _assert_contains(r, "Multimodal model detected", label="multimodal skip log missing")


# ══════════════════════════════════════════════════════════════════════════════
# Combination: SFT + GRPO + ShortGPT (--prune_only)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestSFTGRPOAndShortGPT:
    """Full SFT → GRPO → ShortGPT pipeline on dense models."""

    @_SKIP_BNB
    def test_full_pipeline_bnb(self, tmp_path):
        """qwen-bnb: SFT + GRPO run; ShortGPT prunes (via --prune_experts fallback); finale/ written."""
        r = _run(["--student_model", str(BNB), "--prune_experts"], tmp_path)
        _assert_ok(r, "full_pipeline_bnb")
        assert _sft_complete(tmp_path), _diagnose(r, "SFT sentinel missing")
        _assert_contains(r, "Starting Step 2", label="GRPO step log missing")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after full pipeline")

    @_SKIP_INST
    def test_full_pipeline_instruct(self, tmp_path):
        """qwen-instruct: SFT + GRPO run; ShortGPT prunes (via --prune_experts fallback); finale/ written."""
        r = _run(["--student_model", str(INST), "--prune_experts"], tmp_path)
        _assert_ok(r, "full_pipeline_instruct")
        assert _sft_complete(tmp_path), _diagnose(r, "SFT sentinel missing")
        _assert_contains(r, "Starting Step 2", label="GRPO step log missing")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after full pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# Output formats — merge + quantize (LLM models only)
# ══════════════════════════════════════════════════════════════════════════════
# GGUF formats (f16 / q8_0 / q4_k) use Unsloth's save_pretrained_gguf when
# available, falling back to HF format with a warning.  Both outcomes are
# accepted — we only verify exit 0 and a non-empty finale/.

@pytest.mark.gpu
@pytest.mark.slow
class TestOutputFormats:
    """Merge + quantize combinations for all models."""

    # ── qwen-bnb ────────────────────────────────────────────────────────────

    @_SKIP_BNB
    def test_bnb_merge_bf16(self, tmp_path):
        """qwen-bnb + bf16 merge → finale/config.json present (HF format)."""
        r = _run(["--student_model", str(BNB), "--merge", "--quantize", "bf16"], tmp_path)
        _assert_ok(r, "bnb_merge_bf16")
        assert _has_hf_model(tmp_path), _diagnose(r, "finale/config.json missing after bf16 merge")

    @_SKIP_BNB
    def test_bnb_merge_f16(self, tmp_path):
        """qwen-bnb + f16 → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(BNB), "--merge", "--quantize", "f16"], tmp_path)
        _assert_ok(r, "bnb_merge_f16")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after f16 export")

    @_SKIP_BNB
    def test_bnb_merge_q8(self, tmp_path):
        """qwen-bnb + q8_0 → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(BNB), "--merge", "--quantize", "q8_0"], tmp_path)
        _assert_ok(r, "bnb_merge_q8")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after q8_0 export")

    @_SKIP_BNB
    def test_bnb_merge_q4k(self, tmp_path):
        """qwen-bnb + q4_k → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(BNB), "--merge", "--quantize", "q4_k"], tmp_path)
        _assert_ok(r, "bnb_merge_q4k")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after q4_k export")

    # ── qwen-instruct ────────────────────────────────────────────────────────

    @_SKIP_INST
    def test_instruct_merge_bf16(self, tmp_path):
        """qwen-instruct + bf16 merge → finale/config.json present."""
        r = _run(["--student_model", str(INST), "--merge", "--quantize", "bf16"], tmp_path)
        _assert_ok(r, "instruct_merge_bf16")
        assert _has_hf_model(tmp_path), _diagnose(r, "finale/config.json missing after bf16 merge")

    @_SKIP_INST
    def test_instruct_merge_f16(self, tmp_path):
        """qwen-instruct + f16 → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(INST), "--merge", "--quantize", "f16"], tmp_path)
        _assert_ok(r, "instruct_merge_f16")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after f16 export")

    @_SKIP_INST
    def test_instruct_merge_q8(self, tmp_path):
        """qwen-instruct + q8_0 → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(INST), "--merge", "--quantize", "q8_0"], tmp_path)
        _assert_ok(r, "instruct_merge_q8")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after q8_0 export")

    @_SKIP_INST
    def test_instruct_merge_q4k(self, tmp_path):
        """qwen-instruct + q4_k → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(INST), "--merge", "--quantize", "q4_k"], tmp_path)
        _assert_ok(r, "instruct_merge_q4k")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after q4_k export")

    # ── qwen-vlm ─────────────────────────────────────────────────────────────

    @_SKIP_VLM
    def test_vlm_merge_bf16(self, tmp_path):
        """qwen-vlm + bf16 merge: SFT runs, GRPO skipped, model saved."""
        r = _run(["--student_model", str(VLM), "--merge", "--quantize", "bf16"], tmp_path)
        _assert_ok(r, "vlm_merge_bf16")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after VLM bf16 merge")

    @_SKIP_VLM
    def test_vlm_merge_f16(self, tmp_path):
        """qwen-vlm + f16 → GGUF or HF fallback; finale/ non-empty."""
        r = _run(["--student_model", str(VLM), "--merge", "--quantize", "f16"], tmp_path)
        _assert_ok(r, "vlm_merge_f16")
        assert _finale_has_files(tmp_path), _diagnose(r, "finale/ empty after VLM f16 export")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline check report (pre-flight analysis)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.slow
class TestPipelineCheck:
    """Verify that the pre-flight compatibility report is printed for each model."""

    @_SKIP_BNB
    def test_pipeline_check_bnb(self, tmp_path):
        """qwen-bnb: pipeline-check report printed before training starts."""
        r = _run(["--student_model", str(BNB), "--disable_sft"], tmp_path)
        _assert_ok(r, "pipeline_check_bnb")
        _assert_any(
            r,
            "Pipeline Compatibility",
            "Dense transformer",
            "SFT",
            label="pipeline check report not found in logs",
        )

    @_SKIP_INST
    def test_pipeline_check_instruct(self, tmp_path):
        """qwen-instruct: pipeline-check report printed before training starts."""
        r = _run(["--student_model", str(INST), "--disable_sft"], tmp_path)
        _assert_ok(r, "pipeline_check_instruct")
        _assert_any(
            r,
            "Pipeline Compatibility",
            "Dense transformer",
            "SFT",
            label="pipeline check report not found in logs",
        )

    @_SKIP_VLM
    def test_pipeline_check_vlm(self, tmp_path):
        """qwen-vlm: pipeline-check report identifies multimodal / SSM-hybrid."""
        r = _run(["--student_model", str(VLM), "--disable_sft"], tmp_path)
        _assert_ok(r, "pipeline_check_vlm")
        _assert_any(
            r,
            "Pipeline Compatibility",
            "Multimodal",
            "multimodal",
            label="pipeline check report or multimodal tag not found",
        )


# ══════════════════════════════════════════════════════════════════════════════
# REAP actual pruning  (requires a true MoE model — placeholder)
# ══════════════════════════════════════════════════════════════════════════════
# Neither test model has MoE layers.  Add a real MoE model (e.g. OLMoE-1B-7B,
# Qwen3-30B-A3B) to models/qwen-moe-real and the tests below will activate.

_MOE = ROOT / "models" / "qwen-moe-real"

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.moe
@pytest.mark.skipif(not _MOE.exists(), reason=f"Real MoE model not found at {_MOE}")
class TestREAPActualPruning:
    """REAP on a model that actually has MoE expert layers."""

    def test_prune_only_moe(self, tmp_path):
        """--prune_only on a real MoE model: REAP scores, prunes, writes finale/."""
        r = _run(
            ["--student_model", str(_MOE), "--prune_only",
             "--prune_ratio", "0.5", "--calibration_samples", "4"],
            tmp_path,
        )
        _assert_ok(r, "reap_prune_only_moe")
        _assert_contains(r, "REAP expert pruning complete", label="REAP completion log")
        assert _finale_has_files(tmp_path), _diagnose(r, "no output files after REAP")

    def test_sft_grpo_reap_moe(self, tmp_path):
        """Full pipeline on MoE: SFT + GRPO + REAP prune + save."""
        r = _run(["--student_model", str(_MOE), "--prune_experts"], tmp_path)
        _assert_ok(r, "full_pipeline_moe")
        _assert_contains(r, "REAP expert pruning complete", label="REAP completion log")
        assert _finale_has_files(tmp_path), _diagnose(r, "no output files after full MoE pipeline")

    def test_prune_moe_merge_bf16(self, tmp_path):
        """MoE + prune + bf16 merge → finale/config.json."""
        r = _run(
            ["--student_model", str(_MOE), "--prune_only",
             "--merge", "--quantize", "bf16", "--calibration_samples", "4"],
            tmp_path,
        )
        _assert_ok(r, "reap_moe_merge_bf16")
        assert _has_hf_model(tmp_path), _diagnose(r, "finale/config.json missing after MoE merge")

    def test_prune_moe_merge_q4k(self, tmp_path):
        """MoE + prune + q4_k GGUF export."""
        r = _run(
            ["--student_model", str(_MOE), "--prune_only",
             "--merge", "--quantize", "q4_k", "--calibration_samples", "4"],
            tmp_path,
        )
        _assert_ok(r, "reap_moe_merge_q4k")
        assert _finale_has_files(tmp_path), _diagnose(r, "no output files after MoE q4_k export")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
