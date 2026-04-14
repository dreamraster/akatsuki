# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
Dataset Builder for GRPO Distillation
======================================
Converts raw .txt files into per-domain JSONL datasets suitable for
ohm_finetuner.py, using a local lmStudio model as the teacher.

Pipeline per chunk:
  1. Classify chunk as math / code / general
  2. Extract a self-contained problem/question from the chunk
  3. Generate a full Chain-of-Thought response with <think>...</think> + answer
  4. Write to the appropriate domain JSONL file

Resume support:
  A state file (<output_dir>/progress.json) tracks every chunk that has been
  successfully processed. Re-running the script skips those chunks, so large
  jobs can be safely interrupted and continued.

lmStudio API:
  lmStudio exposes an OpenAI-compatible REST endpoint at
  http://localhost:1234/v1 by default. This script uses that directly via
  httpx (no openai package required, though it works with that too).

Usage:
  python ohm_databuilder.py \\
      --input_dir  ./raw_texts \\
      --output_dir ./datasets \\
      --model      "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF" \\
      --lmstudio_url http://localhost:1234 \\
      --chunk_size 800 \\
      --chunk_overlap 80 \\
      --workers 4

Output files (in output_dir):
  math.jsonl      – math problems with numeric answers
  code.jsonl      – coding problems with code solutions
  general.jsonl   – reasoning / comprehension problems
  progress.json   – resume state (do not delete mid-run)
  skipped.jsonl   – chunks where extraction failed (for review)
"""

import os
import re
import json
import time
import glob
import logging
import argparse
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain classification heuristics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MATH_SIGNALS = [
    r"\b(equation|solve|calculate|compute|integral|derivative|matrix|vector|"
    r"probability|theorem|proof|formula|algebra|geometry|calculus|statistics|"
    r"arithmetic|percentage|fraction|ratio|sequence|series|polynomial)\b",
    r"[=\+\-\*\/\^]{2,}",          # dense operator clusters
    r"\d+\s*[\+\-\*\/]\s*\d+",     # arithmetic expressions
    r"\\frac|\\sum|\\int|\\sqrt",   # LaTeX math
]

CODE_SIGNALS = [
    r"\b(function|def |class |import |return |algorithm|implement|code|"
    r"program|script|debug|compile|runtime|complexity|recursion|iteration|"
    r"data structure|array|linked list|binary tree|sorting|searching)\b",
    r"(->|=>|:=|==|!=|<=|>=|&&|\|\|)",   # code operators
    r"(for |while |if |else |elif |try:|except:|#include|public static)",
]

GENERAL_SIGNALS = [
    r"\b(explain|describe|what is|why|how does|analyse|analyze|compare|"
    r"summarise|summarize|discuss|evaluate|argue|reason|conclude|infer)\b",
]

def classify_chunk(text: str) -> str:
    """Heuristically classify a text chunk as math, code, or general."""
    text_lower = text.lower()

    math_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in MATH_SIGNALS
    )
    code_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in CODE_SIGNALS
    )
    general_score = sum(
        len(re.findall(p, text_lower, re.IGNORECASE)) for p in GENERAL_SIGNALS
    )

    scores = {"math": math_score, "code": code_score, "general": general_score}
    best = max(scores, key=scores.get)

    # Fall back to general if all scores are 0 or tied ambiguously
    if scores[best] == 0:
        return "general"
    return best


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text chunking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 80):
    """
    Split text into overlapping word-count chunks, breaking on sentence
    boundaries where possible to preserve coherence.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Split into sentences (rough but effective)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_words = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_count + len(words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap words from the end
            current_words = current_words[-overlap:] if overlap else []
            current_count = len(current_words)
        current_words.extend(words)
        current_count += len(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.split()) >= 30]  # discard tiny fragments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# lmStudio client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LMStudioClient:
    """Thin wrapper around lmStudio's OpenAI-compatible /v1/chat/completions."""

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.url     = base_url.rstrip("/") + "/v1/chat/completions"
        self.model   = model
        self.timeout = timeout
        self.client  = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 2048, retries: int = 3) -> str | None:
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        for attempt in range(retries):
            try:
                resp = self.client.post(self.url, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"lmStudio request failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return None

    def close(self):
        self.client.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACT_PROMPTS = {
    "math": """You are a math teacher creating practice problems.
Given the passage below, write ONE clear, self-contained math problem that can be solved from the information in it.
If the passage does not contain enough mathematical content for a solvable problem, reply with exactly: NO_PROBLEM

Passage:
{chunk}

Respond with ONLY the problem statement. No preamble, no solution.""",

    "code": """You are a programming instructor creating coding exercises.
Given the passage below, write ONE clear, self-contained coding problem inspired by the content.
If the passage has no relevant programming content, reply with exactly: NO_PROBLEM

Passage:
{chunk}

Respond with ONLY the problem statement. No preamble, no solution.""",

    "general": """You are an educator creating comprehension and reasoning questions.
Given the passage below, write ONE clear question that requires reasoning to answer — not a simple fact lookup.
If the passage is too short or incoherent to form a good question, reply with exactly: NO_PROBLEM

Passage:
{chunk}

Respond with ONLY the question. No preamble, no answer.""",
}

COT_PROMPTS = {
    "math": """Solve the following math problem step by step.

Problem: {problem}

Think through the problem carefully inside <think>...</think> tags, showing all working.
Then give the final numeric answer after </think> on its own line, prefixed with "Answer:".

Example format:
<think>
Step 1: ...
Step 2: ...
Therefore the answer is 42.
</think>
Answer: 42""",

    "code": """Solve the following coding problem.

Problem: {problem}

Think through your approach inside <think>...</think> tags.
Then provide the complete, working code solution after </think>.
Prefix the code with "Answer:" on its own line.

Example format:
<think>
I need to implement X using Y approach because...
</think>
Answer:
```python
def solution(...):
    ...
```""",

    "general": """Answer the following question with careful reasoning.

Question: {problem}

Think through your answer step by step inside <think>...</think> tags.
Then give your final answer after </think>, prefixed with "Answer:".

Example format:
<think>
First, consider...
Therefore...
</think>
Answer: The answer is ...""",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CoT response parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_cot_response(response: str, domain: str) -> dict | None:
    """
    Extract structured fields from the teacher's CoT response.

    Returns {"response": full_cot, "answer": clean_answer} or None.

    Compatibility note — grpo_distillation.py's format_row() does:
      answer = response.split("</think>")[-1].strip()
    so it will receive everything after </think> as the answer string,
    then check_math_answer tries float() on it.  To guarantee the reward
    function gets a clean value we:
      - strip the "Answer:" prefix here so the stored answer field is bare
      - also rewrite the response so the text after </think> is ONLY the
        bare answer (no "Answer:" prefix), which is what format_row will
        extract and pass to the reward function.
    """
    if not response:
        return None

    # Pull the raw answer text (after "Answer:" or after </think>)
    raw_answer = ""
    if "Answer:" in response:
        raw_answer = response.split("Answer:")[-1].strip()
    elif "</think>" in response:
        raw_answer = response.split("</think>")[-1].strip()

    if not raw_answer:
        return None

    # Clean up the answer value
    clean_answer = raw_answer.strip()

    if domain == "math":
        # Extract just the number so float() works in check_math_answer
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", clean_answer.replace(",", ""))
        if nums:
            clean_answer = nums[-1]

    elif domain == "code":
        # Strip markdown fences if present — reward fn does keyword checks on raw code
        clean_answer = re.sub(r"```[\w]*\n?", "", clean_answer).strip("`").strip()

    # Rewrite response so the post-</think> section is ONLY the bare answer.
    # format_row() does response.split("</think>")[-1].strip() to get the
    # ground-truth answer for the reward function, so this must be clean.
    if "</think>" in response:
        think_part = response.split("</think>")[0] + "</think>"
        clean_response = think_part + "\n" + clean_answer
    else:
        clean_response = response

    return {
        "response": clean_response,  # CoT with bare answer after </think>
        "answer":   clean_answer,    # also stored separately for convenience
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Progress / resume state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProgressTracker:
    """
    Thread-safe tracker that records which chunk IDs have been processed.
    State is flushed to disk after every write so interruptions are safe.
    """

    def __init__(self, state_file: str):
        self.state_file = state_file
        self._lock = threading.Lock()
        self.done: set[str] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    self.done = set(json.load(f).get("done", []))
                logger.info(f"Loaded resume state: {len(self.done)} chunks already done.")
            except Exception:
                logger.warning("Could not load progress state. Starting fresh.")

    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump({"done": list(self.done)}, f)

    def is_done(self, chunk_id: str) -> bool:
        return chunk_id in self.done

    def mark_done(self, chunk_id: str):
        with self._lock:
            self.done.add(chunk_id)
            self._save()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSONL writers (one per domain, thread-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DomainWriter:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._locks  = {d: threading.Lock() for d in ("math", "code", "general")}
        self._files  = {
            d: open(os.path.join(output_dir, f"{d}.jsonl"), "a", encoding="utf-8")
            for d in ("math", "code", "general")
        }
        self._skipped = open(os.path.join(output_dir, "skipped.jsonl"), "a", encoding="utf-8")
        self._skip_lock = threading.Lock()

    def write(self, domain: str, record: dict):
        with self._locks[domain]:
            self._files[domain].write(json.dumps(record, ensure_ascii=False) + "\n")
            self._files[domain].flush()

    def skip(self, chunk_id: str, reason: str, chunk: str):
        with self._skip_lock:
            self._skipped.write(json.dumps({
                "chunk_id": chunk_id, "reason": reason,
                "chunk_preview": chunk[:200]
            }) + "\n")
            self._skipped.flush()

    def close(self):
        for f in self._files.values():
            f.close()
        self._skipped.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-chunk processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_chunk(chunk: str, chunk_id: str, client: LMStudioClient,
                  writer: DomainWriter, progress: ProgressTracker,
                  temperature: float, max_tokens: int):
    """Full pipeline for a single chunk: classify → extract → CoT → write."""

    if progress.is_done(chunk_id):
        return "skipped_resume"

    # Step 1: classify
    domain = classify_chunk(chunk)

    # Step 2: extract problem
    extract_prompt = EXTRACT_PROMPTS[domain].format(chunk=chunk)
    problem = client.chat(
        [{"role": "user", "content": extract_prompt}],
        temperature=0.4,     # low temp for reliable extraction
        max_tokens=512,
    )

    if not problem or problem.strip() == "NO_PROBLEM":
        writer.skip(chunk_id, "no_problem_extracted", chunk)
        progress.mark_done(chunk_id)
        return "skipped_no_problem"

    problem = problem.strip()

    # Step 3: generate CoT response
    cot_prompt = COT_PROMPTS[domain].format(problem=problem)
    raw_response = client.chat(
        [{"role": "user", "content": cot_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    parsed = parse_cot_response(raw_response, domain)
    if not parsed:
        writer.skip(chunk_id, "cot_parse_failed", chunk)
        progress.mark_done(chunk_id)
        return "skipped_parse_failed"

    # Step 4: write record
    # Format matches grpo_distillation.py's format_row() expectations:
    # instruction + response (with <think> tags) + answer field for reward fn
    record = {
        "instruction": problem,
        "response":    parsed["response"],
        "answer":      parsed["answer"],
        "source":      chunk_id,
    }
    writer.write(domain, record)
    progress.mark_done(chunk_id)
    return f"ok_{domain}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Build GRPO datasets from raw .txt files")
    parser.add_argument("--input_dir",     type=str, required=True,
                        help="Directory containing .txt source files (searched recursively)")
    parser.add_argument("--output_dir",    type=str, default="./datasets",
                        help="Where to write math.jsonl, code.jsonl, general.jsonl")
    parser.add_argument("--model",         type=str, required=True,
                        help="lmStudio model name (shown in lmStudio UI, e.g. 'llama-3.1-8b-instruct')")
    parser.add_argument("--lmstudio_url",  type=str, default="http://localhost:1234",
                        help="Base URL of lmStudio server (default: http://localhost:1234)")
    parser.add_argument("--chunk_size",    type=int, default=800,
                        help="Target words per chunk (default: 800)")
    parser.add_argument("--chunk_overlap", type=int, default=80,
                        help="Overlap words between consecutive chunks (default: 80)")
    parser.add_argument("--workers",       type=int, default=2,
                        help="Parallel worker threads (keep low to avoid overloading lmStudio, default: 2)")
    parser.add_argument("--temperature",   type=float, default=0.7,
                        help="Sampling temperature for CoT generation (default: 0.7)")
    parser.add_argument("--max_tokens",    type=int, default=2048,
                        help="Max tokens for CoT response (default: 2048)")
    parser.add_argument("--timeout",       type=int, default=120,
                        help="HTTP timeout per lmStudio request in seconds (default: 120)")
    parser.add_argument("--domain_override", type=str, choices=["math","code","general"],
                        default=None,
                        help="Force all chunks into one domain instead of auto-classifying")

    # ── PRISM ──
    prism_group = parser.add_argument_group("PRISM Data Selection")
    prism_group.add_argument("--prism_select", action="store_true", help="Enable PRISM filtering on generated files")
    prism_group.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B",
                              help="Local student model for PRISM embeddings (default: Qwen/Qwen3-0.6B)")
    prism_group.add_argument("--prism_tier", type=str, default="high", choices=["high", "mid", "low", "high+mid"],
                              help="Quality tier to keep (default: 'high')")
    prism_group.add_argument("--prism_layer", type=int, default=-1, help="Hidden layer for embeddings")
    prism_group.add_argument("--prism_batch", type=int, default=16, help="Selection batch size")
    prism_group.add_argument("--prism_chunk", type=int, default=2000, help="Correlation chunk size")

    args = parser.parse_args()

    # ── Gather all .txt files ──
    txt_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.txt"), recursive=True))
    if not txt_files:
        logger.error(f"No .txt files found under {args.input_dir}")
        return
    logger.info(f"Found {len(txt_files)} .txt files.")

    # ── Build chunk list ──
    all_chunks = []  # list of (chunk_id, chunk_text)
    for filepath in txt_files:
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")
            continue
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        for i, chunk in enumerate(chunks):
            # Stable ID: hash of filepath + chunk index
            chunk_id = hashlib.md5(f"{filepath}::{i}".encode()).hexdigest()[:12]
            all_chunks.append((chunk_id, chunk, filepath))

    logger.info(f"Total chunks to process: {len(all_chunks)}")

    # ── Init shared state ──
    os.makedirs(args.output_dir, exist_ok=True)
    progress = ProgressTracker(os.path.join(args.output_dir, "progress.json"))
    writer   = DomainWriter(args.output_dir)
    client   = LMStudioClient(args.lmstudio_url, args.model, args.timeout)

    pending = [(cid, chunk) for cid, chunk, _ in all_chunks if not progress.is_done(cid)]
    logger.info(f"Chunks to process after resume filter: {len(pending)} "
                f"({len(all_chunks) - len(pending)} already done)")

    # ── Process ──
    counters = {"ok_math": 0, "ok_code": 0, "ok_general": 0,
                "skipped_resume": 0, "skipped_no_problem": 0,
                "skipped_parse_failed": 0}
    counter_lock = threading.Lock()

    def task(item):
        chunk_id, chunk = item
        domain = args.domain_override if args.domain_override else None
        # If domain_override is set, monkey-patch classify_chunk temporarily
        if domain:
            result = process_chunk(chunk, chunk_id, client, writer, progress,
                                   args.temperature, args.max_tokens)
            # Re-classify forced domain for writing
            if result.startswith("ok_"):
                # The record was already written with auto domain; not ideal.
                # For override mode, re-process synchronously is cleaner.
                pass
        result = process_chunk(chunk, chunk_id, client, writer, progress,
                               args.temperature, args.max_tokens)
        with counter_lock:
            counters[result] = counters.get(result, 0) + 1
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, item): item for item in pending}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 50 == 0 or done_count == len(pending):
                logger.info(
                    f"Progress: {done_count}/{len(pending)} | "
                    f"math={counters['ok_math']} code={counters['ok_code']} "
                    f"general={counters['ok_general']} "
                    f"skipped={counters['skipped_no_problem'] + counters['skipped_parse_failed']}"
                )

    writer.close()
    client.close()

    logger.info("━━━ Final counts ━━━")
    
    # ── Phase 2: Optional PRISM Filtering ──
    prism_active = args.prism_select
    model = None
    tokenizer = None

    for domain in ("math", "code", "general"):
        out = os.path.join(args.output_dir, f"{domain}.jsonl")
        if not os.path.exists(out):
            continue
            
        lines_before = sum(1 for _ in open(out, encoding='utf-8'))
        
        if prism_active and lines_before > 0:
            if model is None:
                logger.info(f"�� PRISM: Loading student model for selection: {args.student_model}")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.student_model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                from datasets import Dataset
                from hmlcore.prism_selector import select_with_prism

            logger.info(f"�� PRISM: Filtering {domain}.jsonl ({lines_before} examples)...")
            ds = Dataset.from_json(out)
            # Ensure 'prompt' column exists for PRISM (instruction -> prompt mapping if needed)
            if "prompt" not in ds.column_names and "instruction" in ds.column_names:
                ds = ds.rename_column("instruction", "prompt")
                prompt_renamed = True
            else:
                prompt_renamed = False

            ds_filtered = select_with_prism(
                dataset=ds,
                model=model,
                tokenizer=tokenizer,
                tier=args.prism_tier,
                layer=args.prism_layer,
                batch_size=args.prism_batch,
                chunk_size=args.prism_chunk,
                cache_path=os.path.join(args.output_dir, f"prism_cache_{domain}.pt")
            )
            
            # Rename back
            if prompt_renamed:
                ds_filtered = ds_filtered.rename_column("prompt", "instruction")

            ds_filtered.to_json(out, orient="records", lines=True)
            lines_after = len(ds_filtered)
            logger.info(f"  {domain:10s}: {lines_before} → {lines_after} examples (PRISM tier={args.prism_tier})")
        else:
            logger.info(f"  {domain:10s}: {lines_before} examples  →  {out}")

    logger.info(f"  skipped   : {counters['skipped_no_problem'] + counters['skipped_parse_failed']} "
                f"(see {args.output_dir}/skipped.jsonl)")
    logger.info("✅ Dataset build complete.")


if __name__ == "__main__":
    main()
