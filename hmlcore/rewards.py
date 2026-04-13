# By dreamraster · dreaMSCend
"""
hmlcore/rewards.py
===============
All reward functions used by GRPOTrainer, plus the LMStudioJudge class.

Reward functions are plain callables with the signature:
    fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]

Two functions need the tokenizer's eos_token captured at runtime:
    _match_format_exactly  and  _check_math_answer
These are built by build_reward_functions() as closures, not as module-level
callables, because GRPOTrainer does not forward the tokenizer through kwargs.

Public API
----------
build_reward_functions(args, tokenizer) -> list[callable]
    Returns the full reward function list for the given domain.
    Also constructs and returns the LMStudioJudge if needed (caller must
    call judge.close() after training).
"""

import re
import time
import hashlib
import logging
import threading
from collections import OrderedDict

import httpx

import hmlcore.config as cfg

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_format_regex(eos_token: str | None = None) -> re.Pattern:
    """Compile the solution-block extraction regex.

    eos_token is optional — GRPOTrainer does not pass the tokenizer to reward
    functions, so build closures via build_reward_functions() to capture it.
    """
    eos_pattern = re.escape(eos_token) if eos_token else r"\S*"
    return re.compile(
        rf"{re.escape(cfg.REASONING_END)} .*? {re.escape(cfg.SOLUTION_START)}"
        rf" (.+?) {re.escape(cfg.SOLUTION_END)}"
        rf" [\s]{{0,}} (?:{eos_pattern})? [\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL | re.VERBOSE,
    )


def _extract_thinking(response: str) -> str:
    """Return the text between REASONING_START and REASONING_END, or ''."""
    # If the standard tags are found, use them
    if cfg.REASONING_START in response and cfg.REASONING_END in response:
        return response.split(cfg.REASONING_START)[1].split(cfg.REASONING_END)[0]
    
    # Robust fallback for QWEN_JACK: prompt ends with <think>, 
    # so response starts with thinking content and only contains the END tag.
    if getattr(cfg, "QWEN_JACK", False):
        if cfg.REASONING_END in response and cfg.REASONING_START not in response:
            return response.split(cfg.REASONING_END)[0]
        
        # If the model got stuck and never generated REASONING_END, the entire block is thinking.
        if cfg.REASONING_START not in response:
            return response
            
    return ""


def _extract_solution(response: str) -> str:
    """Return the text in the solution block.
    If SOLUTION_START/END are defined and present, use them.
    Otherwise, fall back to everything after REASONING_END.
    """
    if cfg.SOLUTION_START and cfg.SOLUTION_END:
        if cfg.SOLUTION_START in response and cfg.SOLUTION_END in response:
            return response.split(cfg.SOLUTION_START)[1].split(cfg.SOLUTION_END)[0]
    
    # Fallback for empty/missing solution tags (e.g. QWEN_JACK mode)
    if cfg.REASONING_END in response:
        return response.split(cfg.REASONING_END)[-1].strip()
        
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Format rewards  (all domains)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_thinking_termination(prompts, completions, **kwargs):
    """Strict penalty for failing to terminate the thinking block."""
    scores = []
    for c in completions:
        if cfg.REASONING_END in c:
            scores.append(0.5)
        else:
            scores.append(-2.0)
    return scores


def match_format_approximately(prompts, completions, **kwargs):
    """±0.5 per tag — rewards partial tag presence even when structure is wrong."""
    scores = []
    for c in completions:
        s  = 0.5 if c.count(cfg.REASONING_START) == 1 else -1.0
        s += 0.5 if c.count(cfg.REASONING_END)   == 1 else -1.0
        s += 0.5 if c.count(cfg.SOLUTION_START)  == 1 else -1.0
        s += 0.5 if c.count(cfg.SOLUTION_END)    == 1 else -1.0
        scores.append(s)
    return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Math rewards  (rule-based, no network call)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_UNIT_RE = re.compile(
    r"(km|m|cm|mm|kg|g|mg|lb|oz|litre|liter|ml|s|sec|min|hour|day|"
    r"mph|kph|m/s|°c|°f|kelvin|joule|watt|newton|pascal|dollar|euro|"
    r"\$|€|£|%|rad|degree)",
    re.IGNORECASE,
)

_STEP_RE = re.compile(
    r"(therefore|thus|so\b|hence|=>|→|step\s*\d+|firstly|secondly|thirdly|"
    r"finally|because|since|which means|it follows|we get|we have|substitut)",
    re.IGNORECASE,
)


def check_math_working_steps(prompts, completions, **kwargs):
    """Reward for explicit step markers and equation lines in the thinking block.

    Score breakdown:
      +2.0  5+ step-signal words/symbols + equation lines
      +1.0  3–4 combined signals
      +0.5  1–2 combined signals
      -0.5  none found
      -1.0  thinking block is empty
    """
    scores = []
    for c in completions:
        thinking = _extract_thinking(c)
        if not thinking.strip():
            scores.append(-1.0)
            continue
        step_count = len(_STEP_RE.findall(thinking))
        eq_lines   = sum(
            1 for line in thinking.splitlines()
            if "=" in line and any(ch.isdigit() for ch in line)
        )
        total = step_count + eq_lines
        if   total >= 5: scores.append(2.0)
        elif total >= 3: scores.append(1.0)
        elif total >= 1: scores.append(0.5)
        else:            scores.append(-0.5)
    return scores


def check_math_units(prompts, completions, **kwargs):
    """Reward unit consistency: +1.5 / +0.5 / -1.0 based on unit presence.

    - Problem has no units → neutral 0.0 (irrelevant)
    - Problem has units, answer has matching unit → +1.5
    - Problem has units, answer has some unit (not matching) → +0.5
    - Problem has units, answer has no unit → -1.0
    """
    scores = []
    for prompt, c in zip(prompts, completions):
        prompt_units = set(_UNIT_RE.findall(str(prompt).lower()))
        if not prompt_units:
            scores.append(0.0)
            continue
        solution = _extract_solution(c)
        answer_units = set(_UNIT_RE.findall(solution.lower()))
        if answer_units & prompt_units:   scores.append(1.5)
        elif answer_units:                scores.append(0.5)
        else:                             scores.append(-1.0)
    return scores


def check_math_reasoning_quality(prompts, completions, **kwargs):
    """Holistic thinking-block quality.

    Rewards:
      +1.0  5+ lines of structured reasoning
      +0.5  3–4 lines
      +1.0  equations with numeric working (capped at 1.0)
      -1.0  very repetitive text (unique-word ratio < 0.4)
      -1.5  thinking block < 30 words (skipping work)
    """
    scores = []
    for c in completions:
        thinking = _extract_thinking(c)
        words = thinking.split()
        if len(words) < 30:
            scores.append(-1.5)
            continue
        lines = [l.strip() for l in thinking.splitlines() if l.strip()]
        score = 0.0
        if   len(lines) >= 5: score += 1.0
        elif len(lines) >= 3: score += 0.5
        eq_count = sum(1 for l in lines if "=" in l and any(ch.isdigit() for ch in l))
        score += min(eq_count * 0.3, 1.0)
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.4:
            score -= 1.0
        scores.append(round(score, 2))
    return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Code heuristic reward  (fallback when judge is disabled)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CODE_KEYWORDS = ["def ", "class ", "return ", "import ", "if ", "for ", "while "]


def check_code_heuristic(prompts, completions, **kwargs):
    """Keyword-presence heuristic for code correctness (no LLM needed)."""
    scores = []
    for c in completions:
        code = _extract_solution(c)
        score = 0.0
        kw_hits = sum(1 for kw in _CODE_KEYWORDS if kw in code)
        if kw_hits >= 2: score += 2.0
        if len(code.strip()) < 10: score -= 2.0
        thinking = _extract_thinking(c)
        if len(thinking.strip()) > 50: score += 1.0
        scores.append(score)
    return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LMStudio Judge  (code + general)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SCORE_RE = re.compile(r"([0-9](?:\.[0-9])?|10(?:\.0)?)\s*/?\s*10")

_JUDGE_PROMPT_CODE = """You are an expert code reviewer. Score the following code solution out of 10.

Problem:
{problem}

Solution:
{solution}

Scoring criteria:
  Correctness: does it solve the problem? (0-4 points)
  Code quality: clean, readable, idiomatic (0-2 points)
  Reasoning in thinking block: clear approach explanation (0-2 points)
  Edge cases considered (0-2 points)

Respond with ONLY a single line in exactly this format:
Score: X/10"""

_JUDGE_PROMPT_GENERAL = """You are an expert evaluator. Score the following answer out of 10.

Question:
{problem}

Answer:
{solution}

Scoring criteria:
  Accuracy and correctness (0-4 points)
  Quality of reasoning shown (0-3 points)
  Clarity and coherence (0-2 points)
  Completeness (0-1 point)

Respond with ONLY a single line in exactly this format:
Score: X/10"""


class LMStudioJudge:
    """Calls a local lmStudio model to score code/general completions 0–5.

    Scores are cached by SHA-256(prompt + completion) so repeated generations
    of identical text skip the network call. Cache uses LRU eviction bounded
    by max_cache_size.
    """

    def __init__(self, base_url: str, model: str,
                 timeout: int = 60, max_cache_size: int = 2048):
        self.url   = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self._lock   = threading.Lock()
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max    = max_cache_size
        self._hits   = 0
        self._misses = 0
        logger.info(f"LMStudioJudge → {self.url}  model={model}  cache={max_cache_size}")

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _key(self, prompt: str, completion: str) -> str:
        return hashlib.sha256((prompt + "|||" + completion).encode()).hexdigest()

    def _get(self, key: str):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def _put(self, key: str, value: float):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max:
                    self._cache.popitem(last=False)
            self._cache[key] = value

    # ── Network ───────────────────────────────────────────────────────────────

    def _call(self, messages: list) -> float | None:
        try:
            resp = self._client.post(self.url, json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16,
            })
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            m = _SCORE_RE.search(text)
            if m:
                return float(m.group(1)) / 10.0 * 5.0   # normalise to 0–5
        except Exception as e:
            logger.debug(f"Judge call error: {e}")
        return None

    # ── Public ────────────────────────────────────────────────────────────────

    def score(self, prompt: str, completion: str, domain: str) -> float:
        """Return a score in [0, 5]. Returns 0.0 on failure (neutral, not penalising)."""
        key = self._key(prompt, completion)
        cached = self._get(key)
        if cached is not None:
            return cached

        # The rendered prompt string looks like:
        #   {system_msg}<|im_end|>{user_question}<reasoning>
        # We want only the bare user question for the judge — strip the system
        # prefix (everything up to and including the first eos_token occurrence)
        # and strip the trailing generation-prompt tag (REASONING_START).
        raw = prompt.strip()

        # Strip system prefix up to the first eos-like separator token
        for eos_marker in ("<|im_end|>", "</s>", "<|end_of_text|>", "<eos>"):
            if eos_marker in raw:
                raw = raw.split(eos_marker, 1)[-1]
                break

        # Strip trailing generation-prompt tag (REASONING_START)
        if raw.endswith(cfg.REASONING_START):
            raw = raw[: -len(cfg.REASONING_START)]

        problem  = raw.strip()[-800:]   # cap length for the judge
        solution = _extract_solution(completion)

        template = _JUDGE_PROMPT_CODE if domain == "code" else _JUDGE_PROMPT_GENERAL
        messages = [{"role": "user", "content": template.format(
            problem=problem, solution=solution
        )}]

        result = self._call(messages)
        if result is None:
            time.sleep(1)
            result = self._call(messages)
        if result is None:
            logger.debug("Judge returned no parseable score; using neutral 0.0")
            result = 0.0

        self._put(key, result)
        return result

    def cache_stats(self) -> str:
        total    = self._hits + self._misses
        hit_rate = self._hits / max(total, 1) * 100
        return (f"Judge cache: {len(self._cache)}/{self._max} entries, "
                f"{self._hits}/{total} hits ({hit_rate:.1f}%)")

    def close(self):
        self._client.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spatial & Scene Rewards
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import math

def _extract_solution_json(completion: str) -> dict | None:
    """Extract and parse the JSON block between <solution> tags.
    
    Robust to truncation: if </solution> is missing, attempts to parse 
    everything from <solution> to the end of the string.
    """
    try:
        content = _extract_solution(completion)
        content = content.strip()
        return json.loads(content)
    except Exception:
        return None

def check_spatial_precision(prompts, completions, **kwargs):
    """Reward for coordinate precision in 2D space.
    
    Expected ground truth 'completion' format: "[x, y]" or JSON with 'position'.
    Predicted format: JSON with 'position' key.
    
    Score: 5.0 * exp(-distance / 100)  -- smoothed proximity reward.
    """
    scores = []
    # GRPOTrainer passes ground truth via 'completion' column in kwargs (as list)
    gt_list = kwargs.get("completion", [None] * len(completions))
    
    for c, gt_raw in zip(completions, gt_list):
        pred_json = _extract_solution_json(c)
        if not pred_json or "position" not in pred_json:
            scores.append(-2.0)
            continue
            
        try:
            pred_pos = pred_json["position"] # [x, y]
            # Ground truth might be a string "[x, y]" or a list
            if isinstance(gt_raw, str):
                # Try to parse as JSON if it looks like a list
                if gt_raw.startswith("[") and gt_raw.endswith("]"):
                    gt_pos = json.loads(gt_raw)
                else:
                    # Fallback for "PLACE_AND_CONNECT" style ground truth from our builder
                    gt_pos = json.loads(gt_raw).get("position")
            else:
                gt_pos = gt_raw
                
            dist = math.sqrt((pred_pos[0] - gt_pos[0])**2 + (pred_pos[1] - gt_pos[1])**2)
            # Perfect match = 5.0, 100px off = 1.83, 500px off = 0.03
            score = 5.0 * math.exp(-dist / 100.0)
            scores.append(round(score, 2))
        except Exception:
            scores.append(0.0)
    return scores

def check_scene_connectivity(prompts, completions, **kwargs):
    """Reward for correct object-to-object connectivity.
    
    Checks if 'connect_to' ID in prediction matches the ground truth.
    """
    scores = []
    gt_list = kwargs.get("completion", [None] * len(completions))
    
    for c, gt_raw in zip(completions, gt_list):
        pred_json = _extract_solution_json(c)
        if not pred_json:
            scores.append(-1.0)
            continue
            
        try:
            pred_conn = pred_json.get("connect_to")
            if isinstance(gt_raw, str) and gt_raw.startswith("{"):
                gt_conn = json.loads(gt_raw).get("connect_to")
            else:
                gt_conn = None # Unknown
                
            if pred_conn == gt_conn and gt_conn is not None:
                scores.append(3.0)
            elif pred_conn is not None:
                scores.append(-1.0) # Wrong target
            else:
                scores.append(0.0) # No connection attempted
        except Exception:
            scores.append(0.0)
    return scores

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory — call this from trainer.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_reward_functions(args, tokenizer) -> tuple[list, "LMStudioJudge | None"]:
    """Build the reward-function list for the given domain.

    Returns (reward_funcs, judge) where judge may be None.
    Caller is responsible for calling judge.close() after training.
    """
    _eos = tokenizer.eos_token

    # ── Closures that capture eos_token ───────────────────────────────────────
    def _match_format_exactly(prompts, completions, **kwargs):
        regex = get_format_regex(_eos)
        return [3.0 if regex.search(c) else 0.0 for c in completions]

    def _check_math_answer(prompts, completions, **kwargs):
        answers = kwargs.get("answer", [None] * len(completions))
        regex   = get_format_regex(_eos)
        scores  = []
        for c, true_ans_raw in zip(completions, answers):
            true_ans = (true_ans_raw[0] if isinstance(true_ans_raw, list)
                        else true_ans_raw)
            m = regex.search(c)
            if not m:
                scores.append(-2.5)
                continue
            guess = m.group(1).strip()
            try:
                fg = float(guess.replace(",", ""))
                ft = float(str(true_ans).replace(",", ""))
                if fg == ft:
                    scores.append(5.0)
                elif abs(fg - ft) / (abs(ft) + 1e-9) < 0.01:
                    scores.append(3.5)
                else:
                    scores.append(-1.5)
            except Exception:
                scores.append(5.0 if guess == str(true_ans) else 0.0)
        return scores

    # ── Base rewards (all domains) ────────────────────────────────────────────
    reward_funcs = [_match_format_exactly, match_format_approximately, check_thinking_termination]
    judge = None

    # ── Domain-specific rewards ───────────────────────────────────────────────
    if args.domain == "math":
        reward_funcs += [
            _check_math_answer,
            check_math_working_steps,
            check_math_units,
            check_math_reasoning_quality,
        ]
        logger.info("�� Math rewards: correctness + working steps + units + reasoning quality")

    elif args.domain == "scene":
        reward_funcs += [
            check_spatial_precision,
            check_scene_connectivity,
        ]
        logger.info("�� Scene rewards: spatial precision + connectivity mapping")

    elif args.domain in ("code", "general"):
        use_judge = (not args.disable_judge) and bool(args.judge_model)

        if use_judge:
            judge = LMStudioJudge(
                base_url       = args.judge_url,
                model          = args.judge_model,
                timeout        = args.judge_timeout,
                max_cache_size = args.judge_cache_size,
            )
            _dom = args.domain

            def _llm_judge(prompts, completions, **kwargs):
                scores = [judge.score(p, c, _dom)
                          for p, c in zip(prompts, completions)]
                logger.debug(judge.cache_stats())
                return scores

            reward_funcs.append(_llm_judge)
            logger.info(
                f"�� {args.domain.capitalize()} rewards: LLM judge "
                f"→ {args.judge_url}  model={args.judge_model}"
            )
        else:
            if args.domain == "code":
                reward_funcs.append(check_code_heuristic)
                logger.info("�� Code rewards: heuristic fallback "
                            "(pass --judge_model to enable LLM judge)")
            else:
                logger.info("�� General rewards: format only "
                            "(pass --judge_model to enable LLM judge)")

    return reward_funcs, judge
