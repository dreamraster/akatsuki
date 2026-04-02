# By dreamraster · dreaMSCend
"""
hmlcore/data.py
============
Dataset loading, preprocessing, and chat-template configuration.

Public API
----------
setup_chat_template(tokenizer) -> tokenizer
load_and_preprocess_dataset(paths, tokenizer, domain, max_length) -> Dataset
"""

import os
import logging
from datasets import load_dataset, concatenate_datasets

import hmlcore.config as cfg

logger = logging.getLogger(__name__)


def setup_chat_template(tokenizer):
    """Install a custom Jinja2 chat template that:
    - Puts the system message first, followed by eos_token
    - Emits REASONING_START at the start of every assistant turn (and as the
      generation prompt) so the prompt prefix tokenises identically whether
      add_generation_prompt is True or False — this is what silences the
      SFTTrainer tokenisation-mismatch warning.
    """
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '" + cfg.SYSTEM_PROMPT.replace("'", "\\'") + "' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '" + cfg.REASONING_START + "' + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + cfg.REASONING_START + "' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template
    return tokenizer


def load_and_preprocess_dataset(paths: list[str], tokenizer,
                                domain: str = "math",
                                max_length: int = 2048):
    """Load one or more datasets, normalise schema, render prompts, filter by length.

    Accepted raw formats:
      • JSONL with any of: instruction/prompt/question + response/output/answer/solution
      • HuggingFace dataset with a train/cot/default/test split
      • Conversational format with a "messages" column

    Returns a HF Dataset with columns:
      prompt        — rendered string (add_generation_prompt=True), ready for GRPOTrainer
      raw_messages  — [system, user] message list, kept for SFT apply_chat_template
      completion    — ground-truth answer extracted from the response
      full_response — original response text (kept for SFT formatting step)
    """
    all_datasets = []
    for p in paths:
        p = str(p).strip()
        logger.info(f"Loading dataset: {p}")
        try:
            if os.path.exists(p) and p.endswith(".jsonl"):
                all_datasets.append(load_dataset("json", data_files=p, split="train"))
            else:
                ds = load_dataset(p, trust_remote_code=True)
                if isinstance(ds, dict):
                    for split in ("train", "cot", "default", "test"):
                        if split in ds:
                            all_datasets.append(ds[split])
                            break
                    else:
                        all_datasets.append(next(iter(ds.values())))
                else:
                    all_datasets.append(ds)
        except Exception as e:
            logger.error(f"Failed to load '{p}': {e}")

    if not all_datasets:
        raise ValueError("No valid datasets loaded. Aborting.")

    dataset = concatenate_datasets(all_datasets)
    logger.info(f"Total examples after merge: {len(dataset)}")

    def format_row(x):
        # ── Extract instruction and raw response ───────────────────────────
        instruction, response = "", ""
        if "messages" in x:
            for msg in x["messages"]:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "user":
                    instruction = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    response = msg.get("content", "")
        else:
            instruction = x.get("instruction",
                          x.get("prompt",
                          x.get("question", "")))
            response    = x.get("response",
                          x.get("output",
                          x.get("answer",
                          x.get("solution", ""))))

        # ── Extract ground-truth answer for reward functions ──────────────
        answer = response
        if "<think>" in str(response) and "</think>" in str(response):
            answer = str(response).split("</think>")[-1].strip()
        elif "<thought>" in str(response) and "</thought>" in str(response):
            answer = str(response).split("</thought>")[-1].strip()
        elif "####" in str(response):          # GSM8K style
            answer = str(response).split("####")[-1].strip()

        # ── Render prompt string for GRPOTrainer ──────────────────────────
        # GRPOTrainer requires a plain str in the "prompt" column.
        # We also keep raw_messages so the SFT step can append the assistant
        # turn and call apply_chat_template on a consistent messages list.
        raw_messages = [
            {"role": "system", "content": cfg.SYSTEM_PROMPT},
            {"role": "user",   "content": instruction},
        ]
        prompt_str = tokenizer.apply_chat_template(
            raw_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt":       prompt_str,
            "raw_messages": raw_messages,
            "completion":   answer,
            "full_response": response,
        }

    # load_from_cache_file=False prevents stale cached tokenisations when
    # SYSTEM_PROMPT / REASONING tags or the tokenizer change between runs.
    dataset = dataset.map(format_row, remove_columns=dataset.column_names,
                          load_from_cache_file=False)

    # Log a warning if prompts are long, but do NOT filter — SFTTrainer and
    # GRPOTrainer both accept max_length / max_prompt_length and truncate
    # internally.  A hard filter silently discards all data when the tokenizer
    # is verbose (e.g. VLMs with large chat templates).
    if len(dataset) > 0:
        try:
            sample_len = len(
                tokenizer(dataset[0]["prompt"], add_special_tokens=False)["input_ids"]
            )
            if sample_len >= max_length:
                logger.warning(
                    "⚠️ Sample prompt is %d tokens (>= max_length=%d). "
                    "Prompts will be truncated by the trainer. "
                    "Pass a larger --max_length to preserve more context.",
                    sample_len, max_length,
                )
        except Exception:
            pass

    logger.info(f"After preprocessing: {len(dataset)} examples ready.")
    return dataset
