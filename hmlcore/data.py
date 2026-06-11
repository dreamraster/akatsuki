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


def setup_chat_template(tokenizer, is_multimodal: bool = False):
    """Install a chat template.
    If --qwen_jack is enabled, uses the standard Qwen3-thinking template via Unsloth.
    Otherwise, uses the custom Jinja2 template with REASONING_START tags.
    """
    if is_multimodal:
        return tokenizer

    # We can check the tags to decide, or just rely on the template if we had a flag.
    # Since I'm adding the flag to config.py, let's use it.
    if getattr(cfg, "QWEN_JACK", False):
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen3-thinking",
        )
        return tokenizer

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
                                max_length: int = 2048,
                                is_multimodal: bool = False):
    """Load one or more datasets, normalise schema, render prompts, filter by length.

    Accepted raw formats:
      • JSONL with any of: instruction/prompt/question + response/output/answer/solution
      • HuggingFace dataset with a train/cot/default/test split
      • Conversational format with a "messages" column

    Returns a HF Dataset with columns:
      prompt        — rendered string (add_generation_prompt=True), ready for GRPOTrainer
      raw_messages  — [system, user] message list, kept for SFT apply_chat_template
      completion    — ground-truth answer extracted from the response
      full_response — original response text (normalised to <think> if --qwen_jack)
      images        — list of PIL Images (only if is_multimodal=True)
    """
    import re
    from PIL import Image
    def _strip(x): return (x or "").strip()
    think_re = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

    def load_image(img):
        import io
        import requests
        import base64
        
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            # check if HTTP URL
            if img.startswith("http://") or img.startswith("https://"):
                try:
                    resp = requests.get(img, timeout=10)
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load image from URL {img}: {e}")
                    return None
            # check if base64
            elif img.startswith("data:image") or ";base64," in img:
                try:
                    base64_data = img.split(";base64,")[-1]
                    img_data = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(img_data)).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load base64 image: {e}")
                    return None
            # local path
            elif os.path.exists(img):
                try:
                    return Image.open(img).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load image from local path {img}: {e}")
                    return None
        # dict with bytes and/or path (Hugging Face format)
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                try:
                    return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load HF dict image bytes: {e}")
            elif "path" in img and img["path"] is not None:
                if os.path.exists(img["path"]):
                    try:
                        return Image.open(img["path"]).convert("RGB")
                    except Exception as e:
                        logger.warning(f"Failed to load HF dict image path {img['path']}: {e}")
        return None

    def normalize_assistant(text: str) -> str:
        text = _strip(text)
        if not text: return "<think></think>\n"
        m = think_re.search(text)
        if m:
            think_block = m.group(0).strip()
            rest = text[m.end():].lstrip()
            return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
        return f"<think></think>\n{text}".rstrip()

    # ── Parse custom column mappings ──────────────────────────────────────
    instr_cols_raw = "instruction,prompt,question"
    resp_cols_raw  = "response,output,answer,solution"

    if getattr(cfg, "args", None) is not None:
        instr_cols_raw = getattr(cfg.args, "instruction_cols", instr_cols_raw)
        resp_cols_raw  = getattr(cfg.args, "response_cols", resp_cols_raw)

    instr_cols = [c.strip() for c in instr_cols_raw.split(",")]
    resp_cols  = [c.strip() for c in resp_cols_raw.split(",")]

    def _text_from_content(content) -> str:
        """Return plain text from a message content that may be a list of
        multimodal parts (e.g. [{"type": "image", ...}, {"type": "text", "text": "..."}]).
        Non-string content is coerced to avoid passing image blobs to apply_chat_template."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts).strip()
        return str(content) if content else ""

    def format_row(x):
        # ── Extract instruction and raw response ───────────────────────────
        instruction, response = "", ""
        if "messages" in x:
            for msg in x["messages"]:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "user":
                    instruction = _text_from_content(msg.get("content", ""))
                elif msg.get("role") == "assistant":
                    response = _text_from_content(msg.get("content", ""))
        else:
            # Try custom columns first, then fallbacks
            for col in instr_cols:
                if col in x and x[col]:
                    instruction = x[col]
                    break
            
            for col in resp_cols:
                if col in x and x[col]:
                    response = x[col]
                    break

        # ── Normalise response format ──────────────────────────────────────
        if getattr(cfg, "QWEN_JACK", False):
            response = normalize_assistant(response)

        # ── Extract ground-truth answer for reward functions ──────────────
        answer = response
        if "<think>" in str(response) and "</think>" in str(response):
            answer = str(response).split("</think>")[-1].strip()
        elif "<thought>" in str(response) and "</thought>" in str(response):
            answer = str(response).split("</thought>")[-1].strip()
        elif "####" in str(response):          # GSM8K style
            answer = str(response).split("####")[-1].strip()

        # ── Extract and process images if multimodal ────────────────────────
        extracted_images = []
        if "messages" in x:
            for msg in x["messages"]:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "image":
                                img_val = part.get("image") or part.get("image_url")
                                if img_val:
                                    if isinstance(img_val, dict) and "url" in img_val:
                                        extracted_images.append(img_val["url"])
                                    else:
                                        extracted_images.append(img_val)
                            elif part.get("type") == "image_url":
                                img_val = part.get("image_url")
                                if isinstance(img_val, dict) and "url" in img_val:
                                    extracted_images.append(img_val["url"])
                                elif img_val:
                                    extracted_images.append(img_val)

        for col in ["images", "image", "img"]:
            if col in x and x[col] is not None:
                val = x[col]
                if isinstance(val, list):
                    extracted_images.extend(val)
                else:
                    extracted_images.append(val)

        loaded_images = []
        if is_multimodal:
            for img in extracted_images:
                loaded_img = load_image(img)
                if loaded_img is not None:
                    loaded_images.append(loaded_img)
            
            # If no image found, generate 28x28 grey fallback
            if not loaded_images:
                loaded_images.append(Image.new("RGB", (28, 28), color="gray"))

            user_content = [{"type": "image"} for _ in loaded_images] + [{"type": "text", "text": instruction}]
        else:
            user_content = instruction

        # ── Render prompt string for GRPOTrainer ──────────────────────────
        if is_multimodal:
            system_content = [{"type": "text", "text": cfg.SYSTEM_PROMPT}]
        else:
            system_content = cfg.SYSTEM_PROMPT

        raw_messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        prompt_str = tokenizer.apply_chat_template(
            raw_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        ret = {
            "prompt":       prompt_str,
            "raw_messages": raw_messages,
            "completion":   answer,
            "full_response": response,
        }
        if is_multimodal:
            ret["images"] = loaded_images
        return ret

    all_datasets = []
    for p in paths:
        p = str(p).strip()
        logger.info(f"Loading dataset: {p}")
        try:
            # Normalize and resolve relative/absolute path
            p_resolved = os.path.normpath(p)
            if not os.path.isabs(p_resolved):
                # Try relative to current working directory
                cwd_resolved = os.path.normpath(os.path.join(os.getcwd(), p_resolved))
                if os.path.exists(cwd_resolved):
                    p_resolved = cwd_resolved
            
            if os.path.exists(p_resolved) and p_resolved.endswith(".jsonl"):
                import json
                from datasets import Dataset
                rows = []
                with open(p_resolved, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                formatted_rows = [format_row(r) for r in rows]
                ds = Dataset.from_list(formatted_rows)
            else:
                ds = load_dataset(p, trust_remote_code=True)
                if isinstance(ds, dict):
                    for split in ("train", "cot", "default", "test"):
                        if split in ds:
                            ds = ds[split]
                            break
                    else:
                        ds = next(iter(ds.values()))
                
                # Normalise schema immediately after loading to avoid alignment issues 
                # during concatenate_datasets.
                ds = ds.map(format_row, remove_columns=ds.column_names, load_from_cache_file=False)
            
            all_datasets.append(ds)

        except Exception as e:
            logger.error(f"Failed to load '{p}': {e}")

    if not all_datasets:
        raise ValueError("No valid datasets loaded. Aborting.")

    dataset = concatenate_datasets(all_datasets)
    logger.info(f"Total examples after merge: {len(dataset)}")

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
