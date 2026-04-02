# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
Dataset preprocessing script for Qwen instruction-tuning.
Prepares raw datasets into the standard instruction-tuning format.

Usage:
    python ohm_datapreprocessor.py.py \\
        --dataset-name techedata/CodeAlpaca-20k \\
        --output-dir ./data/processed \\
        --model-path Qwen/Qwen-0_6B
"""

import argparse
import json
import os
from typing import Dict, List, Any

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_prompt_template(
    instruction: str,
    input_text: str = None,
    output_text: str = None
) -> tuple:
    """
    Create a prompt in Qwen style chat template format.

    Returns:
        tuple of (prompt, output_text)
    """
    if input_text:
        query = f"{instruction}\n{input_text}"
    else:
        query = instruction

    # Format with chat template (Qwen style)
    prompt = (
        "<|user|>\n"
        f"{query}"
        "<|assistant|>\n"
    )

    return prompt, output_text


def preprocess_alpaca_format(
    example: Dict,
    tokenizer: Any,
    max_length: int = 2048
) -> Dict:
    """Process an Alpaca-style example into training format."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')

    prompt = create_prompt_template(instruction, input_text)
    final_prompt = f"{prompt[0]}{output_text}"

    # Tokenize
    encoding = tokenizer(
        final_prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False
    )

    input_ids = encoding["input_ids"]
    labels = [-100] * len(input_ids)

    # Only train on the output (assistant response)
    if output_text:
        prompt_str = prompt[0]
        prompt_token_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_token_ids)

        for i in range(prompt_length, min(len(input_ids), len(prompt) + 10):
            labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int = 2048
) -> Dataset:
    """Preprocess a dataset for instruction-tuning."""
    def apply_preprocessing(example):
        return preprocess_alpaca_format(example, tokenizer, max_length)

    # Process dataset
    processed = dataset.map(
        lambda x: apply_preprocessing(x),
        remove_columns=['instruction', 'input', 'output'],
        desc="Preprocessing dataset"
    )

    return processed


def format_for_qwen(
    input_text: str,
    instruction: str = None,
    max_length: int = 2048
) -> tuple:
    """
    Format input text and optional instruction for Qwen.

    Returns:
        tuple of (prompt, output_text)
    """
    if instruction:
        query = f"{instruction}\n\n{input_text}"
    else:
        query = input_text

    prompt = (
        "<|user|>\n"
        f"{query}"
        "<|assistant|>"
    )

    return prompt


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess dataset for Qwen instruction-tuning'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='techedata/CodeAlpaca-20k',
        help='Hugging Face dataset name or path to dataset file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/processed',
        help='Directory to save preprocessed data'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen-0_6B',
        help='Path to tokenizer for special tokens'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--num-proc',
        type=int,
        default=4,
        help='Number of processes for parallel processing'
    )

    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Load dataset
    raw_dataset = load_dataset(args.dataset_name)

    # Process each split
    processed_datasets = DatasetDict()
    for split in raw_dataset.keys():
        logger.info(f"Processing {split} split...")
        split_data = raw_dataset[split]
        processed = preprocess_dataset(split_data, tokenizer)
        processed_datasets[split] = processed

    # Save processed dataset
    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_data in processed_datasets.items():
        output_file = os.path.join(args.output_dir, f"{split_name}.jsonl")
        logger.info(f"Saving {split_name} to {output_file}")
        with open(output_file, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

    logger.info(f"Preprocessing complete. Data saved to {args.output_dir}")


if __name__ == '__main__':
    main()
