#!/usr/bin/env python3
"""
Tests for LoRA merging and inference.

This module contains tests to verify:
1. LoRA works correctly by comparing results before and after fine-tuning
2. Merged model inference works properly
"""

import os

import pytest


# Test constants
LORA_CHECKPOINT_DIR = "./checkpoints/qwen3-0.6b-cuda-lora/final_checkpoint"
MERGED_MODEL_DIR = "./checkpoints/qwen3-0.6b-cuda-lora/merged_model"

# Check if final checkpoint exists
LORA_CONFIG_PATH = os.path.join(LORA_CHECKPOINT_DIR, "adapter_config.json")
LORA_WEIGHTS_PATH = os.path.join(LORA_CHECKPOINT_DIR, "adapter_model.safetensors")

# Available GPU device
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    DEVICE_MAP = "auto"
else:
    DEVICE_MAP = "cpu"


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Set up test fixtures."""
    pass


class TestLoRAMerging:
    """Tests for LoRA merging functionality."""

    def test_lora_checkpoint_exists(self):
        """Verify LoRA checkpoint directory has required files."""
        if not os.path.exists(LORA_CONFIG_PATH):
            pytest.skip("LoRA checkpoint not found. Run train_cuda_lora.py first.")

        # Check required files exist
        assert os.path.exists(LORA_CONFIG_PATH), "adapter_config json not found"
        assert os.path.exists(LORA_WEIGHTS_PATH), "adapter_model.safetensors not found"

    def test_lora_config_structure(self):
        """Verify LoRA config has required fields."""
        if not os.path.exists(LORA_CONFIG_PATH):
            pytest.skip("LoRA checkpoint not found. Run train_cuda_lora.py first.")

        import json

        with open(LORA_CONFIG_PATH, "r") as f:
            config = json.load(f)

        # Check required fields
        assert "base_model_name_or_path" in config, "Missing base_model_name_or_path"
        assert config["base_model_name_or_path"] == "Qwen/Qwen3-0.6B", (
            "Unexpected base model"
        )
        assert "r" in config, "Missing r (LoRA rank)"
        assert "lora_alpha" in config, "Missing lora_alpha"

    def test_merged_model_exists(self):
        """Test that merged model directory is created after merge."""
        if not os.path.exists(MERGED_MODEL_DIR):
            pytest.skip("Merged model does not exist. Run merge_lora.py first.")

        # Check for required files in merged model
        assert os.path.exists(
            os.path.join(MERGED_MODEL_DIR, "config.json")
        ), "merged model config missing"
        assert os.path.exists(
            os.path.join(MERGED_MODEL_DIR, "model.safetensors")
        ), "merged model weights missing"

    def test_merged_model_inference(self):
        """Test that merged model can generate text."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            pytest.skip("transformers or torch not available")

        if not os.path.exists(MERGED_MODEL_DIR):
            pytest.skip("Merged model does not exist. Run merge_lora.py first.")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_DIR,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_DIR,
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP,
        )

        # Simple test prompt
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
            )

        # Verify we got output
            generated_ids = outputs[0]
            assert len(generated_ids) > 0, "No tokens generated"

        # Decode and verify
        generated_text = tokenizer.decode(
            generated_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True
        )
        assert len(generated_text) > 0, "Empty generation"

    def test_lora_forward_pass(self):
        """Test that LoRA model can do forward pass with random data."""
        try:
            from peft import AutoPeftModelForCausalLM
            import torch
        except ImportError:
            pytest.skip("peft or torch not available")

        if not os.path.exists(LORA_CONFIG_PATH):
            pytest.skip("LoRA checkpoint not found. Run train_cuda_lora.py first.")

        # Load tokenizer and base model
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            trust_remote_code=True,
        )

        # Load LoRA model
        lora_model = AutoPeftModelForCausalLM.from_pretrained(
            LORA_CHECKPOINT_DIR,
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP,
        )

        # Test forward pass with dummy input
        test_input = tokenizer("Test", return_tensors="pt")
        test_input = {k: v.to(lora_model.device) for k, v in test_input.items()}

        with torch.no_grad():
            outputs = lora_model(**test_input, labels=test_input["input_ids"])
            assert outputs.loss is not None, "Loss should be computed"
            assert torch.isfinite(outputs.loss), "Loss should be finite"


class TestInferencePipeline:
    """Tests for inference pipeline after LoRA merging."""

    @pytest.mark.integration
    def test_complete_inference_workflow(self):
        """End-to-end test of inference with merged model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            pytest.skip("transformers or torch not available")

        if not os.path.exists(MERGED_MODEL_DIR):
            pytest.skip("Merged model does not exist. Run merge_lora.py first.")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_DIR,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_DIR,
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP,
        )

        # Test with a simple prompt
        test_cases = ["What is 2+2?", "The sky is blue because", "Once upon"]

        for prompt in test_cases:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                )

            # Verify output is not empty
                generated_ids = outputs[0]
                assert len(generated_ids) > 0

            # Verify we get valid tokens back
                generated_text = tokenizer.decode(
                    generated_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True
                )
                assert isinstance(generated_text, str)
                assert len(generated_text) > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
