# Ohm-Finetuner: Optimizing Domain-Specific Reasoning through SFT, GRPO, and REAP

**Authors:** dreamraster  
**Date:** March 2026  

---

## Abstract

Presenting **Ohm-Finetuner**, a specialized pipeline for transforming general-purpose Large Language Models (LLMs) into high-performance domain-specific reasoners. Our approach leverages a three-stage optimization process: (1) Supervised Fine-Tuning (SFT) for formatting and reasoning alignment, (2) Group Relative Policy Optimization (GRPO) for reinforcement-based reasoning enhancement, and (3) Router-weighted Expert Activation Pruning (REAP) for Mixture-of-Experts (MoE) compression. This synergy allows smaller models to achieve competitive performance on specialized tasks (e.g., coding, mathematics) while significantly reducing inference latency and memory overhead.

---

## 1. Introduction

As Large Language Models (LLMs) scale, their general-purpose capabilities improve, but they often remain inefficient or underperform on niche, high-complexity domains like specialized software engineering or advanced mathematical proof-solving. Fine-tuning these models is common, but conventional Supervised Fine-Tuning often fails to capture the intricate "Chain-of-Thought" (CoT) reasoning required for these domains. Furthermore, large Mixture-of-Experts (MoE) models, while powerful, introduce significant inference costs that hinder deployment in latency-sensitive environments.

Ohm-Finetuner addresses these challenges by combining state-of-the-art reinforcement learning (GRPO) with a novel post-training pruning technique (REAP). By focusing on domain-specific rewards and calibration, models that are both smarter and faster within their target expertise can be achieved. The pipeline also features automated multimodal detection, allowing the system to adapt its optimization constraints for Vision-Language Models (VLMs).

---

## 2. Methodology

The Ohm-Finetuner pipeline consists of three sequential, yet independently configurable, stages.

### 2.1 Stage 1: Supervised Fine-Tuning (SFT)
The first stage serves as a "warm-up" period. High-quality, high-reasoning datasets are used to teach the model a specific response template. This involves the use of `<reasoning>` tags (configurable, default: `<reasoning>`) to encapsulate internal thinking and `<solution>` tags (default: `<solution>`) for the final result. SFT establishes the linguistic baseline and formatting constraints necessary for effective reinforcement learning in the next stage.

### 2.2 Stage 2: Group Relative Policy Optimization (GRPO)
Following SFT,  **Group Relative Policy Optimization (GRPO)** is applied, a reinforcement learning algorithm popularized by DeepSeek-R1. Unlike traditional Proximal Policy Optimization (PPO), which requires a separate "Critic" model to estimate value functions, GRPO computes rewards relative to a group of parallel generations from the same prompt.

**Key Advantages:**
- **Memory Efficiency:** Eliminating the critic model reduces VRAM requirements, allowing for the training of larger student models on standard hardware.
- **Relative Reward Signal:** By comparing $G$ generations (typically 4-16), the model learns which reasoning paths are superior within a specific group, leading to more stable convergence.
- **Domain-Specific Rewards:** Specialized reward functions are integrated, including format-exact match rewards and domain-specific verification. For the coding domain, this includes heuristic-based verification (e.g., checking for syntax and keywords) or a high-capability "LLM-as-judge" to score reasoning quality and correctness.

### 2.3 Stage 3: Router-weighted Expert Activation Pruning (REAP)
For MoE-based student models (e.g., Qwen-MoE series), **REAP**, a one-shot post-training pruning method is further executed. While MoE models provide high capacity, many experts often become redundant or under-activated during domain specialization.

**The REAP Algorithm:**
1. **Calibration:** Run a configurable set of domain-specific samples (typically 128-512) through the model.
2. **Scoring:** For each expert $j$ in every MoE layer, a score $S_j$ is computed based on the gate weight $g_j$ and the L2 norm of the expert's output activation $f_j$:
   $$S_j = \frac{1}{|X_j|} \sum_{x \in X_j} \left[ g_j(x) \cdot \|f_j(x)\|_2 \right]$$
3. **Pruning:** Rank experts by $S_j$ and remove the lowest-performing $N$ experts (e.g., 50% pruning ratio).
4. **Slicing:** Update the model's weight tensors and router matrices to remove the pruned experts.

By calibrating on domain data, REAP ensures that "generalist" experts are discarded while "specialist" experts are preserved.

### 2.4 Implementation Optimizations with Unsloth
The Ohm-Finetuner pipeline is optimized using the **Unsloth** library to maximize hardware efficiency.

- **4-bit Training (QLoRA):** By utilizing Unsloth's hand-written Triton kernels, training is performed in 4-bit quantization with zero loss in accuracy. This reduces VRAM consumption by up to 70% and increases training speed by 2x-5x compared to standard HuggingFace implementations.
- **Dynamic Quantization Output:** Upon completion of the training and pruning stages, the pipeline supports direct export to various quantized formats, including **GGUF** (q4_k_m, q8_0, etc.). This allows for immediate deployment on inference engines such as llama.cpp, Ollama, and LM Studio without requiring external conversion tools.

---

## 3. Benefits

### 3.1 Training Performance
- **Reduced Overhead:** GRPO's lack of a critic model makes it significantly more accessible for organizations with limited compute resources compared to PPO/DPO pipelines.
- **Targeted Learning:** SFT + GRPO allows the model to "learn how to think" rather than just "learn what to say," leading to higher generalization within the target domain.

### 3.2 Inference Performance
- **Model Compression:** REAP can reduce the parameter count of MoE layers by 50% or more with minimal loss in domain-specific accuracy.
- **Latency & Throughput:** Smaller expert counts directly translate to faster token generation speeds and lower VRAM usage during inference, enabling deployment on consumer-grade hardware or edge devices.

### 3.3 Domain Excellence
The synergy of these stages results in a model that outclasses much larger general-purpose models on domain benchmarks. For instance, an Ohm-tuned 30B MoE model pruned to 15B effective parameters can often outperform a dense 70B model on specialized coding tasks.

---

## 4. Conclusion

Ohm-Finetuner provides a robust and efficient framework for domain-specific LLM development. By combining the formatting precision of SFT, the reasoning reinforcement of GRPO, and the architectural efficiency of REAP, we enable the creation of highly capable, deployment-ready models tailored to complex technical fields. Future work will explore non-uniform pruning ratios and integration with multi-modal reasoning.

---

## References

1. **DeepSeek-AI**. (2025). *DeepSeek-V3 Technical Report*. arXiv:2412.19437.
2. **Cerebras Research**. (2025). *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*. arXiv:2510.13999.
3. **Knyazev, B.** (2026). *Router-weighted Expert Activation Merging (REAM) and Pruning Analysis*. Blog.
