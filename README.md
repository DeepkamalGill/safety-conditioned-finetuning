# Safety-Conditioned Fine-Tuning of Open-Source LLMs

This repository explores **explicitly controllable safety behavior** in open-source large language models via **LoRA fine-tuning** under constrained compute.

Instead of relying on implicit alignment alone, the project introduces a **safety control signal** (`SAFETY_MODE`) and evaluates whether the model can:
- Reliably **refuse unsafe requests**
- **Avoid over-refusing** benign requests
- Exhibit **controllable behavior** based on system instructions

---

## Project Status

- ✅ Synthetic dataset generation  
- ✅ LoRA fine-tuning pipeline (Qwen2.5-1.5B)  
- ✅ Adapter merge for standalone inference  
- ⏳ Quantitative safety evaluation (in progress)  
- ⏳ Qualitative analysis (in progress)  

---

## Motivation

Safety fine-tuning of LLMs is often treated as a monolithic post-training step (e.g., RLHF), making it difficult to:
- Inspect *why* a model refuses
- Control refusal behavior explicitly
- Measure over-refusal vs under-refusal tradeoffs

This project studies a simpler and more transparent alternative:

> **Condition safety behavior explicitly and measure it directly.**

---

## Method Overview

### Safety Conditioning

Each training example includes a system-level control flag:
You are a helpful assistant. SAFETY_MODE=ON

or

You are a helpful assistant. SAFETY_MODE=OFF


The model is trained to:
- **Refuse** unsafe requests when `SAFETY_MODE=ON`
- Provide **high-level, non-procedural explanations** when `SAFETY_MODE=OFF`
- Answer benign requests normally in both modes

---

### Dataset Construction

The dataset is **synthetically generated** using:
- Human-designed **parameterized templates**
- Programmatic expansion under strict constraints
- No procedural or step-by-step harmful content

Prompt categories:
- **Benign** (capability preservation)
- **Clearly unsafe** (must refuse)
- **Ambiguous / dual-use** (judgment & framing)

This mirrors how alignment datasets are often bootstrapped in practice.

---

### Training Setup

- **Base model**: `Qwen/Qwen2.5-1.5B`
- **Fine-tuning**: LoRA (bf16)
- **Trainable parameters**: ~7M
- **Sequence length**: 512
- **Epochs**: 1
- **Compute**: Free-tier GPU (e.g., Colab T4)

Base model weights remain frozen; only adapter parameters are trained.

---

### Adapter Merging

After training, LoRA adapters are **merged into the base model weights** to produce a standalone checkpoint suitable for:
- Inference without PEFT
- Cleaner evaluation
- Easier deployment

---

## Evaluation (In Progress)

Planned metrics:
- **Correct refusal rate** (unsafe + `SAFETY_MODE=ON`)
- **False refusal rate** (benign + `SAFETY_MODE=ON`)
- **ON vs OFF behavior gap** (controllability)

Evaluation compares:
- Base model
- LoRA safety-conditioned model

Heuristic-based refusal detection is used for interpretability.

---

## Reproducibility

- Runs on **Python 3.12** (Colab default)
- No proprietary APIs
- No paid compute
- All artifacts (models, data) are generated locally

Dependencies are listed in `requirements.txt`.

---

## Limitations

- Tested on small model (1.5B parameters)
- Heuristic-based safety evaluation
- Synthetic dataset (not human preference data)
- No RLHF or adversarial red-teaming

The goal is **clarity and control**, not benchmark saturation.

---

## Future Work

- Quantitative analysis of over-refusal vs data ratio
- Comparison between LoRA and QLoRA
- Extension to larger dense or MoE models
- Adversarial prompt evaluation

---

## License

MIT License.