# Re2CS: Rethinking Noisy Data and Optimization Reliability for Code Summarization

This is the official repository for the paper **"Rethinking Noisy Data and Optimization Reliability for Code Summarization with Large Language Models"**.

---

## 📂 Project Structure

The repository is organized according to the following directory structure:

```text
Re2CS/
├── baselines/            # Baseline models for comparison
│   ├── EP4CS/            # Enhanced Prompting Framework for Code Summarization
│   └── PromptCS/         # Continuous prompt learning baseline
├── datasets/             # Data management
│   ├── CodeXglue/        # Standard benchmark dataset used for evaluation
│   └── augments_data/    # Refined pseudo-references generated via MA-SDA
├── eval/                 # Evaluation modules
│   ├── overlap_metrics/  # Standard metrics: BLEU, METEOR, ROUGE-L, SBERT
│   └── quality_metrics/  # Quality metrics: Factual Correctness, Hallucination, etc.
├── frameworks/           # Core technical implementations
│   ├── margin_DPO/       # Implementation of Margin-DPO optimization
│   └── masda_pipeline/   # Multi-agent data augmentation pipeline
├── utils/                # General utility functions
└── main.py               # Main entry point for training and evaluation

```
---
## 📝 Download and Prepare Datasets

You can download the datasets from https://huggingface.co/datasets/qustfmy/Re2CS.


## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/qustfmy/Re2CS.git
cd Re2CS
pip install -r requirements.txt

```

### 2. 📊 Data Augmentation (MA-SDA)

Generate high-quality pseudo-references for your training set:

```bash
python frameworks/masda_pipeline/run_agents.py --input datasets/CodeXglue/python --output datasets/augments_data/python

```

### 3. Training with Margin-DPO

Train the LLM using the joint objective of SFT and Margin-DPO:

```bash
python main.py --method margin_DPO --lambda_sft 0.3 --dataset datasets/augments_data/python

```
---

## 📊 Experimental Results

Extensive experiments across six programming languages (Python, Java, Go, PHP, JavaScript, Ruby) demonstrate the effectiveness of Re2CS:

* 
**Factual Error Reduction**: Factual errors were reduced from 34.2% (GT) to 6.6% (Fusioner).


* 
**Hallucination Suppression**: Hallucination rates dropped from 13.9% to 1.6%.


* 
**Downstream Utility**: Code search performance (MRR) improved by an average of 11.2%.


* 
**Scale Efficiency**: A 0.5B model tuned with Re2CS matches the zero-shot performance of GPT-4o.



---

## 📝 Citation

If you use Re2CS in your research, please cite our paper.
