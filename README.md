# Reasoning Data Pipeline for Supervised Fine-Tuning of LLMs
This repository contains a complete data curation and filtering pipeline for preparing high-quality datasets for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs), with a focus on enhancing logical, mathematical, coding, and linguistic reasoning capabilities.


## 🚀 Overview

This project develops and integrates multiple stages of data preparation:

- ✅ Dataset Curation: Selectively extracted 5 large reasoning-focused datasets (600K+ entries) from Hugging Face.
- ✅ Deduplication: Removed redundant entries using MinHash and Jaccard similarity techniques.
- ✅ Rule-Based Filtering: Applied heuristics (entropy, uniqueness, non-alphabetic token ratio) to improve input diversity.
- ✅ LLM Scoring: Used GPT-4o to evaluate informativeness of a 10K sample via prompt-guided labeling (0–5 scale).
- ✅ Binary Classification: Fine-tuned a TinyBERT model to predict LLM-style quality scores (good/bad).
- ✅ Dataset Filtering: Used the TinyBERT classifier to clean 226K examples, retaining 129K high-quality entries.



## 📊 Datasets Used
All datasets were filtered for reasoning tasks only, excluding factual Q&A to reduce memorization bias.
- MetaMathQA: meta-math/MetaMathQA
- CodeIO: hkust-nlp/CodeIO-PyEdu-Reasoning
- Capybara: LDJnr/Capybara
- OpenMath: open-r1/OpenR1-Math-220k
- Code18k: iamtarun/python_code_instructions_18k_alpaca  


## 🧹 MinHash Deduplication
- File: MinHashDeduplication.py
- Eliminate near-duplicate entries across large-scale reasoning datasets.
- Reducing math dataset size from 395K to 50K, eliminating over 87% redundant entries 

## 🧾 Rule-Based Filtering
- File: RuleBasedFilter.py
- Implemented rule-based quality heuristics to filter noisy or low-value data using:
  - Word entropy
  - Unique word fraction
  - Non-alphabetic character ratio

## 🧪 Scoring with GPT-4o
- File: GPT4oLabelling.ipynb
- Sampled 10K entries from filtered datasets
- Used GPT-4o to rate each entry on a 0–5 scale based on informativeness
- Engineered prompts to elicit consistent evaluations across task types


## 🤖 TinyBERT Quality Classifier
- File: TinyBERT_SFT.ipynb, TinyBERT_Filter.ipynb
- Scores binarized into good (4–5) vs bad (0–3)
- Fine-tuned huawei-noah/TinyBERT_General_4L_312D using Hugging Face Trainer
- Achieved robust performance and deployed on full corpus

