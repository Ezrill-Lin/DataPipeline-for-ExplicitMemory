# Reasoning Data Pipeline for Supervised Fine-Tuning of LLMs
This repository contains a complete data curation and filtering pipeline for preparing high-quality datasets for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs), with a focus on enhancing logical, mathematical, coding, and linguistic reasoning capabilities.

🚀 Overview
This project develops and integrates multiple stages of data preparation:

✅ Dataset Curation: Selectively extracted 5 large reasoning-focused datasets (600K+ entries) from Hugging Face.

✅ Deduplication: Removed redundant entries using MinHash and Jaccard similarity techniques.

✅ Rule-Based Filtering: Applied heuristics (entropy, uniqueness, non-alphabetic token ratio) to improve input diversity.

✅ LLM Scoring: Used GPT-4o to evaluate informativeness of a 10K sample via prompt-guided labeling (0–5 scale).

✅ Binary Classification: Fine-tuned a TinyBERT model to predict LLM-style quality scores (good/bad).

✅ Dataset Filtering: Used the TinyBERT classifier to clean 226K examples, retaining 129K high-quality entries.

📊 Datasets Used
-MetaMathQA: meta-math/MetaMathQA

-CodeIO: hkust-nlp/CodeIO-PyEdu-Reasoning

-Capybara: LDJnr/Capybara

-OpenMath: open-r1/OpenR1-Math-220k

-Code18k: iamtarun/python_code_instructions_18k_alpaca

All datasets were filtered for reasoning tasks only, excluding factual Q&A to reduce memorization bias.
