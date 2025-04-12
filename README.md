# Reasoning Enhancing Data Pipeline for Explicit Memory
This repository contains a complete data curation and filtering pipeline for preparing reasoning-enhancing datasets for Supervised Fine-Tuning (SFT) of Explicit-Memory LLMs in ShareGPT format, with a focus on enhancing logical, mathematical, coding, and linguistic reasoning capabilities.

## 📝 Overview

This project develops and integrates multiple stages of data preparation:

- ✅ Dataset Curation: Selectively extracted 5 large reasoning-focused datasets (600K+ entries) from Hugging Face.
- ✅ Deduplication: Removed redundant entries using MinHash and Jaccard similarity techniques.
- ✅ Rule-Based Filtering: Applied heuristics (entropy, uniqueness, non-alphabetic token ratio) to improve input diversity.
- ✅ LLM Scoring: Used GPT-4o to evaluate informativeness of a 10K sample via prompt-guided labeling (0–5 scale).
- ✅ Binary Classification: Fine-tuned a TinyBERT model to predict LLM-style quality scores (good/bad).
- ✅ Dataset Filtering: Used the TinyBERT classifier to clean 226K examples, retaining 129K high-quality entries.


## 📊 Datasets Used
All datasets were filtered for reasoning tasks only, excluding factual Q&A to reduce memorization bias.
- MetaMathQA: `meta-math/MetaMathQA`
- CodeIO: `hkust-nlp/CodeIO-PyEdu-Reasoning`
- Capybara: `LDJnr/Capybara`
- OpenMath: `open-r1/OpenR1-Math-220k`
- Code18k: `iamtarun/python_code_instructions_18k_alpaca`


## 💻 Usage
First, install the dependencies:
```
pip install -r requirements.txt
```
Then, you can directly see the pipeline working on a sample with only 10,000 entries (instead of 600k) of data loaded and processed by running the following file:
```
python execute_pipeline.py
```
After execution, the pipeline will automatically create a new sft dataset and save it in a JSON file `sftdata.json`.

The final version of the supervised fine-tuning data used for Explicit Memory project is: `sftdata.json`



## 📌 Notice 
Currently, the pipeline is curated exclusively for the Explicit Memory project. It can not be extended to arbitrary datasets other than those listed above.
The reason for that is that the execution of the pipeline depends on a definition of the following dict which stores dataset-specific information:
```
raw_datasets = {
        "meta-math/MetaMathQA": ['train', 'query', None, False],
        "LDJnr/Capybara": ['train', 'conversation', 'conversation', False],
        "iamtarun/python_code_instructions_18k_alpaca": ['train', 'instruction', None, False],
        "open-r1/OpenR1-Math-220k": ['train', 'problem', None, False],
        "hkust-nlp/CodeIO-PyEdu-Reasoning": ['train', 'prompt', None, True],
    }
```
For more details, please go through the `excute_pipeline.py` file.

If you want to adopt the pipeline to other datasets, you need to manually modify the `raw_datasets` dict in `execute_pipeline.py`.



## 📁 Section Intro of the Pipeline

### MinHash Deduplication
- File: `MinHashDeduplication.py`
- Eliminate near-duplicate entries across large-scale reasoning datasets.
- Reducing math dataset size from 395K to 50K, eliminating over 87% redundant entries 

### Rule-Based Filtering
- File: `RuleBasedFilter.py`
- Implemented rule-based quality heuristics to filter noisy or low-value data using:
  - Word entropy
  - Unique word fraction
  - Non-alphabetic character ratio

### Scoring with GPT-4o
- File: `/ipynb/GPT4oLabelling.ipynb`
- Sampled 10K entries from filtered datasets
- Used GPT-4o to rate each entry on a 0–5 scale based on informativeness
- Engineered prompts to elicit consistent evaluations across task types


## TinyBERT Quality Classifier
- File: `/ipynb/TinyBERT_SFT.ipynb`, `TinyBERT_Filter.py`
- Scores binarized into good (4–5) vs bad (0–3)
- Fine-tuned huawei-noah/TinyBERT_General_4L_312D using Hugging Face Trainer
- Achieved robust performance and deployed on full corpus

