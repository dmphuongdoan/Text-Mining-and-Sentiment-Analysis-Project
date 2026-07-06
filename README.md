# Project 13: The Aesthetics of Generation
### Stylometric Analysis of Writing Style in Qwen Language Models

**Course:** Text Mining and Sentiment Analysis (Natural Language Processing)  
**University:** Università degli Studi di Milano  
**By:** Duy My Phuong Doan 
duymyphuong.doan@studenti.unimi.it

---

## Overview

This project investigates whether different variants of the Qwen model family exhibit
measurably distinct stylometric fingerprints. By collecting texts generated under
identical prompts and extracting quantitative stylistic features, we ask:

> **Do Large Language Models have a recognizable "voice"?**

---

## Models Compared

| Model | Parameters | Generation |
|---|---|---|
| Qwen2.5-7B-Instruct | 7B | 2nd gen |
| Qwen2.5-72B-Instruct | 72B | 2nd gen |
| Qwen3-8B | 8B | 3rd gen |

---

## Methodology

1. **Corpus Generation** — 36 texts (3 models × 4 genres × 3 repeats) via HuggingFace API
2. **Stylometric Analysis** — 9 features: lexical diversity, sentence length, POS ratios, punctuation frequency, syntactic depth, sentiment polarity, readability
3. **Visualization** — PCA and t-SNE dimensionality reduction
4. **Classification** — Random Forest with Leave-One-Out Cross-Validation
5. **Style Transfer** *(optional)* — Testing whether one model can emulate another's style

---

## Key Results

- Classifier accuracy: **63.9%** (vs. 33.3% chance baseline)
- Genre has a stronger effect on stylometric features than model identity
- Qwen2.5-72B writes the longest sentences with least punctuation → most formal style
- Qwen3-8B uses the most punctuation with shortest sentences → most expressive style
- Style transfer achieved **44.4%** feature shift success across 3 runs

---

## Project Structure

```
nlp-aesthetics/
├── data/
│   ├── corpus.json          # 36 generated texts
│   ├── features.csv         # extracted stylometric features
│   └── style_transfer.json  # style transfer results
├── src/
│   ├── collector.py         # text collection via HuggingFace API
│   ├── analyse.py           # stylometric feature extraction
│   ├── visualizer.py        # PCA, t-SNE, bar charts
│   ├── evaluator.py         # Random Forest classifier
│   └── style_transfer.py    # style transfer experiment
├── notebooks/
│   └── demo.ipynb           # full pipeline demonstration
├── results/                 # generated figures
├── requirements.txt
└── README.md
```
---

## Setup

```bash
# Clone repository
git clone https://github.com/dmphuongdoan/Text-Mining-and-Sentiment-Analysis-Project.git
cd Text-Mining-and-Sentiment-Analysis-Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Collect texts from HuggingFace API
python3 src/collector.py

# 2. Extract stylometric features
python3 src/analyse.py

# 3. Generate visualizations
python3 src/visualizer.py

# 4. Run classifier evaluation
python3 src/evaluator.py

# 5. Run style transfer experiment (optional)
python3 src/style_transfer.py
```

Or run the full demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Requirements

- Python 3.12+
- HuggingFace account with free-tier API token
- See `requirements.txt` for full list of dependencies

---

## References

- Opara, C. (2024). StyloAI: Distinguishing AI-generated content with stylometric analysis. *AIED 2024*.
- Kumarage, T. et al. (2023). Stylometric detection of AI-generated text in Twitter timelines. *arXiv:2303.03697*.
- Zellers, R. et al. (2019). Defending against neural fake news. *NeurIPS 32*.

---

## AI Usage Disclaimer

Parts of this project were developed with the assistance of Claude (Anthropic).
The AI was used to support code development and prompt design.
All content has been reviewed, edited, and validated by the author.
