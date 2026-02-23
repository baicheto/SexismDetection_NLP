
# Sexism Detection in Tweets using LSTMs and Transformers (Part 1)

## Project Overview

This project addresses **binary sexism detection** in social media text using the **EXIST dataset**. The goal is to classify tweets as *sexist* or *not sexist* by comparing traditional neural architectures (BiLSTM with static word embeddings) against modern **Transformer-based models**. The project implements a complete NLP pipeline, from data acquisition and preprocessing to model training, evaluation, and comparison.

## Dataset

* **Source:** EXIST dataset
* **Task:** Binary classification (Sexist vs. Not Sexist)
* **Labels:** Aggregated using majority voting from multiple annotators
* **Splits:** Training, Validation, Test

## Data Preprocessing

The preprocessing pipeline is designed for social media text and includes:

* Lowercasing
* Removal of emojis, URLs, mentions, hashtags, and special characters
* Quote normalization
* Lemmatization with POS tagging using WordNet
* Filtering ambiguous samples (no majority vote)
* Label encoding (`YES → 1`, `NO → 0`)

## Baseline Models (Static Embeddings + LSTM)

Two recurrent neural network architectures are implemented as baselines:

1. **BiLSTM (single layer)**
2. **BiLSTM (stacked layers)**

### Word Embeddings

* Pretrained **GloVe-Twitter embeddings**
* Out-of-vocabulary (OOV) tokens assigned random vectors
* A global `[UNK]` embedding computed as the mean of known embeddings
* Tweets padded to the maximum sequence length

### Training

* Binary cross-entropy loss
* Adam optimizer
* Early stopping
* Custom callback to track **macro F1-score**
* Multiple random seeds for robustness

## Transformer-Based Model

A pretrained **RoBERTa Transformer** is used for contextual text classification.

### Model

* **Model card:** `cardiffnlp/twitter-roberta-base-hate`
* Pretrained on Twitter data and hate-related tasks
* Uses **self-attention** to capture long-range and contextual dependencies
* Subword tokenization via Byte-Pair Encoding (BPE)

### Implementation

* Tokenization and attention masks handled with `AutoTokenizer`
* Classification head fine-tuned using `AutoModelForSequenceClassification`
* Training and evaluation managed via Hugging Face `Trainer`
* Evaluation metric: **macro F1-score**
* Best model selected automatically based on validation F1

## Evaluation Metrics

* **Macro F1-score** (primary metric)
* Accuracy
* Classification reports on validation and test sets


## Technologies Used

* Python
* TensorFlow / Keras
* Hugging Face Transformers & Datasets
* Gensim (GloVe embeddings)
* NLTK
* Scikit-learn
* Pandas, NumPy

Here is a **ready-to-paste GitHub README section** (clean Markdown formatting, properly structured):

---

# Prompting for Sexism Detection with Large Language Models (Part 2)

## Project Overview

The second part of the project explores **prompt-based sexism detection** using **Large Language Models (LLMs)** instead of fine-tuning task-specific classifiers.

The main research question addressed is:

> **How do two different versions of the same LLM behave on the same classification task?**

We specifically compare two versions of the same model family:

* **Mistral-v0.2**
* **Mistral-v0.3**

The goal is to analyze performance differences across **zero-shot**, **few-shot**, and **dynamic few-shot** prompting strategies.

---

## Models Used

Two open-weight instruction-tuned models were evaluated:

* **Mistral-7B-Instruct-v0.2**
* **Mistral-7B-Instruct-v0.3**

The expectation was that the newer version (**v0.3**), with an extended vocabulary (32,768 tokens), would outperform **v0.2**.
However, experimental results contradicted this hypothesis.

---

## Prompting Strategies

### Zero-Shot Prompting

The model receives only the task instruction:

```
Classify the following tweet as SEXIST (YES) or NOT SEXIST (NO).
```

No examples are provided.

---

### Static Few-Shot Prompting

The model is provided with a fixed set of labeled examples before the target tweet:

* *n* examples labeled **YES**
* *n* examples labeled **NO**

These examples are selected from the beginning of the demonstrations dataset.

---

### Dynamic Few-Shot Prompting

A similarity-based example selection strategy was implemented:

* For each target tweet, the most semantically similar examples are selected.
* Similarity is computed using **Euclidean distance** in embedding space.
* The selected examples are balanced (half **YES**, half **NO**).

This strategy aims to provide more relevant demonstrations tailored to each input.

---

## Inference Pipeline

The prompting pipeline follows these steps:

1. Prompt formatting
2. Tokenization
3. Model inference
4. Extraction of textual output (`"YES"` / `"NO"`)
5. Conversion to binary labels (1 / 0)
6. Metric computation

A **low temperature** (near greedy decoding) was used to reduce randomness, as the task is deterministic classification rather than generative text production.

---

## Evaluation Metrics

* **Accuracy**
* **Error Rate (Fail Ratio)**
* **Confusion Matrix** (TP, TN, FP, FN)

---

## Experimental Results

### Accuracy Comparison

| Model Configuration | Accuracy | Fail Ratio |
| ------------------- | -------- | ---------- |
| v0.2 – Zero-Shot    | 0.740    | 0.260      |
| v0.3 – Few-Shot     | 0.733    | 0.267      |
| v0.2 – Dynamic FS   | 0.713    | 0.287      |
| v0.2 – Few-Shot     | 0.673    | 0.327      |
| v0.3 – Zero-Shot    | 0.587    | 0.413      |

---

## Confusion Matrix Highlights

Key observations:

* **Mistral v0.3 produced significantly more false positives than v0.2.**
* **v0.2 Zero-Shot achieved the best overall accuracy.**
* Few-shot prompting did not consistently outperform zero-shot.
* Dynamic few-shot helped **v0.2** but degraded **v0.3** performance.





