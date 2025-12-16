
# Sexism Detection in Tweets using LSTMs and Transformers

## Project Overview

This project addresses **binary sexism detection** in social media text using the **EXIST dataset**. The goal is to classify tweets as *sexist* or *not sexist* by comparing traditional neural architectures (BiLSTM with static word embeddings) against modern **Transformer-based models**. The project implements a complete NLP pipeline, from data acquisition and preprocessing to model training, evaluation, and comparison.

## Dataset

* **Source:** EXIST dataset (NLP UNIBO course material)
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



