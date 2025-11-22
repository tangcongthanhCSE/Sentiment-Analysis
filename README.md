# Sentiment-Analysis
# 🎬 IMDB Movie Sentiment Analysis with Transformers
## 📖 Project Overview
Sentiment analysis is a cornerstone of Natural Language Processing (NLP). This project leverages Transfer Learning using the DistilBERT architecture (a lightweight version of BERT) to classify IMDB movie reviews as either Positive or Negative.

Unlike traditional approaches (e.g., LSTM, Bag-of-Words), this Transformer-based model captures deep contextual relationships between words using the Self-Attention mechanism, achieving superior accuracy and inference speed.

## 🚀 Key Features
State-of-the-Art Architecture: Fine-tuned a pre-trained DistilBERT model (distilbert-base-uncased) from Hugging Face.

Custom PyTorch Pipeline: Implemented a custom Dataset class and DataLoader for efficient memory management and batch processing.

Text Preprocessing: Automated HTML tag removal, special character handling, and sub-word tokenization.

Inference Engine: Built a prediction function that returns both the sentiment label and a Confidence Score (Probability).

Manual Training Loop: Constructed the training/validation loops from scratch using PyTorch to handle backpropagation and gradient updates manually.

## 📊 Dataset
Source: IMDB Dataset of 50K Movie Reviews

Size: 50,000 reviews (Balanced: 25k Positive, 25k Negative).

Preprocessing: Reviews were cleaned (HTML removal), truncated/padded to a maximum length of 256 tokens.

## 🛠️ Technologies Used
Language: Python

Deep Learning: PyTorch

NLP Library: Hugging Face Transformers

Data Processing: Pandas, NumPy, Scikit-learn

Visualization: Matplotlib, Seaborn

## ⚙️ Methodology
1. Exploratory Data Analysis (EDA)
Analyzed the distribution of review lengths to determine optimal tokenizer padding/truncation strategies.

Verified class balance to ensure unbiased training.

2. Tokenization & Data Preparation
Utilized DistilBertTokenizerFast for sub-word tokenization.

Created a custom PyTorch Dataset class to handle dynamic padding and attention masks.

3. Fine-tuning (Training)
Model: DistilBertForSequenceClassification (Binary Classification).

Optimizer: AdamW (Adaptive Moment Estimation with Weight Decay).

Loss Function: CrossEntropyLoss (Implicitly handled by the Hugging Face model wrapper).

Strategy: Fine-tuned for 2-3 epochs to prevent overfitting on the pre-trained weights.
