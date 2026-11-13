# BERT Emotion Detection & Classification Model

This directory contains the **BERT-based Deep Learning Emotion Classifier** for the Sentix Real-Time Multilingual Empathy Companion project.

---

## 📌 Overview

The BERT model replaces/complements the classical Support Vector Regression (SVR) and LinearSVC models located in the `Model/` directory. By leveraging bidirectional context attention from Transformer architectures, BERT achieves significantly higher accuracy, context sensitivity, and nuanced emotion recognition.

> ⚠️ **Note:** The existing SVR model files in `Model/` remain completely untouched so you can test and compare both models side-by-side.

---

## 📊 Performance Benchmark: SVR vs BERT

Evaluated on the official project test set (`Dataset/test.txt` - 1,775 samples across 4 primary emotion categories):

| Metric | SVR Model (`Model/`) | BERT Model (`BERT_Model/`) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **54.65%** | **96.39%** | **+41.74% 🚀** |
| **Weighted Precision** | **0.5584** | **0.9641** | **+0.4057** |
| **Weighted Recall** | **0.5465** | **0.9639** | **+0.4174** |
| **Weighted F1-Score** | **0.5385** | **0.9640** | **+0.4255** |
| **Avg Latency (CPU)** | **5.48 ms** | **50.32 ms** | **+44.84 ms** |

### Per-Class F1-Score Breakdown

| Emotion Class | SVR F1-Score | BERT F1-Score |
| :--- | :---: | :---: |
| **Anger** | 0.26 | **0.92** |
| **Fear** | 0.32 | **0.94** |
| **Joy** | 0.79 | **0.99** |
| **Sadness** | 0.45 | **0.96** |

---

## 📁 Directory Structure

```text
BERT_Model/
├── bert_detector.py      # Main inference wrapper (compatible with Flask backend)
├── evaluate.py           # Benchmark comparison script (SVR vs BERT)
├── train_bert.py         # PyTorch fine-tuning script for training on custom dataset
├── README.md             # Model documentation and user guide
└── fine_tuned_bert/     # Model weights checkpoint directory (created upon training)
```

---

## 🚀 Quick Start Guide

### 1. Compare SVR vs BERT Performance
Run the automated evaluation script to test both models on the dataset test split:
```bash
python BERT_Model/evaluate.py
```

### 2. Test Standalone BERT Predictions
Run interactive/sample test predictions using BERT:
```bash
python BERT_Model/bert_detector.py
```

### 3. Fine-Tune BERT on Custom Dataset
To re-train or fine-tune BERT on `Dataset/train.txt` and `Dataset/val.txt`:
```bash
python BERT_Model/train_bert.py
```

---

## ⚙️ How to Switch Models in Flask Backend

You can toggle between **BERT** and **SVR** without modifying any application code using the `EMOTION_MODEL_TYPE` environment variable:

### Use BERT Model (Default - Recommended for Highest Accuracy):
```bash
# Windows PowerShell
$env:EMOTION_MODEL_TYPE="bert"; python -m backend.app

# Linux / macOS
EMOTION_MODEL_TYPE="bert" python -m backend.app
```

### Use Original SVR Model:
```bash
# Windows PowerShell
$env:EMOTION_MODEL_TYPE="svr"; python -m backend.app

# Linux / macOS
EMOTION_MODEL_TYPE="svr" python -m backend.app
```
