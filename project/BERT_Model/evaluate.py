import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.emotion_detector import detect_emotion_svr
from BERT_Model.bert_detector import detect_emotion_bert


def evaluate_models():
    print("=" * 70)
    print("      Model Evaluation: SVR (Ridge/LinearSVC) vs BERT Transformer")
    print("=" * 70)

    test_path = BASE_DIR / "Dataset" / "test.txt"
    if not test_path.exists():
        print(f"Error: Test dataset file not found at {test_path}")
        return

    # Load test data
    df = pd.read_csv(test_path, sep=';', names=['text', 'label'])
    print(f"Loaded {len(df)} test samples from {test_path.name}")

    target_4_emotions = {"anger", "fear", "joy", "sadness"}
    df_4class = df[df['label'].isin(target_4_emotions)].copy()

    print(f"Evaluating on 4 core project emotions ({len(df_4class)} samples)...")

    texts = df_4class['text'].tolist()
    y_true = df_4class['label'].tolist()

    # -----------------------------------------------------------------------
    # 1. SVR Model Evaluation (Forces SVR path)
    # -----------------------------------------------------------------------
    print("\n[1/2] Running inference with SVR Model...")
    os.environ["EMOTION_MODEL_TYPE"] = "svr"
    svr_preds = []
    t0 = time.time()
    for text in texts:
        res = detect_emotion_svr(text)
        pred = res['primary']
        if pred == "neutral" and "all" in res and res["all"]:
            pred = max(res["all"], key=res["all"].get)
        svr_preds.append(pred)
    svr_time = time.time() - t0
    svr_latency_ms = (svr_time / len(texts)) * 1000

    svr_acc = accuracy_score(y_true, svr_preds)
    svr_p, svr_r, svr_f1, _ = precision_recall_fscore_support(y_true, svr_preds, average='weighted', zero_division=0)

    # -----------------------------------------------------------------------
    # 2. BERT Model Evaluation (Forces BERT path)
    # -----------------------------------------------------------------------
    print("\n[2/2] Running inference with BERT Model...")
    os.environ["EMOTION_MODEL_TYPE"] = "bert"
    bert_preds = []
    t0 = time.time()
    for text in texts:
        res = detect_emotion_bert(text)
        pred = res['primary']
        if pred == "neutral" and "all" in res and res["all"]:
            pred = max(res["all"], key=res["all"].get)
        bert_preds.append(pred)
    bert_time = time.time() - t0
    bert_latency_ms = (bert_time / len(texts)) * 1000

    bert_acc = accuracy_score(y_true, bert_preds)
    bert_p, bert_r, bert_f1, _ = precision_recall_fscore_support(y_true, bert_preds, average='weighted', zero_division=0)

    # -----------------------------------------------------------------------
    # 3. Print Comparison Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                    PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'SVR Model':<18} | {'BERT Model':<18} | {'Improvement':<12}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {svr_acc*100:>17.2f}% | {bert_acc*100:>17.2f}% | {((bert_acc-svr_acc)*100):>+11.2f}%")
    print(f"{'Weighted Precision':<25} | {svr_p:>18.4f} | {bert_p:>18.4f} | {(bert_p-svr_p):>+12.4f}")
    print(f"{'Weighted Recall':<25} | {svr_r:>18.4f} | {bert_r:>18.4f} | {(bert_r-svr_r):>+12.4f}")
    print(f"{'Weighted F1-Score':<25} | {svr_f1:>18.4f} | {bert_f1:>18.4f} | {(bert_f1-svr_f1):>+12.4f}")
    print(f"{'Avg Latency (ms/sample)':<25} | {svr_latency_ms:>18.2f} | {bert_latency_ms:>18.2f} | {(bert_latency_ms-svr_latency_ms):>+12.2f}")
    print("=" * 70)

    print("\n--- Detailed SVR Classification Report ---")
    print(classification_report(y_true, svr_preds, zero_division=0))

    print("\n--- Detailed BERT Classification Report ---")
    print(classification_report(y_true, bert_preds, zero_division=0))


if __name__ == "__main__":
    evaluate_models()
