import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "fine_tuned_bert"

# Emotion Label Mappings (Matching dataset & standard emotion labels)
LABEL_LIST = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Fast In-Memory Pre-Tokenized PyTorch Dataset
# ---------------------------------------------------------------------------
class FastTextEmotionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64, max_samples=None):
        df = pd.read_csv(file_path, sep=';', names=['text', 'label'])
        texts = []
        labels = []
        for _, row in df.iterrows():
            t = str(row['text']).strip()
            l = str(row['label']).strip().lower()
            if l in LABEL2ID:
                texts.append(t)
                labels.append(LABEL2ID[l])

        if max_samples and len(texts) > max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]

        print(f"Pre-tokenizing {len(texts)} samples from {Path(file_path).name}...")
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx]
        }


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += len(labels)

        if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
            print(f"  Step [{step + 1}/{len(dataloader)}] - Batch Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("      Optimized Fast BERT Training Pipeline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")

    train_path = DATASET_DIR / "train.txt"
    val_path = DATASET_DIR / "val.txt"
    test_path = DATASET_DIR / "test.txt"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Training or Validation file missing in {DATASET_DIR}")

    # Load Tokenizer
    print("\n[1/5] Loading BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load Datasets with fast pre-tokenization
    print("[2/5] Creating Data Loaders...")
    train_dataset = FastTextEmotionDataset(train_path, tokenizer, max_len=MAX_LEN, max_samples=4000)
    val_dataset = FastTextEmotionDataset(val_path, tokenizer, max_len=MAX_LEN, max_samples=1000)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Load Model
    print("[3/5] Initializing BERT Model for Sequence Classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(DEVICE)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training Loop
    print("\n[4/5] Fine-tuning BERT model...")
    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = eval_epoch(model, val_loader, DEVICE)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{EPOCHS} [{elapsed:.1f}s] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            print(f"   -> Saving fine-tuned BERT checkpoint to {OUTPUT_DIR}...")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            # Save label map and metadata
            config_meta = {
                "base_model": MODEL_NAME,
                "emotions": LABEL_LIST,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "best_val_f1": float(best_val_f1),
                "best_val_accuracy": float(val_acc)
            }
            with open(OUTPUT_DIR / "meta.json", "w") as f:
                json.dump(config_meta, f, indent=4)
            print("   -> Checkpoint saved successfully!")

    total_time = time.time() - start_time
    print(f"\n[5/5] Training & model export completed in {total_time/60:.2f} minutes.")

    # Evaluate on Test Set
    if test_path.exists():
        print("\n" + "=" * 60)
        print("          Final Test Set Evaluation")
        print("=" * 60)
        test_dataset = FastTextEmotionDataset(test_path, tokenizer, max_len=MAX_LEN)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
        best_model.to(DEVICE)

        test_loss, test_acc, test_p, test_r, test_f1, preds, labels = eval_epoch(best_model, test_loader, DEVICE)

        print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
        print(f"Test Precision: {test_p:.4f}")
        print(f"Test Recall:    {test_r:.4f}")
        print(f"Test F1-Score:  {test_f1:.4f}\n")

        target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
        report = classification_report(labels, preds, target_names=target_names, zero_division=0)
        print("Classification Report:")
        print(report)


if __name__ == "__main__":
    main()
