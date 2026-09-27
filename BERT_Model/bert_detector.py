import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

FINE_TUNED_MODEL_DIR = Path(__file__).resolve().parent / "fine_tuned_bert"
FALLBACK_HF_MODEL = "nateraw/bert-base-uncased-emotion"

from backend.utils import (
    clip_intensity,
    intensity_to_severity,
    handle_negation,
    get_polarity,
    polarity_gate,
    suppress_secondary_emotions,
)

EMOTIONS = ["anger", "fear", "joy", "sadness"]
ALL_EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

_NEUTRAL_THRESHOLD = 0.4
_SHORT_TEXT_MAX_WORDS = 4

_bert_model = None
_bert_tokenizer = None
_model_device = None
_id2label = {}
_label2id = {}
_loaded_source = None


def load_bert_model():
    """
    Lazy load BERT emotion classifier.
    1. Attempts to load fine-tuned model from BERT_Model/fine_tuned_bert/
    2. Falls back to HuggingFace model nateraw/bert-base-uncased-emotion
    """
    global _bert_model, _bert_tokenizer, _model_device, _id2label, _label2id, _loaded_source

    if _bert_model is not None:
        return True

    _model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Option 1: Local Fine-Tuned BERT Checkpoint
    if FINE_TUNED_MODEL_DIR.exists() and (FINE_TUNED_MODEL_DIR / "config.json").exists():
        try:
            print(f"[BERT Detector] Loading fine-tuned model from {FINE_TUNED_MODEL_DIR}...")
            _bert_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
            _bert_model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
            _bert_model.to(_model_device)
            _bert_model.eval()
            _id2label = _bert_model.config.id2label
            _label2id = _bert_model.config.label2id
            _loaded_source = "local_fine_tuned"
            print("[BERT Detector] Fine-tuned model loaded successfully.")
            return True
        except Exception as e:
            print(f"[BERT Detector] Failed loading local model: {e}. Falling back to HuggingFace...")

    # Option 2: Pre-trained HF Emotion Model
    try:
        print(f"[BERT Detector] Loading pretrained HF model ({FALLBACK_HF_MODEL})...")
        _bert_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_HF_MODEL)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_HF_MODEL)
        _bert_model.to(_model_device)
        _bert_model.eval()
        _id2label = _bert_model.config.id2label
        _label2id = _bert_model.config.label2id
        _loaded_source = "hf_pretrained"
        print("[BERT Detector] Pretrained HF BERT model loaded successfully.")
        return True
    except Exception as e:
        print(f"[BERT Detector] Could not load BERT model: {e}")
        _bert_model = None
        _bert_tokenizer = None
        return False


def detect_emotion_bert(text: str, original_text: str = None):
    """
    BERT-based Emotion Detection matching project schema.
    Returns:
    {
        "primary": dominant_emotion,
        "severity": severity_level,
        "intensity": score,
        "all": { "anger": score, "fear": score, "joy": score, "sadness": score, ... }
    }
    """
    success = load_bert_model()

    # Fallback to heuristic if BERT model failed to load
    if not success or _bert_model is None or _bert_tokenizer is None:
        polarity = get_polarity(text)
        scores = {emo: 0.1 for emo in EMOTIONS}
        if polarity == "positive":
            scores["joy"] = 0.9
        elif polarity == "negative":
            scores["sadness"] = 0.9
        else:
            scores["joy"] = 0.4
            scores["sadness"] = 0.4

        final_scores = suppress_secondary_emotions(scores)
        dominant = max(final_scores, key=final_scores.get)
        intensity = final_scores[dominant]

        if original_text and intensity >= _NEUTRAL_THRESHOLD:
            emphasis = original_text.count('!') + original_text.count('?') * 0.5
            if emphasis >= 2:
                intensity = min(1.0, intensity + 0.15)

        if intensity < _NEUTRAL_THRESHOLD or len(text.split()) <= _SHORT_TEXT_MAX_WORDS:
            dominant = "neutral"

        severity = intensity_to_severity(intensity)
        return {
            "primary": dominant,
            "severity": severity,
            "intensity": round(intensity, 3),
            "all": {k: round(v, 3) for k, v in final_scores.items()},
        }

    # Preprocess text (with optional negation handling)
    text_proc = handle_negation(text)

    # Tokenize input text for BERT
    inputs = _bert_tokenizer(
        text_proc,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(_model_device)

    # BERT Forward Pass
    with torch.no_grad():
        outputs = _bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    # Map output probabilities to emotion dictionary
    raw_scores = {}
    for idx, prob in enumerate(probs):
        label_name = _id2label.get(idx, str(idx)).lower()
        raw_scores[label_name] = float(prob)

    # Extract primary 4 emotions (anger, fear, joy, sadness) + mapped emotions
    target_scores = {}
    for emo in EMOTIONS:
        target_scores[emo] = raw_scores.get(emo, 0.0)

    # Map "love" into joy boost if present
    if "love" in raw_scores:
        target_scores["joy"] = max(target_scores["joy"], raw_scores["love"])

    # Polarity gating for safety and logical consistency
    polarity = get_polarity(text_proc)
    gated_scores = polarity_gate(target_scores, polarity)

    # Secondary emotion suppression
    final_scores = suppress_secondary_emotions(gated_scores)

    # Select primary dominant emotion
    dominant = max(final_scores, key=final_scores.get)
    intensity = clip_intensity(final_scores[dominant])

    # Punctuation/Emphasis boost from original text
    if original_text and intensity >= _NEUTRAL_THRESHOLD:
        emphasis = original_text.count('!') + original_text.count('?') * 0.5
        if emphasis >= 2:
            intensity = min(1.0, intensity + 0.15)

    # Neutral override check for short or low-confidence inputs
    if intensity < _NEUTRAL_THRESHOLD or len(text.split()) <= _SHORT_TEXT_MAX_WORDS:
        severity = intensity_to_severity(intensity)
        return {
            "primary": "neutral",
            "severity": severity,
            "intensity": round(intensity, 3),
            "all": {k: round(clip_intensity(v), 3) for k, v in final_scores.items()},
        }

    severity = intensity_to_severity(intensity)
    return {
        "primary": dominant,
        "severity": severity,
        "intensity": round(intensity, 3),
        "all": {k: round(clip_intensity(v), 3) for k, v in final_scores.items()},
    }


if __name__ == "__main__":
    test_texts = [
        "I am so happy and excited today!",
        "This makes me incredibly angry and annoyed.",
        "I am afraid of what might happen next.",
        "I feel deeply saddened by the recent loss.",
        "Hi",
    ]
    print("Testing BERT Emotion Detector...")
    for t in test_texts:
        res = detect_emotion_bert(t)
        print(f"\nText: '{t}'")
        print(f"Primary: {res['primary']} | Intensity: {res['intensity']} | Severity: {res['severity']}")
        print(f"All Scores: {res['all']}")
