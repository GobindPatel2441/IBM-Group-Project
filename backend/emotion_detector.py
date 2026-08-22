import os
import joblib
from pathlib import Path

from .utils import (
    clip_intensity,
    intensity_to_severity,
    handle_negation,
    get_polarity,
    polarity_gate,
    suppress_secondary_emotions,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("EMOTION_MODEL_DIR", BASE_DIR / "Model")).resolve()

EMOTIONS = ["anger", "fear", "joy", "sadness"]

# Confidence threshold: scores below this are treated as neutral
_NEUTRAL_THRESHOLD = 0.4
# Short messages (≤ this many words) default to neutral
_SHORT_TEXT_MAX_WORDS = 4

_models = {}
_vectorizers = {}
_primary_classifier = None
_primary_vectorizer = None
_primary_classifier_checked = False


def _load_models():
    """
    Lazily load SVR models if they are present on disk.
    If model files are missing, we leave the caches empty so that
    downstream logic can fall back to a lightweight heuristic-based
    emotion detector instead of crashing.
    """
    if _models:
        return

    for emo in EMOTIONS:
        model_path = MODEL_DIR / f"{emo}_svr_model.joblib"
        vec_path = MODEL_DIR / f"{emo}_svr_vectorizer.joblib"

        # If any required file is missing, or if unpickling fails
        # (for example because sklearn is not installed in this
        # environment), we clear any partially loaded state and
        # return so that the heuristic fallback path is used.
        if not model_path.exists() or not vec_path.exists():
            _models.clear()
            _vectorizers.clear()
            return

        try:
            _models[emo] = joblib.load(model_path)
            _vectorizers[emo] = joblib.load(vec_path)
        except Exception:
            _models.clear()
            _vectorizers.clear()
            return


def _load_primary_classifier():
    global _primary_classifier_checked, _primary_classifier, _primary_vectorizer

    if _primary_classifier_checked:
        return

    _primary_classifier_checked = True

    model_path = MODEL_DIR / "primary_emotion_classifier.joblib"
    vec_path = MODEL_DIR / "primary_emotion_vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        return

    try:
        _primary_classifier = joblib.load(model_path)
        _primary_vectorizer = joblib.load(vec_path)
    except Exception:
        _primary_classifier = None
        _primary_vectorizer = None


def detect_emotion(text: str, original_text: str = None):
    _load_models()
    _load_primary_classifier()

    # If the SVR models are not available, fall back to a simple
    # polarity-based heuristic so the API still returns a valid
    # emotion payload instead of a 500 error.
    if not _models or not _vectorizers:
        polarity = get_polarity(text)

        # Start from a small baseline for all emotions.
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

        # Emphasis boost from original text (only if emotion is already meaningful)
        if original_text and intensity >= _NEUTRAL_THRESHOLD:
            emphasis = original_text.count('!') + original_text.count('?') * 0.5
            if emphasis >= 2:
                intensity = min(1.0, intensity + 0.15)

        # --- Neutral override: short text or low confidence ---
        if intensity < _NEUTRAL_THRESHOLD or len(text.split()) <= _SHORT_TEXT_MAX_WORDS:
            dominant = "neutral"
            intensity = round(intensity, 3)
            severity = intensity_to_severity(intensity)
            return {
                "primary": "neutral",
                "severity": severity,
                "intensity": intensity,
                "all": {k: round(v, 3) for k, v in final_scores.items()},
            }

        severity = intensity_to_severity(intensity)

        return {
            "primary": dominant,
            "severity": severity,
            "intensity": round(intensity, 3),
            "all": {k: round(v, 3) for k, v in final_scores.items()},
        }

    # Full model-based path
    # 1. Negation handling
    text_proc = handle_negation(text)

    # 2. Raw SVR predictions
    raw = {}
    for emo in EMOTIONS:
        vec = _vectorizers[emo].transform([text_proc])
        raw[emo] = clip_intensity(_models[emo].predict(vec)[0])

    # 3. Polarity gating
    polarity = get_polarity(text_proc)
    gated = polarity_gate(raw, polarity)

    # 4. Dominant suppression
    final_scores = suppress_secondary_emotions(gated)

    # 5. Pick dominant emotion
    dominant = max(final_scores, key=final_scores.get)
    if _primary_classifier is not None and _primary_vectorizer is not None:
        try:
            predicted = str(_primary_classifier.predict(_primary_vectorizer.transform([text_proc]))[0])
            if predicted in EMOTIONS:
                dominant = predicted
        except Exception:
            pass

    intensity = clip_intensity(final_scores[dominant])

    # Emphasis boost from original text (only if emotion is already meaningful)
    if original_text and intensity >= _NEUTRAL_THRESHOLD:
        emphasis = original_text.count('!') + original_text.count('?') * 0.5
        if emphasis >= 2:
            intensity = min(1.0, intensity + 0.15)

    # --- Neutral override: short text or low confidence ---
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
