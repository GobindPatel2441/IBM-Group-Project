import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    _sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    _sia = SentimentIntensityAnalyzer()


def intensity_to_severity(x):
    if x < 0.2: return 0
    if x < 0.4: return 1
    if x < 0.6: return 2
    if x < 0.8: return 3
    return 4

def clip_intensity(x):
    return max(0.0, min(1.0, float(x)))

def suppress_secondary_emotions(scores, threshold=0.6):
    """
    scores = {'anger': x, 'fear': y, ...}
    """
    dominant_emotion = max(scores, key=scores.get)
    dominant_value = scores[dominant_emotion]

    adjusted = scores.copy()

    if dominant_value >= threshold:
        for emo in scores:
            if emo != dominant_emotion:
                adjusted[emo] *= 0.6   # suppress others

    return adjusted

def handle_negation(text):
    negations = ["not", "never", "no", "n't"]
    tokens = text.split()

    result = []
    negate = False

    for t in tokens:
        if t.lower() in negations:
            negate = True
            result.append(t)
            continue

        if negate:
            result.append("NOT_" + t)
            negate = False
        else:
            result.append(t)

    return " ".join(result)

from nltk.sentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

def get_polarity(text):
    score = _sia.polarity_scores(text)["compound"]
    if score >= 0.3:
        return "positive"
    elif score <= -0.3:
        return "negative"
    else:
        return "neutral"
    
def polarity_gate(scores, polarity):
    adjusted = scores.copy()

    if polarity == "positive":
        if "sadness" in adjusted: adjusted["sadness"] *= 0.3
        if "anger" in adjusted: adjusted["anger"] *= 0.4
        if "fear" in adjusted: adjusted["fear"] *= 0.4

    elif polarity == "negative":
        if "joy" in adjusted: adjusted["joy"] *= 0.3
        if "love" in adjusted: adjusted["love"] *= 0.3

    return adjusted

def wants_affirmation_only(text: str) -> bool:
    text = text.lower()
    triggers = [
        "stop asking",
        "no questions",
        "just tell me",
        "just reassure me",
        "i just want affirmation",
        "just say something",
        "don't ask",
        "wish me luck",
        "just support me"
    ]
    return any(t in text for t in triggers)

# ── EMOJI & COLLOQUIAL AFFECT NORMALIZATION ──────────────────────────────
_EMOJI_SLANG_MAP = {
    "😭": " crying heartbroken ",
    "😢": " crying sad ",
    "🥺": " pleading vulnerable ",
    "😞": " sad disappointed ",
    "😔": " depressed down ",
    "😡": " furious angry ",
    "🤬": " swearing outraged ",
    "😠": " mad irritated ",
    "😱": " shocked terrified ",
    "😨": " scared panicked ",
    "😰": " anxious nervous ",
    "😊": " warm happy ",
    "😄": " cheerful delighted ",
    "❤️": " loving grateful ",
    "💖": " caring affectionate ",
    "✨": " hopeful inspired ",
    "🎉": " celebrating excited ",
    "smh": " shaking my head disappointed ",
    "fml": " feeling terrible distressed ",
    "lmao": " laughing delighted ",
    "rofl": " laughing happy ",
    "lol": " laughing amused ",
    "idk": " confused uncertain ",
    "tbh": " to be honest ",
}

def normalize_text_affect(text: str) -> str:
    """
    Expands emojis, abbreviations, and affective slang into canonical English
    phrases to maximize emotion classifier fidelity.
    """
    if not text:
        return ""
    proc = text
    for token, expansion in _EMOJI_SLANG_MAP.items():
        if token in proc:
            proc = proc.replace(token, expansion)
    return " ".join(proc.split())

def compute_valence_arousal(scores: dict) -> dict:
    """
    Computes Russell's Circumplex Model coordinates from categorical emotions:
    - Valence: [-1.0 (unpleasant/negative) to +1.0 (pleasant/positive)]
    - Arousal: [0.0 (calm/deactivated) to 1.0 (excited/activated)]
    """
    joy = float(scores.get("joy", 0.0))
    love = float(scores.get("love", 0.0))
    anger = float(scores.get("anger", 0.0))
    fear = float(scores.get("fear", 0.0))
    sadness = float(scores.get("sadness", 0.0))

    # Valence: positive emotions minus negative emotions
    pos_weight = (joy * 1.0) + (love * 0.9)
    neg_weight = (anger * 0.8) + (fear * 0.85) + (sadness * 0.95)
    valence = clip_intensity(pos_weight - neg_weight)
    # Scale from [0, 1] difference to [-1.0, 1.0]
    valence_bipolar = max(-1.0, min(1.0, float(pos_weight - neg_weight)))

    # Arousal: activation level
    arousal = (anger * 0.85) + (fear * 0.80) + (joy * 0.65) + (sadness * 0.25)
    arousal = clip_intensity(arousal)

    return {
        "valence": round(valence_bipolar, 3),
        "arousal": round(arousal, 3),
    }

