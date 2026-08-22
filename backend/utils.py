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
        adjusted["sadness"] *= 0.3
        adjusted["anger"] *= 0.4
        adjusted["fear"] *= 0.4

    elif polarity == "negative":
        adjusted["joy"] *= 0.3

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
