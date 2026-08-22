def safety_level(text: str) -> str:
    """
    Classifies safety risk level based on explicit intent.
    Returns: 'high' or 'low'
    """
    text = text.lower()

    # Explicit self-harm / suicide intent (HIGH RISK)
    high_risk_phrases = [
        "kill myself",
        "end my life",
        "commit suicide",
        "want to die",
        "i want to die",
        "i don't want to live",
        "self harm",
        "hurt myself",
        "cut myself"
    ]

    for phrase in high_risk_phrases:
        if phrase in text:
            return "high"

    return "low"


def safety_response():
    return (
        "I'm really sorry you're feeling this way.\n\n"
        "You are not alone. Talking to someone you trust can help.\n\n"
        "United States: Call or text 988 for the Suicide & Crisis Lifeline.\n"
        "If you are in immediate danger, call your local emergency number.\n\n"
        "If you want, I can stay here and talk with you."
    )
