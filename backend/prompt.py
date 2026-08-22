import re

# -----------------------------
# Affirmation Detection
# -----------------------------

AFFIRM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"stop asking",
        r"stop asking questions",
        r"no questions?",
        r"dont ask",
        r"don't ask",
        r"dont ask questions",
        r"don't ask questions",
        r"just reassure",
        r"just support",
        r"just tell me",
        r"only support",
        r"wish me luck",
        r"no advice",
        r"i just need (support|reassurance|encouragement)",
        r"just need affirmation",
        r"i just want affirmation",
    ]
]


def wants_affirmation_only(text: str, history=None) -> bool:
    text = text.lower().strip()

    # Check current message
    if any(pattern.search(text) for pattern in AFFIRM_PATTERNS):
        return True

    history = history or []

    # Check recent user messages
    for item in reversed(history[-6:]):
        if item.get("role") != "user":
            continue

        content = (item.get("content") or item.get("text", "")).lower().strip()

        if any(pattern.search(content) for pattern in AFFIRM_PATTERNS):
            return True

    return False


# -----------------------------
# Prompt Builder
# -----------------------------

def build_prompt(user_text, emotion_info, history=None, affirm_only=False,
                 original_text=None, facial_emotion=None):

    user_text = user_text.strip()
    emotion_info = emotion_info or {}

    primary = (emotion_info.get("primary") or "neutral")
    severity = (emotion_info.get("severity") or "low")
    intensity = emotion_info.get("intensity", 0.0)

    # ── Neutral vs emotional prompt paths ──────────────────────────
    if primary == "neutral" or intensity < 0.4:
        # LOW / NO EMOTION → respond naturally, no emotional framing
        emotion_block = """
The user's message does not show strong emotion.
Respond naturally and conversationally.
Do NOT assume the user is upset, stressed, or struggling.
Do NOT offer emotional support unless the user explicitly asks for it.
"""
        tone = "Be friendly, natural, and conversational."
    else:
        # MEANINGFUL EMOTION detected
        tone = "Be calm, validating, and kind."
        if severity in {"medium", "high"}:
            tone = "Be extra gentle, grounding, and validating."

        # Use hedged language — "might be feeling" not "is feeling"
        emotion_block = f"""
The user might be feeling {primary} (confidence: {intensity:.2f}, severity: {severity}).
Respond with appropriate empathy — but only to what they actually said.
"""

        # Emotion-specific guidance
        if primary == "anxiety":
            emotion_block += """
- Offer gentle reassurance
- Normalize uncertainty
"""
        elif primary == "sadness":
            emotion_block += """
- Validate their feelings
- Avoid jumping straight to solutions
"""
        elif primary in ("frustration", "anger"):
            emotion_block += """
- Acknowledge the difficulty calmly
- Stay non-judgmental
"""
        elif primary == "fear":
            emotion_block += """
- Offer reassurance
- Help ground them
"""

    if facial_emotion:
        emotion_block += f"\nAdditionally, a facial expression scan detected the user looks: {facial_emotion.upper()}."

    history = history or []

    # Limit history for token efficiency
    recent_history = history[-4:]

    history_lines = []
    for item in recent_history:
        role = item.get("role", "user")
        content = item.get("content") or item.get("text", "")
        history_lines.append(f"{role.upper()}: {content}")

    history_block = "\n".join(history_lines) or "None"

    # Question behavior
    if affirm_only:
        question_rule = """
Conversation mode: affirmation_only = True
- NEVER ask questions
- Provide reassurance and encouragement
- End with a supportive statement
"""
    else:
        question_rule = """
- You may ask ONE gentle follow-up question if helpful
- Do NOT ask multiple questions
- Do NOT interrogate the user
"""

    # Translation context (toned down — no amplification instruction)
    translation_note = ""
    if original_text and original_text != user_text:
        translation_note = """
Note: The user wrote in their native language. The text above is a translation.
Respond to the translated meaning as-is. Do not add emotional weight that isn't there.
"""

    prompt = f"""
You are a calm, emotionally intelligent AI companion.

{emotion_block}

Tone: {tone}

CRITICAL RULES:
- Keep replies SHORT (2–4 sentences)
- Sound natural and human
- Respond ONLY to what the user explicitly says
- Do NOT infer deeper psychological meaning unless clearly expressed
- Do NOT over-interpret simple or factual statements
- Do NOT assume hidden distress behind ordinary messages
- Never argue, lecture, or over-explain

{question_rule}
{translation_note}
Conversation history:
{history_block}

User message:
"{user_text}"

Respond like a caring friend — but stay grounded in what was actually said.
"""

    return prompt
