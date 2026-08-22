from flask import Flask, request, jsonify, Response
import json
import re
import time
import base64
import numpy as np
import cv2
import os
from deepface import DeepFace
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .utils import wants_affirmation_only

from .emotion_detector import detect_emotion
try:
    from .prompt import build_prompt, wants_affirmation_only
except ImportError:
    from .prompt import build_prompt

    def wants_affirmation_only(_text: str) -> bool:
        return False
from .safety import safety_level, safety_response
from .local_model import generate_clean_response, generate_clean_response_stream, OLLAMA_URL
from .translator import (
    translate_to_english,
    translate_from_english,
    detect_input_language,
    SUPPORTED_LANGUAGES,
    SPEECH_LANG_MAP,
)
import requests
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Sentence splitter for streaming translation
# ---------------------------------------------------------------------------
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
_FORCE_FLUSH_LEN = 120


def _split_sentences(text: str):
    """Return (complete_sentences_list, incomplete_remainder).

    If no sentence boundary is found and the buffer exceeds
    _FORCE_FLUSH_LEN characters, the entire buffer is flushed
    as a partial sentence to avoid stalling the stream.
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    if not parts:
        return [], text

    last = parts[-1]
    if last and not re.search(r'[.!?]\s*$', last):
        complete = [p for p in parts[:-1] if p]
        # Force-flush: if no sentence was extracted and remainder is long,
        # push the whole buffer out so the user isn't left waiting.
        if not complete and len(last) > _FORCE_FLUSH_LEN:
            return [last], ""
        return complete, last

    return [p for p in parts if p], ""


@app.route("/chat", methods=["POST"])
def chat():
    t_total = time.time()

    user_text = request.json.get("message", "")
    history = request.json.get("history", [])
    language = request.json.get("language", "English")
    image_b64 = request.json.get("image", None)

    # ── Facial Emotion Detection ────────────────────────────────────
    facial_emotion = None
    if image_b64:
        try:
            encoded_data = image_b64.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(results, list) and len(results) > 0:
                facial_emotion = results[0]['dominant_emotion']
            elif isinstance(results, dict):
                facial_emotion = results.get('dominant_emotion')
            logger.info("[Debug] Facial emotion detected: %s", facial_emotion)
        except Exception as e:
            logger.error(f"Facial emotion detection failed: {e}")

    # ── Validate language ──────────────────────────────────────────
    if language not in SUPPORTED_LANGUAGES:
        language = "English"

    # Output translation always follows the user's dropdown selection
    needs_output_translation = language != "English"

    # ── Language detection safety layer ─────────────────────────────
    original_text = user_text
    if language != "English":
        detected = detect_input_language(user_text)
        if detected == "English":
            user_text_en = user_text
            logger.info(
                "Input is English despite '%s' selected — skipping input translation",
                language,
            )
        else:
            if detected != language:
                logger.warning(
                    "Language mismatch: selected=%s detected=%s — trusting user",
                    language, detected,
                )
            t_in = time.time()
            user_text_en = translate_to_english(user_text, language)
            logger.info("[Timing] Input translation: %.2fs", time.time() - t_in)

            # Translation length safety: if output is wildly different in
            # length, the translation likely distorted meaning — fall back.
            orig_len = len(user_text.split())
            trans_len = len(user_text_en.split())
            if orig_len > 0 and trans_len > 0:
                ratio = trans_len / orig_len
                if ratio > 3.0 or ratio < 0.3:
                    logger.warning(
                        "Translation length mismatch (ratio=%.2f) — falling back to original",
                        ratio,
                    )
                    user_text_en = user_text
    else:
        user_text_en = user_text

    # ── DEBUG: pipeline transparency ──────────────────────────────────
    logger.info("[Debug] Original: %s", original_text[:120])
    logger.info("[Debug] Translated: %s", user_text_en[:120])
    logger.info("[Debug] Language: %s", language)

    # ── SAFETY CHECK (runs on English text) ────────────────────────
    level = safety_level(user_text_en)

    if level == "high":
        safe_resp = safety_response()
        if needs_output_translation:
            safe_resp = translate_from_english(safe_resp, language)
        return jsonify({
            "emotion": "distress",
            "response": safe_resp,
            "language": language,
        })

    # ── EMOTION DETECTION — pass original text for emphasis ─────────
    emotion_info = detect_emotion(user_text_en, original_text=original_text)

    # ── DEBUG: emotion transparency ─────────────────────────────────
    logger.info("[Debug] Emotion scores: %s", emotion_info.get("all", {}))
    logger.info("[Debug] Final emotion: %s (intensity=%.3f, severity=%s)",
                emotion_info.get("primary"), emotion_info.get("intensity", 0),
                emotion_info.get("severity"))

    # ── PROMPT BUILDING (English + original for empathy) ───────────
    affirm_only = wants_affirmation_only(user_text_en, history)
    prompt = build_prompt(
        user_text_en,
        emotion_info,
        history,
        affirm_only=affirm_only,
        original_text=original_text if needs_output_translation else None,
        facial_emotion=facial_emotion,
    )

    app.logger.info("Generating response...")

    # ── STREAMING RESPONSE with latency logging ────────────────────
    def generate():
        yield json.dumps({"type": "emotion", "data": emotion_info}) + "\n"

        t_llm = time.time()

        if needs_output_translation:
            # Sentence-level streaming: buffer → sentence end → translate → emit
            buffer = ""
            t_out_total = 0.0
            for chunk in generate_clean_response_stream(prompt):
                buffer += chunk
                sentences, buffer = _split_sentences(buffer)
                for sentence in sentences:
                    t_out = time.time()
                    translated = translate_from_english(sentence, language)
                    t_out_total += time.time() - t_out
                    yield json.dumps({"type": "chunk", "data": translated + " "}) + "\n"

            # Flush remaining incomplete sentence
            remainder = buffer.strip()
            if remainder:
                t_out = time.time()
                translated = translate_from_english(remainder, language)
                t_out_total += time.time() - t_out
                yield json.dumps({"type": "chunk", "data": translated}) + "\n"

            logger.info("[Timing] LLM: %.2fs | Output translation: %.2fs | Total: %.2fs",
                        time.time() - t_llm - t_out_total, t_out_total, time.time() - t_total)
        else:
            # English — stream word-by-word (unchanged)
            for chunk in generate_clean_response_stream(prompt):
                yield json.dumps({"type": "chunk", "data": chunk}) + "\n"

            logger.info("[Timing] LLM: %.2fs | Total: %.2fs",
                        time.time() - t_llm, time.time() - t_total)

    return Response(generate(), mimetype='application/x-ndjson')


@app.route("/languages", methods=["GET"])
def get_languages():
    """Return the list of supported languages and their speech codes."""
    return jsonify({
        "languages": SUPPORTED_LANGUAGES,
        "speech_codes": SPEECH_LANG_MAP,
    })


@app.route("/status", methods=["GET"])
def status():
    """Lightweight health check for the Ollama backend."""
    health_url = OLLAMA_URL.replace("/api/generate", "/api/tags")
    try:
        response = requests.get(health_url, timeout=3)
        online = response.ok
    except requests.exceptions.RequestException:
        online = False

    return jsonify({"status": "online" if online else "offline"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
