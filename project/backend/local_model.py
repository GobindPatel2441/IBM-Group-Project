"""
LLM gateway — routes to Ollama or OpenAI depending on the config.

Other modules only need to import from here:
    from .local_model import generate_clean_response, generate_clean_response_stream, OLLAMA_URL
"""
import json
import logging
import requests

from .config import LLM_PROVIDER, OLLAMA_URL, OLLAMA_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

MODEL_NAME = OLLAMA_MODEL

# Re-export OLLAMA_URL so existing imports in app.py keep working
__all__ = [
    "generate_clean_response",
    "generate_clean_response_stream",
    "OLLAMA_URL",
]


# ── Ollama implementations ─────────────────────────────────────────────

def _ollama_stream(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 120,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
    except requests.exceptions.RequestException:
        yield "Sorry, I'm having a little trouble responding right now. Can you try again?"


def _ollama_full(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 120,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException:
        return "Sorry, I'm having a little trouble responding right now. Can you try again?"


# ── Public API (auto-routes based on LLM_PROVIDER) ────────────────────

def generate_clean_response_stream(prompt: str):
    """
    Sends an emotion-conditioned prompt to the configured LLM
    and streams the generated response text back.
    """
    if LLM_PROVIDER == "openai":
        from .openai_model import generate_clean_response_stream as _openai_stream
        yield from _openai_stream(prompt)
    else:
        yield from _ollama_stream(prompt)


def generate_clean_response(prompt: str) -> str:
    """
    Sends an emotion-conditioned prompt to the configured LLM
    and returns the generated response text.
    """
    if LLM_PROVIDER == "openai":
        from .openai_model import generate_clean_response as _openai_full
        return _openai_full(prompt)
    else:
        return _ollama_full(prompt)
