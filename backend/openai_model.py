"""
OpenAI / ChatGPT integration — drop-in alternative to the Ollama backend.

Exposes the same two functions as local_model.py:
  - generate_clean_response(prompt) -> str
  - generate_clean_response_stream(prompt) -> generator[str]
"""
import logging
from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

# Shared generation parameters (mirrors Ollama's settings)
_GENERATION_KWARGS = {
    "model": OPENAI_MODEL,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 200,
}


def generate_clean_response_stream(prompt: str):
    """
    Streams the ChatGPT response token-by-token, yielding text chunks
    exactly like the Ollama streamer so the rest of the pipeline is unchanged.
    """
    try:
        stream = _client.chat.completions.create(
            **_GENERATION_KWARGS,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as e:
        logger.error("OpenAI streaming error: %s", e)
        yield "Sorry, I'm having trouble reaching ChatGPT right now. Please check your API key and try again."


def generate_clean_response(prompt: str) -> str:
    """
    Non-streaming ChatGPT call — returns the full response as a string.
    """
    try:
        response = _client.chat.completions.create(
            **_GENERATION_KWARGS,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        return "Sorry, I'm having trouble reaching ChatGPT right now. Please check your API key and try again."
