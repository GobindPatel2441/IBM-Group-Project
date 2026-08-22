import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:latest"


import json

def generate_clean_response_stream(prompt: str):
    """
    Sends an emotion-conditioned prompt to Ollama (llama3.1:8b)
    and streams the generated response text back.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 120
        }
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
        yield "Sorry, I’m having a little trouble responding right now. Can you try again?"

def generate_clean_response(prompt: str) -> str:
    """
    Sends an emotion-conditioned prompt to Ollama (llama3.1:8b)
    and returns the generated response text.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 120
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        # Hard failure fallback (very rare if Ollama is running)
        return "Sorry, I’m having a little trouble responding right now. Can you try again?"
