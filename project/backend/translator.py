"""
translator.py — Bidirectional translation wrapper for the empathy companion.

Uses NLLB-200-distilled-600M for translating between English and 11 Indian
languages.  The model is pre-warmed in a background thread so it's ready
before the first user request.

Public API
----------
    init_translator()                           — force-load (optional)
    translate_to_english(text, src_lang)         — Indian lang → English
    translate_from_english(text, tgt_lang)       — English → Indian lang
    detect_input_language(text)                  — Unicode-range heuristic
"""

import os
import re
import logging
import threading
from pathlib import Path
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parents[1]
_TRANSLATOR_DIR = _BASE_DIR / "Translator Model" / "translator"
_MODEL_DIR = _TRANSLATOR_DIR / "model_simple"
_HF_HOME = Path(os.environ.setdefault(
    "HF_HOME", str(_TRANSLATOR_DIR / ".hf_home")
))
_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ---------------------------------------------------------------------------
# Language mappings
# ---------------------------------------------------------------------------
LANG_MAP: dict[str, str] = {
    "Hindi":     "hin_Deva",
    "Odia":      "ory_Orya",
    "Oriya":     "ory_Orya",
    "Marathi":   "mar_Deva",
    "Telugu":    "tel_Telu",
    "Tamil":     "tam_Taml",
    "Malayalam":  "mal_Mlym",
    "Kannada":   "kan_Knda",
    "Gujarati":  "guj_Gujr",
    "Punjabi":   "pan_Guru",
    "Bengali":   "ben_Beng",
    "Assamese":  "asm_Beng",
}

SPEECH_LANG_MAP: dict[str, str] = {
    "English":    "en-US",
    "Hindi":      "hi-IN",
    "Bengali":    "bn-IN",
    "Tamil":      "ta-IN",
    "Telugu":     "te-IN",
    "Marathi":    "mr-IN",
    "Gujarati":   "gu-IN",
    "Kannada":    "kn-IN",
    "Malayalam":  "ml-IN",
    "Punjabi":    "pa-IN",
    "Odia":       "or-IN",
    "Assamese":   "as-IN",
}

SUPPORTED_LANGUAGES: list[str] = ["English"] + sorted(LANG_MAP.keys())

_ENG_LATN = "eng_Latn"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None
_loaded = False
_loading_lock = threading.Lock()

# [UPGRADE 4] Only cache texts shorter than this to bound memory
_MAX_CACHEABLE_LEN = 500


# ---------------------------------------------------------------------------
# Unicode-range language detection
# ---------------------------------------------------------------------------
def detect_input_language(text: str) -> str:
    """Return a language name based on Unicode script ranges."""
    checks = [
        ("\u0900", "\u097F", "Hindi"),
        ("\u0B00", "\u0B7F", "Odia"),
        ("\u0C00", "\u0C7F", "Telugu"),
        ("\u0B80", "\u0BFF", "Tamil"),
        ("\u0D00", "\u0D7F", "Malayalam"),
        ("\u0C80", "\u0CFF", "Kannada"),
        ("\u0A80", "\u0AFF", "Gujarati"),
        ("\u0A00", "\u0A7F", "Punjabi"),
        ("\u0980", "\u09FF", "Bengali"),
    ]
    for lo, hi, lang in checks:
        if any(lo <= c <= hi for c in text):
            return lang
    return "English"


# ---------------------------------------------------------------------------
# Model loading — thread-safe with double-checked locking
# ---------------------------------------------------------------------------
def init_translator() -> None:
    """Load the NLLB model.  Thread-safe, idempotent."""
    global _model, _tokenizer, _device, _loaded
    if _loaded:
        return

    with _loading_lock:
        if _loaded:          # double-check after acquiring lock
            return

        _HF_HOME.mkdir(parents=True, exist_ok=True)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_kwargs: dict = {"cache_dir": str(_HF_HOME)}
        if _HF_TOKEN:
            load_kwargs["token"] = _HF_TOKEN

        model_source = str(_MODEL_DIR) if (_MODEL_DIR / "config.json").exists() \
            else "facebook/nllb-200-distilled-600M"

        logger.info("Loading NLLB translator from %s …", model_source)

        _tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)

        if torch.cuda.is_available():
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                model_source, device_map="auto",
                quantization_config=qconfig, **load_kwargs,
            )
        else:
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                model_source, **load_kwargs
            ).to(_device)

        _model.eval()
        _loaded = True
        logger.info("NLLB translator ready (device=%s).", _device)


def _ensure_loaded() -> None:
    if not _loaded:
        init_translator()


# ---------------------------------------------------------------------------
# [UPGRADE 2] Background preloading — non-blocking warmup
# ---------------------------------------------------------------------------
def _background_preload():
    try:
        init_translator()
        logger.info("Background translator preload complete.")
    except Exception:
        logger.warning("Background preload failed — will retry on first request.")

threading.Thread(target=_background_preload, daemon=True).start()


# ---------------------------------------------------------------------------
# Internal inference
# ---------------------------------------------------------------------------
def _get_model_device():
    if hasattr(_model, "device") and _model.device is not None:
        return _model.device
    return next(_model.parameters()).device


def _get_forced_bos_id(lang_code: str) -> int:
    if hasattr(_tokenizer, "lang_code_to_id"):
        return _tokenizer.lang_code_to_id[lang_code]
    return _tokenizer.convert_tokens_to_ids(lang_code)


def _translate(text: str, src_code: str, tgt_code: str) -> str:
    """Run a single NLLB translation."""
    _tokenizer.src_lang = src_code
    active_device = _get_model_device()

    inputs = _tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(active_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            forced_bos_token_id=_get_forced_bos_id(tgt_code),
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )

    result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _post_process(result)


def _post_process(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) > 1:
        text = text[0].upper() + text[1:]
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"([.,!?])(?=[^\s\d])", r"\1 ", text)
    return text


# ---------------------------------------------------------------------------
# [UPGRADE 4] Cached translation helpers — only for short texts
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _translate_to_en_cached(text: str, source_language: str) -> str:
    return _translate(text, LANG_MAP[source_language], _ENG_LATN)


@lru_cache(maxsize=256)
def _translate_from_en_cached(text: str, target_language: str) -> str:
    return _translate(text, _ENG_LATN, LANG_MAP[target_language])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def translate_to_english(text: str, source_language: str) -> str:
    """Translate text from *source_language* to English."""
    if source_language == "English":
        return text

    src_code = LANG_MAP.get(source_language)
    if not src_code:
        logger.warning("Unsupported source language: %s", source_language)
        return text

    try:
        _ensure_loaded()
        if len(text) <= _MAX_CACHEABLE_LEN:
            return _translate_to_en_cached(text, source_language)
        return _translate(text, src_code, _ENG_LATN)
    except Exception:
        logger.exception("translate_to_english failed — returning original text")
        return text


def translate_from_english(text: str, target_language: str) -> str:
    """Translate English text into *target_language*."""
    if target_language == "English":
        return text

    tgt_code = LANG_MAP.get(target_language)
    if not tgt_code:
        logger.warning("Unsupported target language: %s", target_language)
        return text

    try:
        _ensure_loaded()
        if len(text) <= _MAX_CACHEABLE_LEN:
            return _translate_from_en_cached(text, target_language)
        return _translate(text, _ENG_LATN, tgt_code)
    except Exception:
        logger.exception("translate_from_english failed — returning original text")
        return text
