import os
import re
import tempfile
import base64
import logging
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logger = logging.getLogger(__name__)

# Load the SenseVoice model globally so it stays in memory
# Setting device to cpu by default to avoid CUDA OOM if deepface/ollama uses GPU,
# but funasr handles device automatically if specified.
try:
    _model_dir = "iic/SenseVoiceSmall"
    _sense_model = AutoModel(
        model=_model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu" 
    )
    logger.info("SenseVoice model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SenseVoice model: {e}")
    _sense_model = None

# Mapping SenseVoice emotion tags to Sentix emotion names
EMOTION_MAP = {
    "<|HAPPY|>": "joy",
    "<|SAD|>": "sadness",
    "<|ANGRY|>": "anger",
    "<|NEUTRAL|>": "neutral",
    # If there are other tags, we can add them here
}

def detect_audio_emotion(audio_b64: str) -> str:
    """
    Decodes the base64 audio, saves it to a temp file, runs SenseVoice,
    and returns the detected emotion.
    Returns None if no emotion is detected or an error occurs.
    """
    if not _sense_model:
        return None

    try:
        # Decode base64
        # Format usually is data:audio/webm;base64,....
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
            
        audio_data = base64.b64decode(audio_b64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
            
        # Run SenseVoice
        # Output format is [{'text': '<|en|><|NEUTRAL|><|Speech|><|woit|> hello'}]
        res = _sense_model.generate(
            input=tmp_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        
        # Clean up temp file
        os.remove(tmp_path)
        
        if not res or len(res) == 0:
            return None
            
        raw_text = res[0].get("text", "")
        
        # Look for emotion tags in raw text
        for tag, sentix_emotion in EMOTION_MAP.items():
            if tag in raw_text:
                logger.info(f"Audio emotion detected: {sentix_emotion} (from tag {tag})")
                return sentix_emotion
                
        logger.info(f"No specific emotion tag found in SenseVoice output: {raw_text}")
        return None
        
    except Exception as e:
        logger.error(f"Audio emotion detection failed: {e}")
        return None
