"""
ASL Module Configuration for Sentix
===================================
Defines artifact paths, device auto-selection, architecture shapes,
and runtime thresholds for alphabet and word recognition.
"""

from pathlib import Path
import os
import torch

# Base directory for the ASL package
ASL_DIR = Path(__file__).resolve().parent
MODELS_DIR = ASL_DIR / "models"

# ── Model Artifact Paths ──────────────────────────────────────────────────
ALPHABET_MODEL_DIR = MODELS_DIR / "alphabet"
WORD_MODEL_DIR = MODELS_DIR / "word"
MEDIAPIPE_MODEL_DIR = MODELS_DIR / "mediapipe"

# Alphabet Model Artifacts
ALPHABET_BEST_MODEL = ALPHABET_MODEL_DIR / "best_model.pth"
ALPHABET_SCALER_PATH = ALPHABET_MODEL_DIR / "scaler.pkl"
ALPHABET_LABEL_MAP = ALPHABET_MODEL_DIR / "label_map.json"
ALPHABET_CONFIG_JSON = ALPHABET_MODEL_DIR / "config.json"

# Word Model Artifacts
WORD_BEST_MODEL = WORD_MODEL_DIR / "best_model.pth"
WORD_LABEL_MAP = WORD_MODEL_DIR / "label_map.json"
WORD_NORM_MEAN = WORD_MODEL_DIR / "norm_mean.npy"
WORD_NORM_STD = WORD_MODEL_DIR / "norm_std.npy"
WORD_CONFIG_JSON = WORD_MODEL_DIR / "config.json"

# MediaPipe Task Files
HAND_LANDMARKER_TASK = MEDIAPIPE_MODEL_DIR / "hand_landmarker.task"
POSE_LANDMARKER_TASK = MEDIAPIPE_MODEL_DIR / "pose_landmarker.task"

# ── Hardware / Device Configuration ──────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Alphabet (Fingerspelling MLP) Architecture & Settings ───────────────
MLP_INPUT_DIM = 126  # 21 right hand * 3 + 21 left hand * 3
MLP_NUM_CLASSES = 29
FS_CONFIDENCE_THRESHOLD = 0.70
FS_HOLD_FRAMES = 8
LETTER_PAUSE_SECONDS = 0.8

# ── Word (Pose Transformer) Architecture & Settings ─────────────────────
TRANSFORMER_INPUT_DIM = 225  # right hand (63) + left hand (63) + pose (99)
TRANSFORMER_SEQ_LEN = 64
TRANSFORMER_D_MODEL = 512
TRANSFORMER_N_HEADS = 8
TRANSFORMER_N_LAYERS = 6
TRANSFORMER_D_FF = 2048
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_NUM_CLASSES = 200  # Exact trained vocabulary count

WORD_CONFIDENCE_HIGH = 0.20
WORD_CONFIDENCE_MED = 0.10
MOTION_THRESHOLD = 0.005
WORD_PAUSE_SECONDS = 2.0

# ── Session Management Settings ─────────────────────────────────────────
SESSION_TTL_SECONDS = 600  # 10 minutes inactivity timeout
MAX_SENTENCE_WORDS = 30

