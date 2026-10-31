"""
ASL Service (Singleton Model & Inference Coordinator)
====================================================
Coordinates model loading, landmark extraction, normalization,
per-session state management, and prediction pipelines.
"""

import base64
import logging
import pickle
import time
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import torch

from .config import (
    ALPHABET_BEST_MODEL,
    ALPHABET_SCALER_PATH,
    WORD_BEST_MODEL,
    WORD_NORM_MEAN,
    WORD_NORM_STD,
    DEVICE,
    MLP_INPUT_DIM,
    TRANSFORMER_INPUT_DIM,
    TRANSFORMER_SEQ_LEN,
    WORD_CONFIDENCE_HIGH,
    WORD_CONFIDENCE_MED,
)
from .alphabet_model import FingerspellingMLP
from .word_model import PoseTransformer
from .landmark_extractor import LandmarkExtractor
from .session_manager import SessionManager, ASLSession

logger = logging.getLogger(__name__)


class ASLService:
    """
    Singleton service maintaining loaded models and dispatching frame inference.
    """

    _instance: Optional["ASLService"] = None

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        logger.info("Initializing ASLService on device: %s", self.device)

        # MediaPipe landmark extractor
        self.extractor: Optional[LandmarkExtractor] = None
        self._init_extractor()

        # Fingerspelling MLP Model & Preprocessor
        self.fs_model: Optional[FingerspellingMLP] = None
        self.fs_scaler: Optional[Any] = None
        self.fs_label_map: Dict[str, int] = {}
        self.fs_idx_to_class: Dict[int, str] = {}
        self._load_alphabet_model()

        # Word Pose Transformer & Normalizer
        self.word_model: Optional[PoseTransformer] = None
        self.word_norm_mean: Optional[np.ndarray] = None
        self.word_norm_std: Optional[np.ndarray] = None
        self.word_label_map: Dict[str, int] = {}
        self.word_idx_to_class: Dict[int, str] = {}
        self._load_word_model()

        # Session Manager
        self.session_manager = SessionManager()

        logger.info("ASLService initialized successfully.")

    @classmethod
    def get_instance(cls) -> "ASLService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_extractor(self):
        try:
            self.extractor = LandmarkExtractor()
        except Exception as e:
            logger.error("Failed to initialize MediaPipe landmark extractor: %s", e)
            self.extractor = None

    def _load_alphabet_model(self):
        if not ALPHABET_BEST_MODEL.exists():
            logger.warning("Fingerspelling checkpoint not found at %s", ALPHABET_BEST_MODEL)
            return

        try:
            ckpt = torch.load(ALPHABET_BEST_MODEL, map_location=self.device, weights_only=False)
            model = FingerspellingMLP(ckpt["input_dim"], ckpt["num_classes"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.fs_model = model.to(self.device)
            self.fs_label_map = ckpt.get("label_map", {})
            self.fs_idx_to_class = {v: k for k, v in self.fs_label_map.items()}

            if ALPHABET_SCALER_PATH.exists():
                with open(ALPHABET_SCALER_PATH, "rb") as f:
                    self.fs_scaler = pickle.load(f)

            logger.info("Fingerspelling MLP loaded (%d classes).", ckpt.get("num_classes", 29))
        except Exception as e:
            logger.error("Failed loading Fingerspelling model: %s", e)

    def _load_word_model(self):
        if not WORD_BEST_MODEL.exists():
            logger.warning("Word model checkpoint not found at %s", WORD_BEST_MODEL)
            return

        try:
            ckpt = torch.load(WORD_BEST_MODEL, map_location=self.device, weights_only=False)
            model = PoseTransformer(
                input_dim=ckpt["input_dim"],
                seq_len=ckpt["seq_len"],
                d_model=ckpt["d_model"],
                n_heads=ckpt["n_heads"],
                n_layers=ckpt["n_layers"],
                d_ff=ckpt["d_ff"],
                num_classes=ckpt["num_classes"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.word_model = model.to(self.device)
            self.word_label_map = ckpt.get("label_map", {})
            self.word_idx_to_class = {v: k for k, v in self.word_label_map.items()}

            if WORD_NORM_MEAN.exists() and WORD_NORM_STD.exists():
                self.word_norm_mean = np.load(WORD_NORM_MEAN).squeeze()
                self.word_norm_std = np.load(WORD_NORM_STD).squeeze()

            logger.info("Pose Transformer loaded (%d classes).", ckpt.get("num_classes", 200))
        except Exception as e:
            logger.error("Failed loading Word model: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Return operational health of models and device."""
        return {
            "alphabet_model_available": self.fs_model is not None,
            "alphabet_scaler_available": self.fs_scaler is not None,
            "alphabet_classes": len(self.fs_idx_to_class),
            "word_model_available": self.word_model is not None,
            "word_norm_available": self.word_norm_mean is not None,
            "word_classes": len(self.word_idx_to_class),
            "mediapipe_available": self.extractor is not None,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "active_sessions": self.session_manager.active_sessions_count(),
            "ready": bool(self.fs_model and self.word_model and self.extractor),
        }

    def decode_frame(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image string to RGB NumPy array."""
        try:
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error("Failed to decode image frame: %s", e)
            return None

    def predict_letter(self, hand_lm: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Inference for static alphabet letters from 126 hand landmarks."""
        if self.fs_model is None or self.fs_scaler is None:
            return "", 0.0, []

        has_hand = bool(float(np.sum(np.abs(hand_lm))) > 0)
        if not has_hand:
            return "", 0.0, []

        try:
            norm_lm = self.fs_scaler.transform(hand_lm.reshape(1, -1))
            t = torch.tensor(norm_lm, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.fs_model(t), dim=-1)
                k = min(5, probs.size(1))
                topk_conf, topk_idx = torch.topk(probs, k=k, dim=-1)

            top1_char = self.fs_idx_to_class.get(int(topk_idx[0, 0].item()), "")
            top1_conf = float(topk_conf[0, 0].item())

            top_preds = [
                {
                    "label": self.fs_idx_to_class.get(int(topk_idx[0, i].item()), ""),
                    "confidence": float(round(topk_conf[0, i].item(), 3)),
                }
                for i in range(k)
            ]
            return top1_char, top1_conf, top_preds
        except Exception as e:
            logger.error("Error during predict_letter: %s", e)
            return "", 0.0, []

    def predict_word(
        self, sequence: List[np.ndarray]
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Inference for temporal signs from 64-frame sequence of 225 landmarks."""
        if self.word_model is None or self.word_norm_mean is None:
            return "?", 0.0, []

        try:
            seq_arr = np.array(sequence, dtype=np.float32)
            seq_norm = (seq_arr - self.word_norm_mean) / self.word_norm_std
            t = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.word_model(t)
                probs = torch.softmax(logits, dim=-1)
                k = min(5, probs.size(1))
                topk_conf, topk_idx = torch.topk(probs, k=k, dim=-1)

            top1_word = self.word_idx_to_class.get(int(topk_idx[0, 0].item()), "?")
            top1_conf = float(topk_conf[0, 0].item())

            top5 = [
                {
                    "label": self.word_idx_to_class.get(int(topk_idx[0, i].item()), "?"),
                    "confidence": float(round(topk_conf[0, i].item(), 3)),
                }
                for i in range(k)
            ]
            return str(top1_word), top1_conf, top5
        except Exception as e:
            logger.error("Error during predict_word: %s", e)
            return "?", 0.0, []

    def process_frame(
        self, session_id: str, frame_rgb: np.ndarray, mode: str = "alphabet"
    ) -> Dict[str, Any]:
        """
        Process a webcam frame within a user session context.
        """
        session = self.session_manager.get_or_create_session(session_id=session_id, mode=mode)
        session.set_mode(mode)
        now = time.time()

        if self.extractor is None:
            return {
                "success": False,
                "error": "Landmark extractor not initialized",
                "mode": mode,
            }

        # ── Landmark Extraction ──────────────────────────────────────────
        hand_lm = self.extractor.extract_hand_landmarks(frame_rgb)
        full_lm = self.extractor.extract_full_landmarks(frame_rgb)

        has_hand = bool(float(np.sum(np.abs(hand_lm))) > 0)
        session.sentence_builder.update_hand_presence(has_hand, now=now)
        is_moving = session.update_motion(full_lm, now=now)

        # Append to temporal sequence buffer for word recognition
        session.seq_buffer.append(full_lm)

        # ── Mode-Specific Recognition ───────────────────────────────────
        if session.mode == "alphabet":
            letter, conf, top_preds = self.predict_letter(hand_lm)
            accepted, hold_progress = session.sentence_builder.feed_letter(
                letter, conf, now=now
            )

            # A prediction is stable if hold threshold reached or progressing high
            stable = bool(hold_progress >= 1.0 or accepted)

            return {
                "success": True,
                "session_id": session.session_id,
                "mode": "alphabet",
                "prediction": letter.upper() if letter else "",
                "confidence": float(round(conf, 3)),
                "stable": stable,
                "accepted": accepted,
                "hold_progress": float(round(hold_progress, 2)),
                "has_hand": has_hand,
                "top_predictions": top_preds,
                "current_word": session.sentence_builder.get_current_word(),
                "sentence": session.sentence_builder.get_sentence(),
            }

        else:
            # ── Word Mode ───────────────────────────────────────────────
            word, conf, top5 = "", 0.0, []
            stable = False
            accepted = False

            if len(session.seq_buffer) == TRANSFORMER_SEQ_LEN and has_hand:
                word, conf, top5 = self.predict_word(list(session.seq_buffer))

                # Word boundary acceptance: High confidence & signing motion settled
                if conf >= WORD_CONFIDENCE_HIGH:
                    stable = True
                    if not session.is_signing:
                        accepted = session.sentence_builder.feed_word(
                            word, conf, is_signing=session.is_signing
                        )
                elif conf >= WORD_CONFIDENCE_MED:
                    stable = False

            return {
                "success": True,
                "session_id": session.session_id,
                "mode": "word",
                "prediction": word,
                "confidence": float(round(conf, 3)),
                "stable": stable,
                "accepted": accepted,
                "has_hand": has_hand,
                "is_signing": bool(session.is_signing),
                "top_predictions": top5,
                "buffer_len": len(session.seq_buffer),
                "buffer_ready": len(session.seq_buffer) == TRANSFORMER_SEQ_LEN,
                "sentence": session.sentence_builder.get_sentence(),
            }

