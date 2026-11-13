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
    WORD_CONFIDENCE_MARGIN,
)
from .alphabet_model import FingerspellingMLP
from .word_model import PoseTransformer, SpatioTemporalAttentionModel
from .landmark_extractor import LandmarkExtractor
from .session_manager import SessionManager, ASLSession

logger = logging.getLogger(__name__)


class ASLService:
    """
    Singleton service maintaining loaded models and dispatching frame inference.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ASLService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ASLService":
        """Return the singleton ASLService instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.device = DEVICE
        self.session_manager = SessionManager()
        self.extractor: Optional[LandmarkExtractor] = None

        # Alphabet Model Components
        self.fs_model: Optional[FingerspellingMLP] = None
        self.fs_scaler = None
        self.fs_label_map: Dict[str, int] = {}
        self.fs_idx_to_class: Dict[int, str] = {}

        # Word Model Components
        self.word_model = None
        self.word_norm_mean: Optional[np.ndarray] = None
        self.word_norm_std: Optional[np.ndarray] = None
        self.word_label_map: Dict[str, int] = {}
        self.word_idx_to_class: Dict[int, str] = {}

        self._init_models()
        self._initialized = True

    def _init_models(self):
        """Loads all model checkpoints and feature scalers."""
        logger.info("Initializing ASL Service on %s", self.device)
        self._load_alphabet_model()
        self._load_word_model()
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
            model_type = ckpt.get("model_type", "PoseTransformer")
            num_classes = ckpt.get("num_classes", 54)

            if model_type == "SpatioTemporalAttentionModel":
                model = SpatioTemporalAttentionModel(
                    input_dim=ckpt.get("input_dim", 225),
                    seq_len=ckpt.get("seq_len", TRANSFORMER_SEQ_LEN),
                    d_model=ckpt.get("d_model", 256),
                    hidden_dim=ckpt.get("hidden_dim", 128),
                    num_classes=num_classes,
                    n_heads=ckpt.get("n_heads", 4),
                )
            else:
                model = PoseTransformer(
                    input_dim=ckpt.get("input_dim", 225),
                    seq_len=ckpt.get("seq_len", TRANSFORMER_SEQ_LEN),
                    d_model=ckpt.get("d_model", 512),
                    n_heads=ckpt.get("n_heads", 8),
                    n_layers=ckpt.get("n_layers", 6),
                    d_ff=ckpt.get("d_ff", 2048),
                    num_classes=num_classes,
                )

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.word_model = model.to(self.device)
            self.word_label_map = ckpt.get("label_map", {})
            self.word_idx_to_class = {v: k for k, v in self.word_label_map.items()}

            if WORD_NORM_MEAN.exists() and WORD_NORM_STD.exists():
                self.word_norm_mean = np.load(WORD_NORM_MEAN).squeeze()
                self.word_norm_std = np.load(WORD_NORM_STD).squeeze()

            logger.info("Word model loaded (%s, %d classes).", model_type, num_classes)
        except Exception as e:
            logger.error("Failed loading Word model: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Return operational health of models and device."""
        return {
            "success": True,
            "ready": bool(self.fs_model is not None and self.word_model is not None and self.extractor is not None),
            "alphabet_model_available": self.fs_model is not None,
            "alphabet_scaler_available": self.fs_scaler is not None,
            "alphabet_classes": len(self.fs_label_map),
            "word_model_available": self.word_model is not None,
            "word_norm_available": bool(self.word_norm_mean is not None and self.word_norm_std is not None),
            "word_classes": len(self.word_label_map),
            "mediapipe_available": self.extractor is not None,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "active_sessions": self.session_manager.get_active_session_count(),
        }

    def decode_image(self, b64_str: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string into an RGB numpy image array."""
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error("Error decoding base64 image: %s", e)
            return None

    # Alias for compatibility with routes and tests
    decode_frame = decode_image

    def predict_letter(
        self, hand_landmarks: np.ndarray
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Inference for fingerspelling from single-frame 126 hand landmarks."""
        if self.fs_model is None:
            return "", 0.0, []

        try:
            has_hand = bool(float(np.sum(np.abs(hand_landmarks))) > 0)
            if not has_hand:
                return "", 0.0, []

            features = hand_landmarks.reshape(1, -1)
            if self.fs_scaler is not None:
                features = self.fs_scaler.transform(features)

            t = torch.tensor(features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.fs_model(t)
                probs = torch.softmax(logits, dim=-1)
                k = min(5, probs.size(1))
                topk_conf, topk_idx = torch.topk(probs, k=k, dim=-1)

            top1_letter = self.fs_idx_to_class.get(int(topk_idx[0, 0].item()), "")
            top1_conf = float(topk_conf[0, 0].item())

            top5 = [
                {
                    "label": self.fs_idx_to_class.get(int(topk_idx[0, i].item()), ""),
                    "confidence": float(round(topk_conf[0, i].item(), 3)),
                }
                for i in range(k)
            ]
            return str(top1_letter), top1_conf, top5
        except Exception as e:
            logger.error("Error during predict_letter: %s", e)
            return "", 0.0, []

    def predict_word(
        self, sequence: List[np.ndarray]
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Inference for temporal signs from sequence of 225 landmarks."""
        if self.word_model is None or self.word_norm_mean is None:
            return "", 0.0, []

        try:
            seq_arr = np.array(sequence, dtype=np.float32)
            seq_norm = (seq_arr - self.word_norm_mean) / self.word_norm_std
            t = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.word_model(t)
                probs = torch.softmax(logits, dim=-1)
                k = min(5, probs.size(1))
                topk_conf, topk_idx = torch.topk(probs, k=k, dim=-1)

            top1_word = self.word_idx_to_class.get(int(topk_idx[0, 0].item()), "")
            top1_conf = float(topk_conf[0, 0].item())

            top5 = [
                {
                    "label": self.word_idx_to_class.get(int(topk_idx[0, i].item()), ""),
                    "confidence": float(round(topk_conf[0, i].item(), 3)),
                }
                for i in range(k)
            ]
            return str(top1_word), top1_conf, top5
        except Exception as e:
            logger.error("Error during predict_word: %s", e)
            return "", 0.0, []

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
            sign_status = "no_hand"

            buf_len = len(session.seq_buffer)
            if not has_hand:
                sign_status = "no_hand"
                display_prediction = ""
                session.rolling_predictions.clear()
                session.rolling_confidences.clear()
            elif buf_len < 16:
                sign_status = "uncertain"
                display_prediction = "Uncertain Sign"
            else:
                buf_list = list(session.seq_buffer)
                if buf_len < TRANSFORMER_SEQ_LEN:
                    # Resample uniformly to 32 frames for immediate response
                    indices = np.linspace(0, buf_len - 1, TRANSFORMER_SEQ_LEN, dtype=int).tolist()
                    seq_sampled = [buf_list[i] for i in indices]
                else:
                    seq_sampled = buf_list

                raw_word, raw_conf, raw_top5 = self.predict_word(seq_sampled)
                raw_word_upper = raw_word.upper() if raw_word else ""
                top5 = [{"label": t["label"].upper(), "confidence": t["confidence"]} for t in raw_top5]

                # Margin between top-1 and top-2
                margin = (
                    (top5[0]["confidence"] - top5[1]["confidence"])
                    if len(top5) >= 2
                    else (top5[0]["confidence"] if top5 else 0.0)
                )

                # Check confidence against calibrated thresholds
                if raw_conf >= WORD_CONFIDENCE_HIGH and margin >= WORD_CONFIDENCE_MARGIN:
                    smoothed_word, smoothed_conf, is_consensus = session.add_prediction(raw_word_upper, raw_conf)
                    if is_consensus:
                        word = smoothed_word
                        conf = smoothed_conf
                        sign_status = "valid"
                        stable = True
                        display_prediction = word
                        if not session.is_signing:
                            accepted = session.sentence_builder.feed_word(
                                word, conf, is_signing=session.is_signing
                            )
                    else:
                        sign_status = "uncertain"
                        display_prediction = "Uncertain Sign"
                        conf = raw_conf
                else:
                    sign_status = "uncertain"
                    display_prediction = "Uncertain Sign"
                    conf = raw_conf

            return {
                "success": True,
                "session_id": session.session_id,
                "mode": "word",
                "prediction": display_prediction,
                "confidence": float(round(conf, 3)),
                "sign_status": sign_status,
                "stable": stable,
                "accepted": accepted,
                "has_hand": has_hand,
                "is_moving": is_moving,
                "top_predictions": top5,
                "current_word": session.sentence_builder.get_current_word(),
                "sentence": session.sentence_builder.get_sentence(),
            }
