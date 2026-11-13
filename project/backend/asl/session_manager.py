"""
Session Management for ASL Recognition
======================================
Provides per-client session isolation, circular sequence buffer management,
motion tracking, and automatic resource reclamation on idle timeout.
"""

from collections import deque
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .config import (
    TRANSFORMER_SEQ_LEN,
    SESSION_TTL_SECONDS,
    MOTION_THRESHOLD,
    WORD_PAUSE_SECONDS,
)
from .sentence_builder import SentenceBuilder


class ASLSession:
    """
    State container for an individual active ASL recognition session.
    """

    def __init__(self, session_id: str, mode: str = "alphabet"):
        self.session_id: str = session_id
        self.mode: str = mode if mode in ("alphabet", "word") else "alphabet"
        self.created_at: float = time.time()
        self.last_active: float = time.time()

        # Sentence construction and stabilization
        self.sentence_builder = SentenceBuilder()

        # Word-mode 64-frame sequence buffer (each frame is (225,) float32)
        self.seq_buffer: deque = deque(maxlen=TRANSFORMER_SEQ_LEN)

        # Motion detection state
        self.prev_landmarks: Optional[np.ndarray] = None
        self.last_motion_time: float = time.time()
        self.is_signing: bool = False

        # Prediction stabilization and temporal anti-flicker window
        self.rolling_predictions: deque = deque(maxlen=5)
        self.rolling_confidences: deque = deque(maxlen=5)

        # Adaptive EMA smoothing state for skeletal landmarks
        self.smoothed_landmarks: Optional[np.ndarray] = None
        self.smoothed_hand_landmarks: Optional[np.ndarray] = None

        # Prediction and history log
        self.prediction_history: List[Dict[str, Any]] = []

    def touch(self):
        """Update last active timestamp."""
        self.last_active = time.time()

    def set_mode(self, mode: str):
        """Switch mode and reset temporal buffers cleanly."""
        if mode in ("alphabet", "word") and mode != self.mode:
            self.mode = mode
            self.seq_buffer.clear()
            self.sentence_builder.clear()
            self.prev_landmarks = None
            self.smoothed_landmarks = None
            self.smoothed_hand_landmarks = None
            self.is_signing = False
            self.rolling_predictions.clear()
            self.rolling_confidences.clear()
            self.touch()

    def smooth_hand_landmarks(self, landmarks: np.ndarray, base_alpha: float = 0.75) -> np.ndarray:
        """
        Adaptive Exponential Moving Average (EMA) filter on 126 hand landmarks.
        Dynamically adjusts smoothing: steadier when holding still, responsive when moving.
        """
        if landmarks is None or len(landmarks) == 0:
            return landmarks
        if np.sum(np.abs(landmarks)) == 0:
            self.smoothed_hand_landmarks = None
            return landmarks

        if self.smoothed_hand_landmarks is None or self.smoothed_hand_landmarks.shape != landmarks.shape:
            self.smoothed_hand_landmarks = landmarks.copy()
            return landmarks

        delta = float(np.abs(landmarks - self.smoothed_hand_landmarks).mean())
        # If moving fast (high delta), trust new frame (alpha -> 0.95); if steady, smooth heavily (alpha -> 0.60)
        alpha = min(0.95, max(0.60, base_alpha + delta * 6.0))
        self.smoothed_hand_landmarks = alpha * landmarks + (1.0 - alpha) * self.smoothed_hand_landmarks
        return self.smoothed_hand_landmarks.astype(np.float32)

    def smooth_full_landmarks(self, landmarks: np.ndarray, base_alpha: float = 0.78) -> np.ndarray:
        """
        Adaptive Exponential Moving Average (EMA) filter on 225 full landmarks.
        Prevents jitter in 32-frame sequences while maintaining dynamic stroke fidelity.
        """
        if landmarks is None or len(landmarks) == 0:
            return landmarks
        if np.sum(np.abs(landmarks[:126])) == 0:
            self.smoothed_landmarks = None
            return landmarks

        if self.smoothed_landmarks is None or self.smoothed_landmarks.shape != landmarks.shape:
            self.smoothed_landmarks = landmarks.copy()
            return landmarks

        delta = float(np.abs(landmarks[:126] - self.smoothed_landmarks[:126]).mean())
        alpha = min(0.96, max(0.65, base_alpha + delta * 5.0))
        self.smoothed_landmarks = alpha * landmarks + (1.0 - alpha) * self.smoothed_landmarks
        return self.smoothed_landmarks.astype(np.float32)

    @staticmethod
    def compute_normalized_entropy(probs: np.ndarray) -> float:
        """
        Compute Shannon entropy normalized to [0, 1].
        Values close to 1.0 indicate high uncertainty (uniform guessing).
        Values close to 0.0 indicate high confidence in a specific class.
        """
        if probs is None or len(probs) <= 1:
            return 1.0
        p = np.clip(probs, 1e-12, 1.0)
        p = p / np.sum(p)
        entropy = -float(np.sum(p * np.log2(p)))
        max_entropy = float(np.log2(len(p)))
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0

    def add_prediction(self, word: str, conf: float) -> Tuple[str, float, bool]:
        """
        Record prediction into rolling window and return majority consensus.
        Returns: (smoothed_word, smoothed_conf, is_consensus)
        """
        self.rolling_predictions.append(word)
        self.rolling_confidences.append(conf)

        if not self.rolling_predictions:
            return "", 0.0, False

        counts: Dict[str, int] = {}
        for w in self.rolling_predictions:
            if w and w != "Uncertain Sign":
                counts[w] = counts.get(w, 0) + 1

        if not counts:
            return "", 0.0, False

        majority_word = max(counts, key=counts.get)
        count = counts[majority_word]
        
        # Consensus: appears in at least 3 out of 5 frames (or >= 60% of window)
        is_consensus = count >= min(3, len(self.rolling_predictions))
        
        confs = [c for w, c in zip(self.rolling_predictions, self.rolling_confidences) if w == majority_word]
        avg_conf = float(np.mean(confs)) if confs else 0.0

        return majority_word, avg_conf, is_consensus

    def update_motion(self, full_lm: np.ndarray, now: Optional[float] = None) -> bool:
        """
        Check differential skeletal motion to detect signing starts and stops.
        """
        if now is None:
            now = time.time()

        motion = False
        if self.prev_landmarks is not None and len(full_lm) >= 126:
            # Check upper body & hands differential motion
            diff = float(np.abs(full_lm[:126] - self.prev_landmarks[:126]).mean())
            motion = diff > MOTION_THRESHOLD

        if motion:
            self.last_motion_time = now
            self.is_signing = True
        elif self.is_signing and (now - self.last_motion_time) > WORD_PAUSE_SECONDS:
            self.is_signing = False

        self.prev_landmarks = full_lm.copy()
        return motion


class SessionManager:
    """
    Thread-safe registry for managing active ASL recognition sessions.
    """

    def __init__(self, ttl_seconds: float = SESSION_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, ASLSession] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: Optional[str] = None, mode: str = "alphabet") -> ASLSession:
        """Create a new isolated session or replace an existing one."""
        with self._lock:
            self._cleanup_expired_locked()
            sid = session_id.strip() if (session_id and session_id.strip()) else str(uuid.uuid4())
            session = ASLSession(session_id=sid, mode=mode)
            self._sessions[sid] = session
            return session

    def get_session(self, session_id: str) -> Optional[ASLSession]:
        """Retrieve an active session and update its activity timestamp."""
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def get_or_create_session(self, session_id: Optional[str] = None, mode: str = "alphabet") -> ASLSession:
        """Get an existing session or initialize a new one if not found."""
        with self._lock:
            self._cleanup_expired_locked()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            sid = session_id.strip() if (session_id and session_id.strip()) else str(uuid.uuid4())
            session = ASLSession(session_id=sid, mode=mode)
            self._sessions[sid] = session
            return session

    def remove_session(self, session_id: str) -> bool:
        """Explicitly destroy a session upon stop."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def active_sessions_count(self) -> int:
        """Return total active session count."""
        with self._lock:
            self._cleanup_expired_locked()
            return len(self._sessions)

    def get_active_session_count(self) -> int:
        """Alias for active_sessions_count."""
        return self.active_sessions_count()

    @property
    def sessions(self) -> Dict[str, ASLSession]:
        """Direct access to sessions dict."""
        return self._sessions

    def _cleanup_expired_locked(self):
        """Purge sessions inactive longer than TTL."""
        now = time.time()
        expired = [
            sid for sid, sess in self._sessions.items()
            if (now - sess.last_active) > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

