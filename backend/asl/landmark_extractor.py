"""
MediaPipe Landmark Extractor
============================
Thread-safe landmark extraction for:
1. Hand landmarks (126 dims: right 63 + left 63)
2. Pose + Hand landmarks (225 dims: right 63 + left 63 + pose 99)
"""

from typing import Optional, Tuple
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import (
    HAND_LANDMARKER_TASK,
    POSE_LANDMARKER_TASK,
    MLP_INPUT_DIM,
    TRANSFORMER_INPUT_DIM,
)

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """
    Encapsulates MediaPipe Tasks detectors and frame extraction methods.
    """

    def __init__(self, hand_confidence: float = 0.5, pose_confidence: float = 0.5):
        self.hand_detector: Optional[vision.HandLandmarker] = None
        self.pose_detector: Optional[vision.PoseLandmarker] = None
        self._init_detectors(hand_confidence, pose_confidence)

    def _init_detectors(self, hand_conf: float, pose_conf: float):
        if not HAND_LANDMARKER_TASK.exists():
            raise FileNotFoundError(f"Missing hand landmark task file at {HAND_LANDMARKER_TASK}")
        if not POSE_LANDMARKER_TASK.exists():
            raise FileNotFoundError(f"Missing pose landmark task file at {POSE_LANDMARKER_TASK}")

        # Initialize Hand Detector
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_TASK)),
            num_hands=2,
            min_hand_detection_confidence=hand_conf,
            min_hand_presence_confidence=hand_conf,
            min_tracking_confidence=hand_conf,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # Initialize Pose Detector
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_LANDMARKER_TASK)),
            min_pose_detection_confidence=pose_conf,
            min_pose_presence_confidence=pose_conf,
            min_tracking_confidence=pose_conf,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        logger.info("MediaPipe detectors loaded successfully.")

    def extract_hand_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract 126 hand landmarks from an RGB frame.
        Order: Right Hand (63) + Left Hand (63).
        """
        if self.hand_detector is None:
            return np.zeros(MLP_INPUT_DIM, dtype=np.float32)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.hand_detector.detect(mp_image)
            left_hand = np.zeros(63, dtype=np.float32)
            right_hand = np.zeros(63, dtype=np.float32)

            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
                    ).flatten()

                    if result.handedness and idx < len(result.handedness):
                        label = result.handedness[idx][0].category_name
                        if label == "Left":
                            left_hand = landmarks
                        else:
                            right_hand = landmarks
                    else:
                        if idx == 0:
                            right_hand = landmarks
                        else:
                            left_hand = landmarks

            return np.concatenate([right_hand, left_hand])
        except Exception as e:
            logger.error("Error extracting hand landmarks: %s", e)
            return np.zeros(MLP_INPUT_DIM, dtype=np.float32)

    def extract_full_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract 225 landmarks from an RGB frame.
        Order: Right Hand (63) + Left Hand (63) + Pose (99).
        """
        if self.hand_detector is None or self.pose_detector is None:
            return np.zeros(TRANSFORMER_INPUT_DIM, dtype=np.float32)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Hand detection
            hand_result = self.hand_detector.detect(mp_image)
            left_hand = np.zeros(63, dtype=np.float32)
            right_hand = np.zeros(63, dtype=np.float32)

            if hand_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    lms = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
                    ).flatten()
                    if hand_result.handedness and idx < len(hand_result.handedness):
                        label = hand_result.handedness[idx][0].category_name
                        if label == "Left":
                            left_hand = lms
                        else:
                            right_hand = lms
                    else:
                        if idx == 0:
                            right_hand = lms
                        else:
                            left_hand = lms

            # Pose detection
            pose_result = self.pose_detector.detect(mp_image)
            pose = np.zeros(99, dtype=np.float32)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]],
                    dtype=np.float32,
                ).flatten()

            return np.concatenate([right_hand, left_hand, pose])
        except Exception as e:
            logger.error("Error extracting full landmarks: %s", e)
            return np.zeros(TRANSFORMER_INPUT_DIM, dtype=np.float32)

    def close(self):
        """Release MediaPipe resources."""
        if self.hand_detector:
            try:
                self.hand_detector.close()
            except Exception:
                pass
            self.hand_detector = None
        if self.pose_detector:
            try:
                self.pose_detector.close()
            except Exception:
                pass
            self.pose_detector = None

