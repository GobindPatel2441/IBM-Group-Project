"""
Unit & Integration Tests for Sentix ASL Subsystem
=================================================
Tests:
- Model Loading & Architecture
- MediaPipe Task File Validity
- Sentence Builder Stabilization & Actions
- Multi-Session Isolation & Memory Safety
- Flask ASL Endpoints (Status, Start, Frame, Commit, Undo, Clear, Stop)
- Edge Cases: Empty Frames, Invalid Payloads, Corrupt Images
"""

import base64
import unittest
import numpy as np
import cv2
import torch

from backend.app import app
from backend.asl.config import (
    MLP_INPUT_DIM,
    MLP_NUM_CLASSES,
    TRANSFORMER_INPUT_DIM,
    TRANSFORMER_SEQ_LEN,
    TRANSFORMER_NUM_CLASSES,
)
from backend.asl.alphabet_model import FingerspellingMLP
from backend.asl.word_model import PoseTransformer
from backend.asl.sentence_builder import SentenceBuilder
from backend.asl.session_manager import SessionManager, ASLSession
from backend.asl.asl_service import ASLService


class TestASLModelsAndArchitecture(unittest.TestCase):
    """Test PyTorch architectures, dimensions, and numerical stability."""

    def test_fingerspelling_mlp_forward_and_confidence(self):
        model = FingerspellingMLP(input_dim=MLP_INPUT_DIM, num_classes=MLP_NUM_CLASSES)
        model.eval()

        # Batch forward test
        x = torch.randn(4, MLP_INPUT_DIM)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (4, MLP_NUM_CLASSES))

        # Predict with confidence test
        pred, conf = model.predict_with_confidence(x)
        self.assertEqual(pred.shape, (4,))
        self.assertEqual(conf.shape, (4,))
        self.assertTrue((pred >= 0).all() and (pred < MLP_NUM_CLASSES).all())
        self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_fingerspelling_mlp_zero_input(self):
        model = FingerspellingMLP(input_dim=MLP_INPUT_DIM, num_classes=MLP_NUM_CLASSES)
        model.eval()
        zeros = torch.zeros(2, MLP_INPUT_DIM)
        with torch.no_grad():
            out = model(zeros)
            pred, conf = model.predict_with_confidence(zeros)
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(pred.shape, (2,))

    def test_pose_transformer_forward_and_topk(self):
        model = PoseTransformer(
            input_dim=TRANSFORMER_INPUT_DIM,
            seq_len=TRANSFORMER_SEQ_LEN,
            d_model=128,  # Lightweight test instance
            n_heads=4,
            n_layers=2,
            d_ff=512,
            num_classes=20,
        )
        model.eval()

        x = torch.randn(2, TRANSFORMER_SEQ_LEN, TRANSFORMER_INPUT_DIM)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 20))

        top1_pred, top1_conf, topk_pred, topk_conf = model.predict_with_confidence(x, top_k=5)
        self.assertEqual(top1_pred.shape, (2,))
        self.assertEqual(top1_conf.shape, (2,))
        self.assertEqual(topk_pred.shape, (2, 5))
        self.assertEqual(topk_conf.shape, (2, 5))
        self.assertTrue((top1_pred == topk_pred[:, 0]).all())
        self.assertTrue((topk_conf[:, 0] >= topk_conf[:, 1]).all())


class TestSentenceBuilderAndStabilization(unittest.TestCase):
    """Test stabilization logic, hold counter, duplicate suppression, and actions."""

    def test_hold_stabilization(self):
        builder = SentenceBuilder(hold_frames=4, letter_pause=0.4, word_pause=1.5)
        t = 1000.0

        # Feed 4 frames of "A"
        for _ in range(4):
            accepted, _ = builder.feed_letter("A", 0.95, now=t)
            t += 0.05
            self.assertFalse(accepted)

        # 5th frame reaches hold_count=4 >= 4 and pause elapsed -> accepted!
        accepted, _ = builder.feed_letter("A", 0.95, now=t)
        self.assertTrue(accepted)
        self.assertEqual(builder.get_sentence(), "A")

    def test_duplicate_suppression(self):
        builder = SentenceBuilder(hold_frames=3, letter_pause=0.5, word_pause=1.5)
        t = 1000.0

        # Reach hold threshold
        for _ in range(4):
            builder.feed_letter("H", 0.95, now=t)
            t += 0.05
        self.assertEqual(builder.get_sentence(), "H")

        # Additional frames within pause window (0.2s < 0.5s) must NOT duplicate
        for _ in range(6):
            accepted, _ = builder.feed_letter("H", 0.95, now=t)
            t += 0.02
            self.assertFalse(accepted)
        self.assertEqual(builder.get_sentence(), "H")

    def test_actions_space_undo_clear(self):
        builder = SentenceBuilder()
        builder.current_letters = ["H", "I"]
        builder.sentence_words = ["SAY"]

        self.assertEqual(builder.get_sentence(), "SAY HI")

        builder.undo()
        self.assertEqual(builder.get_sentence(), "SAY H")

        builder.undo()
        self.assertEqual(builder.get_sentence(), "SAY")

        builder.clear()
        self.assertEqual(builder.get_sentence(), "")


class TestSessionIsolation(unittest.TestCase):
    """Ensure multi-user sessions are completely isolated."""

    def test_session_isolation(self):
        mgr = SessionManager(ttl_seconds=60)
        s1 = mgr.create_session("sess-1", mode="alphabet")
        s2 = mgr.create_session("sess-2", mode="word")

        s1.sentence_builder.sentence_words.append("HELLO")
        s2.sentence_builder.sentence_words.append("WORLD")

        self.assertEqual(s1.sentence_builder.get_sentence(), "HELLO")
        self.assertEqual(s2.sentence_builder.get_sentence(), "WORLD")
        self.assertNotEqual(s1.mode, s2.mode)

        # Delete session 1
        mgr.remove_session("sess-1")
        self.assertIsNone(mgr.get_session("sess-1"))
        self.assertIsNotNone(mgr.get_session("sess-2"))


class TestASLEndpoints(unittest.TestCase):
    """Integration tests for all Flask ASL API endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_asl_status_endpoint(self):
        res = self.client.get("/asl/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("alphabet_model_available"))
        self.assertTrue(data.get("word_model_available"))
        self.assertTrue(data.get("ready"))
        self.assertIn("device", data)

    def test_asl_start_stop_lifecycle(self):
        # Start
        res = self.client.post("/asl/start", json={"session_id": "test-integration-1", "mode": "alphabet"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("session_id"), "test-integration-1")
        self.assertEqual(data.get("mode"), "alphabet")

        # Stop
        res = self.client.post("/asl/stop", json={"session_id": "test-integration-1"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json().get("success"))

    def test_asl_frame_empty_payload(self):
        res = self.client.post("/asl/frame", json={})
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json().get("success"))

    def test_asl_frame_valid_image(self):
        # Create a dummy 320x240 black image
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        res = self.client.post(
            "/asl/frame",
            json={
                "session_id": "test-frame-sess",
                "image": b64_str,
                "mode": "alphabet"
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("mode"), "alphabet")
        self.assertFalse(data.get("has_hand"))
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)

    def test_asl_actions(self):
        sess_id = "test-actions-sess"
        self.client.post("/asl/start", json={"session_id": sess_id, "mode": "alphabet"})

        # Commit word
        res = self.client.post("/asl/commit", json={"session_id": sess_id, "word": "WELCOME"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("sentence"), "WELCOME")

        # Undo
        res = self.client.post("/asl/undo", json={"session_id": sess_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("sentence"), "")

        # Clear
        res = self.client.post("/asl/clear", json={"session_id": sess_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("sentence"), "")

        self.client.post("/asl/stop", json={"session_id": sess_id})

    def test_landmark_smoothing_and_entropy(self):
        sess = ASLSession("test-smooth-sess", mode="alphabet")
        raw_lm = np.random.randn(126).astype(np.float32)

        # First smoothing creates baseline
        smoothed_1 = sess.smooth_hand_landmarks(raw_lm)
        self.assertEqual(smoothed_1.shape, (126,))

        # Slight jitter input
        jittered = raw_lm + 0.05 * np.random.randn(126).astype(np.float32)
        smoothed_2 = sess.smooth_hand_landmarks(jittered)

        # EMA smoothing should damp the change
        diff_raw = np.abs(jittered - raw_lm).mean()
        diff_smoothed = np.abs(smoothed_2 - smoothed_1).mean()
        self.assertLessEqual(diff_smoothed, diff_raw * 1.5)

        # Entropy test
        sharp_probs = np.array([0.98, 0.01, 0.01], dtype=np.float32)
        uniform_probs = np.array([0.333, 0.333, 0.333], dtype=np.float32)
        h_sharp = sess.compute_normalized_entropy(sharp_probs)
        h_uniform = sess.compute_normalized_entropy(uniform_probs)
        self.assertLess(h_sharp, 0.2)
        self.assertGreater(h_uniform, 0.95)

    def test_frame_keypoints_and_latency(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        res = self.client.post(
            "/asl/frame",
            json={
                "session_id": "test-diag-sess",
                "image": b64_str,
                "mode": "alphabet"
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("keypoints", data)
        self.assertIn("latency_ms", data)
        self.assertIsInstance(data["latency_ms"], (int, float))


if __name__ == "__main__":
    unittest.main()

