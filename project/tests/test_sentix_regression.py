"""
Sentix Core Regression Test Suite
=================================
Verifies that all original Sentix features remain completely functional:
1. Text Chat Endpoint & Routing
2. Safety Filter & Crisis Intervention
3. Emotion Detection (Text, Facial, Audio hooks)
4. Prompt Construction with Multimodal Metadata
5. Multilingual & Language Mapping Endpoints
6. System Status Health Check
"""

import unittest
from backend.app import app
from backend.safety import safety_level, safety_response
from backend.emotion_detector import detect_emotion
from backend.prompt import build_prompt, wants_affirmation_only
from backend.translator import SUPPORTED_LANGUAGES, SPEECH_LANG_MAP


class TestSentixRegression(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_languages_endpoint(self):
        res = self.client.get("/languages")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("languages", data)
        self.assertIn("speech_codes", data)
        self.assertIn("English", data["languages"])
        self.assertIn("Hindi", data["languages"])
        self.assertEqual(data["speech_codes"]["English"], "en-US")

    def test_status_endpoint(self):
        res = self.client.get("/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertIn("provider", data)

    def test_health_telemetry_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("subsystems", data)
        self.assertIn("hardware", data)
        self.assertIn("llm", data["subsystems"])
        self.assertIn("asl_recognition", data["subsystems"])
        self.assertIn("emotion_classifier", data["subsystems"])
        self.assertIn("translation", data["subsystems"])

    def test_emotion_dimensions_and_emojis(self):
        # Emojis expand to canonical affect words
        res = detect_emotion("I got the job! 🎉❤️😊")
        self.assertIn("primary", res)
        self.assertIn("dimensions", res)
        self.assertIn("valence", res["dimensions"])
        self.assertIn("arousal", res["dimensions"])
        self.assertGreaterEqual(res["dimensions"]["valence"], 0.0)

    def test_safety_filter(self):
        # High risk text triggers intervention
        high_risk = "I want to end my life and hurt myself"
        self.assertEqual(safety_level(high_risk), "high")
        resp = safety_response()
        self.assertIn("988", resp)

        # Low risk normal conversation
        normal = "Hello Sentix, how are you today?"
        self.assertEqual(safety_level(normal), "low")

    def test_emotion_detection(self):
        # Joy / positive statement
        joy_info = detect_emotion("I am so thrilled and happy today!")
        self.assertIn("primary", joy_info)
        self.assertIn("intensity", joy_info)
        self.assertIn("all", joy_info)
        self.assertIn(joy_info.get("primary"), ["joy", "love", "surprise", "sadness", "anger", "fear", "neutral"])

        # Another sample sentence
        meeting_info = detect_emotion("What time is the meeting?")
        self.assertIn("primary", meeting_info)
        self.assertIn("all", meeting_info)

    def test_affirmation_detection(self):
        self.assertTrue(wants_affirmation_only("please just reassure me"))
        self.assertTrue(wants_affirmation_only("no questions, wish me luck"))
        self.assertFalse(wants_affirmation_only("What is the capital of France?"))

    def test_prompt_builder_multimodal_and_asl(self):
        emotion_info = {"primary": "sadness", "intensity": 0.85, "severity": "high"}
        prompt_asl = build_prompt(
            user_text="I am feeling overwhelmed",
            emotion_info=emotion_info,
            history=[{"role": "user", "text": "hello"}],
            affirm_only=True,
            facial_emotion="sad",
            audio_emotion="sadness",
            source="asl",
        )

        self.assertIn("American Sign Language (ASL)", prompt_asl)
        self.assertIn("facial expression scan detected the user looks: SAD", prompt_asl)
        self.assertIn("voice analysis detected the user sounds: SADNESS", prompt_asl)
        self.assertIn("affirmation_only = True", prompt_asl)

    def test_chat_safety_intervention_response(self):
        # When high-risk text is submitted, /chat immediately intercepts and returns crisis info
        res = self.client.post(
            "/chat",
            json={
                "message": "i want to kill myself",
                "history": [],
                "language": "English",
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("emotion"), "distress")
        self.assertIn("988", data.get("response", ""))


if __name__ == "__main__":
    unittest.main()
