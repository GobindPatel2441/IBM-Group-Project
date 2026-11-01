# SENTIX: Multimodal Emotion-Aware AI Chatbot for Adaptive Human Interaction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision_&_Pose-007ACC?style=for-the-badge)](https://developers.google.com/mediapipe)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3-white?style=for-the-badge&logo=ollama&logoColor=black)](https://ollama.com/)

**SENTIX** is an advanced, emotionally intelligent communication companion engineered to bridge affective and linguistic barriers in human-computer interaction. By synthesizing **Real-Time Facial Expression Recognition**, **Acoustic Voice Emotion Detection**, **Fine-Tuned BERT Text Emotion Classification**, **Bidirectional Multilingual Translation (12 Indic Languages)**, and a **Dual-Model American Sign Language (ASL) Recognition Subsystem**, SENTIX delivers an empathetic, multimodal conversational experience.

---

## 🌟 Key Capabilities

### 1. 🤟 Dual-Model American Sign Language (ASL) Recognition
- **Model A (Alphabet Fingerspelling Recognition):**
  - **Architecture:** 4-Layer Multilayer Perceptron (MLP) with Batch Normalization and Dropout.
  - **Feature Input:** 126 skeletal coordinates (21 3D hand landmarks $\times$ 2 hands) normalized via `StandardScaler`.
  - **Vocabulary:** 29 classes (A–Z, Space, Delete).
  - **Accuracy:** **99.30% validation accuracy**, **100.0% unseen test accuracy**.
- **Model B (Dynamic Word-Level ASL Recognition):**
  - **Architecture:** Spatio-Temporal Attention Network (Spatial Linear Projection + Sinusoidal Positional Encoding + 2-Layer Bidirectional LSTM + 4-Head Multi-Head Self-Attention).
  - **Feature Input:** 225 skeletal landmarks (126 hands + 99 upper-body pose coordinates) aggregated across a 32-frame sliding window with global z-score normalization.
  - **Vocabulary:** 54 conversational word classes (e.g., *hello, thank you, please, friend, help, eat, water, yes, no*).
  - **Performance:** **79.27% unseen test top-1 accuracy**, **91.46% unseen test top-5 accuracy**, **2.03 ms CPU inference latency** (~492 FPS throughput).
- **Real-Time Webcam Stabilization Layer:**
  - **5-Frame Rolling Window Consensus:** Suppresses high-frequency flickering by requiring majority agreement ($\ge 3$ of 5 frames).
  - **Confidence & Top-2 Margin Gating:** Dynamic thresholding ($\tau_{\text{min}} = 0.28, \Delta_{\text{margin}} \ge 0.05$).
  - **Triple-State UI Feedback:** Real-time badge indicators:
    - ✋ `NO_HAND`: When hands are absent.
    - ⚠️ `UNCERTAIN_SIGN`: When transitioning between signs or performing ambiguous gestures (prevents false predictions).
    - ✅ `VALID_SIGN`: Confident recognized sign committed on gesture completion.
  - **Interactive ASL Studio:** Full-screen modal with landmark visualization, live sentence builder buffer, and one-click "Send to Sentix" chat handoff.

### 2. 🎭 Multimodal Emotion Intelligence
- **Text Emotion Classification:** Fine-tuned `BERT` transformer (`bert-base-uncased`) classifying text into *anger, fear, joy, sadness,* and *neutral*, computing continuous intensity $[0.0, 1.0]$ and discrete severity (*low, medium, high*). Classical SVR + TF-IDF with negation handling and polarity gating serves as an offline fallback.
- **Facial Emotion Recognition:** Ingests webcam snapshots and runs `DeepFace` with an `MTCNN` (Multi-Task Cascaded CNN) backend for face detection, extracting dominant facial affect (*happy, sad, angry, fear, surprise, neutral*).
- **Acoustic Voice Emotion Recognition:** Analyzes microphone audio inputs via `SenseVoiceSmall` (FunASR foundation model) to extract prosodic vocal sentiment tokens (`<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`).

### 3. 🌐 Multilingual Translation (Meta NLLB-200)
- Bidirectional translation between English and **11 Indian languages**:
  - **Hindi** (`hin_Deva`), **Bengali** (`ben_Beng`), **Tamil** (`tam_Taml`), **Telugu** (`tel_Telu`), **Marathi** (`mar_Deva`), **Gujarati** (`guj_Gujr`), **Kannada** (`kan_Knda`), **Malayalam** (`mal_Mlym`), **Punjabi** (`pan_Guru`), **Odia** (`ory_Orya`), **Assamese** (`asm_Beng`).
- Translates native Indic input into an English semantic pivot for affective reasoning, then translates the LLM's response back to the user's selected language in real time using sentence-level boundary buffering.

### 4. 🧠 Adaptive Empathetic LLM Reasoning
- Conditioned on fused multimodal affect: text intensity, facial expression, vocal tone, and dialogue history.
- **Safety Crisis Bypass:** Deterministic filter in `backend/safety.py` scanning for acute self-harm/crisis intent. Instantly intercepts and returns emergency crisis intervention resources (988 Lifeline) in the user's language, **completely bypassing LLM inference** to prevent hallucination.
- **Configurable LLM Backend:** Runs locally via **Meta Llama 3** on Ollama or via cloud **OpenAI GPT-3.5-Turbo**.

---

## 🏛️ System Architecture

SENTIX is organized into an **8-Layer Modular Architecture**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 1: USER INTERACTION & FRONTEND CLIENT               │
│  (Chat UI, Web Audio Mic, Canvas Webcam, 12-Lang Selector, ASL Studio) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST / SSE Stream
┌───────────────────────────────────▼────────────────────────────────────┐
│               LAYER 2: APPLICATION BACKEND & API GATEWAY                │
│ (Flask 3.x, CORS, /chat, /asl/* Routes, Session Manager, NDJSON Stream) │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
┌───────────────────▼───────────────┐  ┌─────────────▼───────────────────┐
│ LAYER 3: MULTIMODAL PREPROCESSING │  │  LAYER 4: EMOTION INTELLIGENCE  │
│ - Text Script & Ratio Validation  │  │ - Fine-Tuned BERT Emotion Model │
│ - FunASR Audio VAD Segmentation   │  │ - DeepFace MTCNN Facial Affect  │
│ - MTCNN Face Bounding Box & Align │  │ - SenseVoiceSmall Audio Emotion │
│ - MediaPipe Hands (126D) & Pose   │  │ - ASL Model A (29-Class MLP)    │
│ - 32-Frame Sequence Buffer (225D) │  │ - ASL Model B (54-Class Attn)   │
└───────────────────┬───────────────┘  └─────────────┬───────────────────┘
                    │                                │
┌───────────────────▼────────────────────────────────▼───────────────────┐
│            LAYER 5: LANGUAGE PROCESSING (Meta NLLB-200)                │
│    (Indic -> English Pivot | LRU Cache | English -> Indic Stream)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│      LAYER 6: CONTEXT, SAFETY & DYNAMIC PROMPT CONSTRUCTION            │
│   (Crisis Safety Intercept | Dialogue Window | Multimodal Affect Fuse) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│      LAYER 7: LLM INFERENCE & RESPONSE GENERATION LAYER                │
│    (Ollama Meta Llama 3 / OpenAI GPT-3.5-Turbo | Token Generator)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│      LAYER 8: PERSISTENT STORAGE, SESSIONS & INFRASTRUCTURE            │
│  (Client LocalStorage, RAM Sessions, PyTorch .pth Models, Config .env) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```text
├── backend/
│   ├── app.py                     # Main Flask server & multimodal API gateway
│   ├── config.py                  # Centralized configuration & .env loader
│   ├── audio_emotion.py           # SenseVoiceSmall audio emotion analysis
│   ├── emotion_detector.py        # Emotion classification router (BERT / SVR)
│   ├── local_model.py             # LLM router & Ollama Llama 3 streaming gateway
│   ├── openai_model.py            # OpenAI GPT-3.5-Turbo gateway
│   ├── prompt.py                  # Empathetic prompt engineering & affect fusion
│   ├── safety.py                  # Deterministic self-harm crisis filter & 988 bypass
│   ├── translator.py              # Meta NLLB-200 bidirectional translation wrapper
│   ├── utils.py                   # Negation handling & polarity gating helpers
│   └── asl/                       # ASL Recognition Subsystem
│       ├── alphabet_model.py      # Model A: 4-Layer MLP for static fingerspelling
│       ├── word_model.py          # Model B: SpatioTemporalAttentionModel for dynamic signs
│       ├── landmark_extractor.py  # MediaPipe Hands & Pose skeletal feature extractor
│       ├── session_manager.py     # 5-frame rolling consensus & buffer management
│       ├── sentence_builder.py    # ASL word concatenation & sentence state
│       ├── asl_service.py         # Real-time inference coordinator & stabilization
│       ├── routes.py              # Flask Blueprint for /asl/* endpoints
│       └── models/                # Serialized model checkpoints
│           ├── alphabet/          # Model A weights (best_model.pth, scaler.pkl)
│           ├── word/              # Model B weights (best_model.pth, norm_mean/std)
│           └── mediapipe/         # MediaPipe landmarker task binaries (.task)
├── frontend/
│   ├── index.html                 # Main conversational UI
│   ├── script.js                  # Chat orchestration, audio/webcam capture
│   ├── style.css                  # Liquid-glass styles & status badges
│   ├── asl.html                   # Interactive ASL Studio workspace
│   ├── asl.js                     # ASL HUD, skeleton canvas & chat handoff
│   └── liquid-glass/              # Custom liquid glass UI component library
├── BERT_Model/
│   ├── bert_detector.py           # PyTorch fine-tuned BERT emotion classifier
│   └── fine_tuned_bert/           # Trained BERT transformer weights & tokenizer
├── Model/                         # SVR & TF-IDF regression fallback models
├── tests/                         # Full automated test suite (19 unit tests)
├── requirements.txt               # Complete Python runtime dependencies
└── .env                           # Centralized environment configuration
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10 or 3.11** (64-bit recommended)
- **Webcam & Microphone** (for facial, speech, and ASL features)
- **Ollama** (optional, for local Llama 3 execution): [https://ollama.com/](https://ollama.com/)

### 2. Clone the Repository
```bash
git clone https://github.com/GobindPatel2441/IBM-Group-Project.git
cd IBM-Group-Project
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create or verify your `.env` file in the project root:

```env
# Choose LLM Provider: 'ollama' or 'openai'
LLM_PROVIDER=ollama

# If using Ollama (Local Llama 3):
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3:latest

# If using OpenAI:
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Emotion Model Selector: 'bert' (default) or 'svr'
EMOTION_MODEL_TYPE=bert
```

If using Ollama, pull the Llama 3 model in a separate terminal:
```bash
ollama pull llama3
```

---

## 🚀 Running the System

### 1. Launch the Flask Backend
```bash
python -m backend.app
```
The server will initialize models and start listening at:  
👉 **`http://127.0.0.1:5000`**

### 2. Open the Frontend
Simply open **`http://127.0.0.1:5000`** in Google Chrome or any modern browser.

### 3. Using the ASL Recognition Studio
1. Click the **ASL Studio** toggle button in the chat input dock.
2. Select **Alphabet Mode** for static fingerspelling (A–Z) or **Word Mode** for dynamic signs (54 conversational words).
3. Position yourself in front of the camera. The real-time badge will reflect `✋ Hand Detected`, `⚠️ Uncertain Sign`, or `✋ No Hand Detected`.
4. As words are confirmed by the 5-frame consensus window, they populate the sentence box.
5. Click **Send to Sentix** to transfer the assembled sentence into the main chat for an empathetic AI response.

---

## 🧪 Verification & Automated Testing

SENTIX includes an automated test suite covering all modules:
```bash
python -m unittest discover tests
```
*Output: All 19 unit tests passing across safety checks, prompt builders, translation, and ASL services.*

To verify the live backend endpoints:
```bash
# General health check
curl http://127.0.0.1:5000/status

# ASL subsystem status
curl http://127.0.0.1:5000/asl/status
```

---

## 📡 API Reference

| Endpoint | Method | Description | Payload / Response |
| :--- | :---: | :--- | :--- |
| `/chat` | `POST` | Primary multimodal streaming chat endpoint | `{ "message": str, "language": str, "image": base64, "audio": base64, "source": str }` $\rightarrow$ Streaming NDJSON |
| `/languages` | `GET` | List of supported languages and BCP-47 speech codes | Returns JSON array of 12 supported languages |
| `/status` | `GET` | Health check for active LLM backend | `{ "status": "online", "provider": "ollama" }` |
| `/asl/status` | `GET` | ASL subsystem readiness, device, active sessions | `{ "ready": true, "word_classes": 54, "alphabet_classes": 29 }` |
| `/asl/start` | `POST` | Initialize an ASL recognition session | `{ "session_id": str, "mode": "alphabet"\|"word" }` |
| `/asl/frame` | `POST` | Process a live webcam video frame | `{ "session_id": str, "image": base64, "mode": str }` |
| `/asl/commit` | `POST` | Commit current stable prediction to sentence | `{ "session_id": str }` |
| `/asl/undo` | `POST` | Delete last committed sign/letter | `{ "session_id": str }` |
| `/asl/clear` | `POST` | Clear sentence buffer | `{ "session_id": str }` |
| `/asl/stop` | `POST` | End session and free memory | `{ "session_id": str }` |

---

## 🎓 Academic Attribution & Project Scope

- **Project:** SENTIX — Multimodal Emotion-Aware AI Chatbot for Adaptive Human Interaction
- **Academic Degree:** Bachelor of Technology (B.Tech) in Artificial Intelligence & Data Science
- **Institution:** Sri Sri University, School of Artificial Intelligence and Data Science
- **Faculty / Evaluation:** Major Project Final Year Submission
