# Sentix: Real-time Emotion-Aware Multilingual Empathy Companion

Sentix is a cutting-edge AI assistant designed to bridge the gap between human emotions and artificial intelligence. By combining **Real-time Facial Emotion Detection**, **Real-time Audio Emotion Detection**, **Multilingual Translation**, and **Empathetic LLM Response Generation**, Sentix provides a unique, emotionally resonant interaction experience in multiple Indian languages.

![Sentix Banner](https://img.shields.io/badge/AI-Empathy_Companion-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey?style=for-the-badge&logo=flask)
![LLM](https://img.shields.io/badge/LLM-ChatGPT%20%7C%20Ollama-orange?style=for-the-badge)

---

## 🚀 Key Features

- **🎭 Facial Emotion Recognition**: Uses the webcam to detect user emotions (Happy, Sad, Angry, etc.) in real-time using `DeepFace` (powered by MTCNN).
- **🎙️ Audio Emotion Recognition**: Analyzes voice inputs using the `SenseVoice` model to extract emotional undertones from spoken language.
- **🗣️ Multilingual Support**: Bidirectional translation between English and 11 Indian languages (Hindi, Bengali, Tamil, Telugu, etc.) using the `NLLB-200` model.
- **🧠 Empathetic Intelligence**: Powered by **ChatGPT (OpenAI)** or **Llama 3 (Ollama)**, the AI conditions its responses based on both text sentiment and the user's audio/facial expressions.
- **⚡ Streaming Responses**: Low-latency, sentence-level streaming for a smooth conversational experience.
- **🛡️ Safety First**: Integrated safety layer to handle sensitive topics and provide appropriate, supportive responses.
- **📊 Model Evaluator**: Includes a built-in evaluation script to test the accuracy of the audio and vision models against your own datasets.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (ES6+).
- **Backend**: Flask, Python.
- **AI/ML Models**:
  - **Vision**: [DeepFace](https://github.com/serengil/deepface) (MTCNN Backend).
  - **Audio**: [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) (via FunASR).
  - **Language Model**: ChatGPT API or [Llama 3](https://ollama.com/library/llama3) via Ollama.
  - **Translation**: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M).

---

## 📦 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Ollama (Optional)**: If you want to run Llama 3 locally instead of ChatGPT, [download and install Ollama](https://ollama.com/).

### 2. Clone the Repository
```bash
git clone https://github.com/GobindPatel2441/IBM-Group-Project.git
cd IBM-Group-Project
```

### 3. Setup Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
*(This will install everything including Flask, DeepFace, MTCNN, SenseVoice dependencies, and OpenAI SDK).*

### 5. Configure your API Keys and LLM Provider
1. Open the `.env` file located in the root of the project.
2. Decide whether you want to use **OpenAI (ChatGPT)** or **Ollama (Local Llama 3)**.
3. Edit the file accordingly:

**To use ChatGPT (Recommended for best empathy):**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

**To use Ollama locally:**
```env
LLM_PROVIDER=ollama
# Make sure to run `ollama pull llama3` in your terminal first!
```

---

## 🏃 Running the Project

### 1. Start the Backend Server
```bash
python -m backend.app
```
The server will start at `http://127.0.0.1:5000`. Keep this terminal window open!

### 2. Launch the Frontend
Open the `frontend/index.html` file in your preferred web browser. (You can do this by double-clicking it in your File Explorer).

### 3. Evaluating Model Accuracy
If you want to test how accurate the Vision and Audio emotion models are:
```bash
python evaluate_models.py
```
This will automatically generate an `eval_dataset` folder. You can place test images and audio clips inside their respective emotion folders and run the script again to get an accuracy report!

---

## 📁 Project Structure

```text
├── backend/
│   ├── app.py              # Main Flask server API
│   ├── config.py           # Centralized configuration & .env loader
│   ├── audio_emotion.py    # SenseVoice audio emotion detector
│   ├── emotion_detector.py # Text-based sentiment analysis
│   ├── local_model.py      # LLM Router (Ollama/OpenAI)
│   ├── openai_model.py     # ChatGPT integration
│   ├── translator.py       # NLLB-200 translation wrapper
│   └── prompt.py           # Empathetic prompt engineering
├── frontend/
│   ├── index.html          # Web UI
│   ├── script.js           # Frontend logic & API handling
│   └── style.css           # Styling
├── evaluate_models.py      # Script to test audio/vision accuracy
├── requirements.txt        # Full list of dependencies
└── .env                    # Environment variables (API Keys)
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the emotion detection accuracy, add more languages, or enhance the UI, feel free to fork the repo and submit a PR.
