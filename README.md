# Sentix: Real-time Emotion-Aware Multilingual Empathy Companion

Sentix is a cutting-edge AI assistant designed to bridge the gap between human emotions and artificial intelligence. By combining **Real-time Facial Emotion Detection**, **Multilingual Translation**, and **Empathetic LLM Response Generation**, Sentix provides a unique, emotionally resonant interaction experience in multiple Indian languages.

![Sentix Banner](https://img.shields.io/badge/AI-Empathy_Companion-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey?style=for-the-badge&logo=flask)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-orange?style=for-the-badge)

---

## 🚀 Key Features

- **🎭 Facial Emotion Recognition**: Uses the webcam to detect user emotions (Happy, Sad, Angry, etc.) in real-time using `DeepFace`.
- **🗣️ Multilingual Support**: Bidirectional translation between English and 11 Indian languages (Hindi, Bengali, Tamil, Telugu, etc.) using the `NLLB-200` model.
- **🧠 Empathetic Intelligence**: Powered by `Llama 3` (via Ollama), the AI conditions its responses based on both the text sentiment and the user's facial expression.
- **⚡ Streaming Responses**: Low-latency, sentence-level streaming for a smooth conversational experience.
- **🛡️ Safety First**: Integrated safety layer to handle sensitive topics and provide appropriate, supportive responses.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (ES6+).
- **Backend**: Flask, Python.
- **AI/ML Models**:
  - **Vision**: [DeepFace](https://github.com/serengil/deepface) (Facial Emotion Detection).
  - **Language Model**: [Llama 3](https://ollama.com/library/llama3) via **Ollama**.
  - **Translation**: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (No Language Left Behind).
- **Tools**: OpenCV, PyTorch, Transformers.

---

## 📦 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: [Download and install Ollama](https://ollama.com/).
- **CUDA (Optional)**: Highly recommended for GPU acceleration.

### 2. Clone the Repository
```bash
git clone https://github.com/GobindPatel2441/Major-Project.git
cd Major-Project
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
*(Note: Ensure you have `flask`, `flask-cors`, `deepface`, `transformers`, `torch`, `requests`, and `opencv-python` installed.)*

### 5. Pull the LLM Model
Ensure Ollama is running and pull the Llama 3 model:
```bash
ollama pull llama3
```

---

## 🏃 Running the Project

### 1. Start the Backend
```bash
python -m backend.app
```
The server will start at `http://127.0.0.1:5000`.

### 2. Launch the Frontend
Simply open `frontend/index.html` in your preferred web browser.

### 3. Webcam Emotion Tracker (Standalone)
If you want to test just the facial emotion detection:
```bash
python webcam_emotion.py
```

---

## 📁 Project Structure

```text
├── backend/
│   ├── app.py              # Main Flask server
│   ├── emotion_detector.py # Text-based sentiment analysis
│   ├── local_model.py      # Ollama/Llama3 integration
│   ├── translator.py       # NLLB-200 translation wrapper
│   └── prompt.py           # Empathetic prompt engineering
├── frontend/
│   ├── index.html          # Web UI
│   ├── script.js           # Frontend logic & API handling
│   └── style.css           # Styling
├── Facial Emotion Detector/ # DeepFace local assets
├── Translator Model/       # Local NLLB model storage
└── webcam_emotion.py       # Standalone emotion tracker script
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the emotion detection accuracy, add more languages, or enhance the UI, feel free to fork the repo and submit a PR.

---

