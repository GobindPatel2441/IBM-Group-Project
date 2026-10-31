# ASL & Sentix Architecture & Integration Guide

## 1. Executive Overview
This document details the architectural integration of the standalone Phase 1 American Sign Language (ASL) recognition system into the Sentix multimodal empathy companion.

With this integration, Sentix is an end-to-end multimodal companion accepting:
1. **Text**: Direct keyboard input with real-time empathetic tone detection.
2. **Voice**: Speech transcription and audio emotion analysis via SenseVoice.
3. **Facial Emotion**: Computer vision facial expression tracking via DeepFace (MTCNN backend).
4. **Sign Language**: Webcam-based ASL Alphabet (A–Z, fingerspelling) and ASL Word-Level gestures (200 vocabulary words) via MediaPipe and PyTorch models.
5. **Multilingual Translation**: Seamless bidirectional translation across 11 Indic languages and English via NLLB-200.

---

## 2. Integrated Module Architecture

The ASL module is located in `backend/asl/`:

```
backend/
├── asl/
│   ├── __init__.py               # Package exports & Blueprint
│   ├── config.py                 # Paths, hardware selection & thresholds
│   ├── landmark_extractor.py     # MediaPipe Tasks Hand & Pose extractors
│   ├── alphabet_model.py         # 4-layer Fingerspelling MLP (126 -> 29)
│   ├── word_model.py             # 6-layer Pose Transformer (64x225 -> 200)
│   ├── sentence_builder.py       # Hold counter, duplicate suppression, buffer
│   ├── session_manager.py        # Per-user session isolation & TTL cleanup
│   ├── asl_service.py            # Singleton coordinator (model & frame processing)
│   ├── routes.py                 # Flask Blueprint (/asl/* endpoints)
│   └── models/
│       ├── alphabet/             # best_model.pth, scaler.pkl, label_map.json
│       ├── word/                 # best_model.pth, norm_mean.npy, norm_std.npy, label_map.json
│       └── mediapipe/            # hand_landmarker.task, pose_landmarker.task
```

---

## 3. Machine Learning Models & Normalization

### 3.1 Alphabet Model (Fingerspelling MLP)
- **Input Dimension**: `126` float32 features (Right Hand $21 \times 3$ + Left Hand $21 \times 3$).
- **Output Classes**: `29` classes (`a`–`z`, `del`, `nothing`, `space`).
- **Architecture**:
  - `Linear(126, 512)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)`
  - `Linear(512, 256)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)`
  - `Linear(256, 128)` → `BatchNorm1d` → `ReLU` → `Dropout(0.2)`
  - `Linear(128, 29)`
- **Normalization**: Scikit-Learn `StandardScaler` loaded from `scaler.pkl`.
- **Stabilization**:
  - Requires 8 consecutive frames of the same character (`hold_frames = 8`).
  - Minimum pause of 0.8 seconds between distinct characters to prevent duplicate insertions.

### 3.2 Word Recognition Model (Pose Transformer)
- **Input Dimension**: `225` float32 coordinates per frame:
  - Right hand: 63 coordinates ($21 \times 3$)
  - Left hand: 63 coordinates ($21 \times 3$)
  - Pose: 99 coordinates ($33 \times 3$)
- **Sequence Length**: `64` temporal frames.
- **Output Classes**: `200` vocabulary classes from ASL Citizen.
- **Architecture**:
  - Linear projection ($225 \rightarrow 512$).
  - Learnable `[CLS]` token prepended to sequence.
  - Sinusoidal positional encoding ($64 + 1$ positions).
  - 6-Layer TransformerEncoder (Pre-LayerNorm, $d_{model}=512$, $n_{head}=8$, $d_{ff}=2048$, GELU, dropout 0.1).
  - Classification head operating on `[CLS]` token representation.
- **Normalization**: Feature-wise z-score normalization:
  $$\hat{X} = \frac{X - \mu}{\sigma}$$
  where $\mu$ and $\sigma$ are loaded from `norm_mean.npy` and `norm_std.npy`.
- **Sign Boundary Detection**:
  - Differential skeletal motion across frames:
    $$\Delta = \frac{1}{126} \sum_{i=1}^{126} |X_t[i] - X_{t-1}[i]|$$
  - When $\Delta > 0.005$, user is actively signing.
  - When motion drops and pause exceeds 2.0s, high-confidence prediction is accepted.

---

## 4. Frontend & Shared Webcam Architecture

- **Single Camera Stream**: `frontend/script.js` coordinates camera access. The webcam is shared between the DeepFace emotion tracker and the ASL sign recognizer without opening duplicate streams.
- **Single-Flight Request Pump**: `frontend/asl.js` streams canvas snapshots to `/asl/frame` at ~9 frames per second (110 ms intervals). Overlapping requests are discarded using a mutex flag `isProcessingFrame`.
- **Handoff to Chatbot**: Clicking **"Send to Sentix"** takes the accumulated sentence, populates the chat input `#input`, attaches `{ source: "asl" }`, and triggers the existing streaming `/chat` pipeline.

