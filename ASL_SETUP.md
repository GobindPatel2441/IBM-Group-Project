# ASL & Sentix Setup and Installation Guide

## 1. Prerequisites
- **Operating System**: Windows 10/11 (or Linux/macOS)
- **Python**: Version 3.10 or 3.11 (tested on Python 3.11.9)
- **Webcam**: Standard USB or built-in laptop camera
- **GPU (Optional)**: NVIDIA GPU with CUDA 12.1+ supported (CPU fallback is automatic and highly responsive)

---

## 2. Environment Installation

Ensure the virtual environment is active:

```powershell
# In PowerShell (Windows)
Set-Location "c:\Major project"
.\.venv\Scripts\Activate.ps1
```

Install the unified requirements:

```powershell
pip install -r requirements.txt
```

---

## 3. Verifying Model Artifacts

Verify that all model checkpoints exist under `backend/asl/models/`:

```powershell
Get-ChildItem -Path "backend/asl/models" -Recurse | Select-Object FullName, Length
```

Expected files:
- `backend/asl/models/alphabet/best_model.pth` (~2.8 MB)
- `backend/asl/models/alphabet/scaler.pkl` (~3.4 KB)
- `backend/asl/models/alphabet/label_map.json`
- `backend/asl/models/word/best_model.pth` (~77 MB)
- `backend/asl/models/word/norm_mean.npy` (~1 KB)
- `backend/asl/models/word/norm_std.npy` (~1 KB)
- `backend/asl/models/word/label_map.json`
- `backend/asl/models/mediapipe/hand_landmarker.task` (~7.8 MB)
- `backend/asl/models/mediapipe/pose_landmarker.task` (~5.7 MB)

---

## 4. Starting the Application

### 4.1 Launch Backend
From the project root:

```powershell
# Set Python path to project root
$env:PYTHONPATH = "c:\Major project"
python -m backend.app
```

The Flask server will start on:
```
http://127.0.0.1:5000
```

### 4.2 Launch Frontend
Open `frontend/index.html` in any modern web browser (Google Chrome, Microsoft Edge, Brave):

```powershell
# Using Python's built-in static HTTP server:
python -m http.server 8080 --directory frontend
```
Then visit `http://localhost:8080` or simply double click `frontend/index.html`.

---

## 5. Running the Test Suite

Run all backend ASL tests and regression tests:

```powershell
$env:PYTHONPATH = "c:\Major project"
python -m unittest discover -s tests -p "test_*.py"
```

