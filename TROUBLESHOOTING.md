# Troubleshooting Guide: ASL Recognition Subsystem

## 1. Camera & Permissions Issues

### Symptom: "Camera access denied or unavailable"
- **Cause**: The browser blocked camera access, or another application (Zoom, Teams, Skype) has locked the webcam.
- **Solution**:
  1. Click the lock/tune icon in the browser address bar (left of `localhost:8080` or `127.0.0.1`).
  2. Toggle **Camera** permission to **Allow**.
  3. Ensure no background program is holding exclusive access to the webcam.
  4. Refresh the webpage.

---

## 2. MediaPipe Landmark Extraction

### Symptom: "Landmark extractor not initialized"
- **Cause**: Missing task model binaries in `backend/asl/models/mediapipe/`.
- **Solution**:
  Ensure both files exist:
  - `backend/asl/models/mediapipe/hand_landmarker.task`
  - `backend/asl/models/mediapipe/pose_landmarker.task`

---

## 3. GPU / CUDA vs CPU Acceleration

### Symptom: `torch.cuda.is_available()` returns `False`
- **Cause**: PyTorch was installed without CUDA support or NVIDIA drivers are unavailable.
- **Impact**: The models automatically fall back to CPU. Because the Alphabet MLP runs in ~0.44ms and the Pose Transformer in ~16ms on CPU, the system remains fast and responsive on CPU.
- **Solution (Optional GPU acceleration)**:
  To enable CUDA on Windows with NVIDIA GPUs:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

---

## 4. Frame Streaming & Network Bottlenecks

### Symptom: High latency or sluggish video response
- **Checklist**:
  1. Ensure frame capture resolution is set to $320 \times 240$ (handled automatically by `frontend/asl.js`).
  2. The single-flight mutex in `asl.js` ensures frames do not pile up in the network queue. If the client machine is under heavy load, it automatically skips frames rather than queuing them.
  3. Check the browser console (F12) for network error logs.

---

## 5. Simultaneous ASL and Emotion Tracking

### Symptom: Toggling off emotion tracker stops camera during ASL session
- **Solution**:
  Camera stream coordination in `frontend/script.js` uses reference-counting via `releaseWebcamIfUnused()`.
  The camera track is stopped **only** when both `aslActive` is `false` **and** the Emotion Tracker checkbox is unchecked.

