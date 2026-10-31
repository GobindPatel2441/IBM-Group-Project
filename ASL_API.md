# ASL REST API Reference

The ASL module exposes REST endpoints under the `/asl` prefix.

---

## 1. `GET /asl/status`
Returns subsystem readiness, compute device, and model availability.

### Response `200 OK`:
```json
{
  "success": true,
  "ready": true,
  "device": "cuda:0",
  "cuda_available": true,
  "alphabet_model_available": true,
  "alphabet_scaler_available": true,
  "alphabet_classes": 29,
  "word_model_available": true,
  "word_norm_available": true,
  "word_classes": 200,
  "mediapipe_available": true,
  "active_sessions": 1
}
```

---

## 2. `POST /asl/start`
Initializes a new recognition session.

### Request Body:
```json
{
  "session_id": "user-session-abc",
  "mode": "alphabet"
}
```

### Response `200 OK`:
```json
{
  "success": true,
  "session_id": "user-session-abc",
  "mode": "alphabet",
  "sentence": "",
  "message": "ASL session initialized successfully"
}
```

---

## 3. `POST /asl/frame`
Receives an individual webcam frame and returns real-time predictions and progressive sentence updates.

### Request Body:
```json
{
  "session_id": "user-session-abc",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "mode": "alphabet"
}
```

### Response `200 OK` (Alphabet Mode):
```json
{
  "success": true,
  "session_id": "user-session-abc",
  "mode": "alphabet",
  "prediction": "H",
  "confidence": 0.96,
  "stable": true,
  "accepted": true,
  "hold_progress": 1.0,
  "has_hand": true,
  "top_predictions": [
    { "label": "H", "confidence": 0.96 },
    { "label": "A", "confidence": 0.02 }
  ],
  "current_word": "HI",
  "sentence": "HI"
}
```

### Response `200 OK` (Word Mode):
```json
{
  "success": true,
  "session_id": "user-session-abc",
  "mode": "word",
  "prediction": "HELLO",
  "confidence": 0.88,
  "stable": true,
  "accepted": false,
  "has_hand": true,
  "is_signing": true,
  "top_predictions": [
    { "label": "HELLO", "confidence": 0.88 },
    { "label": "MORNING", "confidence": 0.05 },
    { "label": "WELCOME", "confidence": 0.03 }
  ],
  "buffer_len": 64,
  "buffer_ready": true,
  "sentence": "HELLO"
}
```

---

## 4. `POST /asl/commit`
Explicitly commits the current fingerspelled word or an override string into the sentence buffer.

### Request Body:
```json
{
  "session_id": "user-session-abc",
  "word": "THANKYOU"
}
```

### Response `200 OK`:
```json
{
  "success": true,
  "sentence": "HELLO THANKYOU",
  "current_word": ""
}
```

---

## 5. `POST /asl/undo`
Removes the last accepted token (letter or word).

### Request Body:
```json
{
  "session_id": "user-session-abc"
}
```

### Response `200 OK`:
```json
{
  "success": true,
  "sentence": "HELLO",
  "current_word": ""
}
```

---

## 6. `POST /asl/clear`
Clears the active sentence buffer and resets temporal sequence queues.

### Request Body:
```json
{
  "session_id": "user-session-abc"
}
```

### Response `200 OK`:
```json
{
  "success": true,
  "sentence": "",
  "current_word": ""
}
```

---

## 7. `POST /asl/stop`
Releases session resources.

### Request Body:
```json
{
  "session_id": "user-session-abc"
}
```

### Response `200 OK`:
```json
{
  "success": true,
  "message": "ASL session stopped and released successfully"
}
```

