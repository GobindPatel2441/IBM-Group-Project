/**
 * Sentix ASL (American Sign Language) Recognition Controller
 * ==========================================================
 * Manages webcam acquisition, single-flight throttled frame streaming (8-10 fps),
 * dual recognition modes (Alphabet & Word), temporal stabilization visualization,
 * progressive sentence builder, and seamless handoff into the Sentix /chat pipeline.
 */

(function () {
    "use strict";

    // ── Configuration & State ────────────────────────────────────────────────
    const FRAME_RATE_MS = 110; // ~9 fps for optimal latency vs network load
    let sessionId = "sentix-asl-" + Math.random().toString(36).substring(2, 10);
    let currentMode = "alphabet"; // "alphabet" or "word"
    let isRunning = false;
    let isProcessingFrame = false;
    let frameLoopTimer = null;
    let cachedApiBase = null;

    // DOM Elements Cache
    let modalEl, backdropEl, previewVideo, previewCanvas;
    let modeAlphaBtn, modeWordBtn, startBtn, stopBtn;
    let livePredEl, confValEl, confBarEl, holdBarEl, holdContainerEl;
    let statusBadgeEl, handBadgeEl, top5ListEl, sentenceBoxEl;
    let spaceBtn, undoBtn, clearBtn, sendToSentixBtn;

    // ── Helper: Resolve API Base URL ─────────────────────────────────────────
    async function getApiBase() {
        if (cachedApiBase) return cachedApiBase;
        if (typeof window.resolveApiBase === "function") {
            try {
                cachedApiBase = await window.resolveApiBase();
                return cachedApiBase;
            } catch (e) {
                console.warn("Falling back to window.location.origin");
            }
        }
        cachedApiBase = window.location.origin.includes("5000")
            ? window.location.origin
            : "http://127.0.0.1:5000";
        return cachedApiBase;
    }

    // ── Initialization on DOM Ready ──────────────────────────────────────────
    window.addEventListener("DOMContentLoaded", () => {
        cacheDomElements();
        bindEvents();
        checkBackendReadiness();
    });

    function cacheDomElements() {
        modalEl = document.getElementById("asl-modal");
        backdropEl = document.getElementById("asl-modal-backdrop");
        previewVideo = document.getElementById("asl-webcam-preview");
        previewCanvas = document.getElementById("asl-capture-canvas");

        modeAlphaBtn = document.getElementById("asl-mode-alpha");
        modeWordBtn = document.getElementById("asl-mode-word");
        startBtn = document.getElementById("asl-start-btn");
        stopBtn = document.getElementById("asl-stop-btn");

        livePredEl = document.getElementById("asl-live-prediction");
        confValEl = document.getElementById("asl-confidence-val");
        confBarEl = document.getElementById("asl-confidence-bar");
        holdBarEl = document.getElementById("asl-hold-bar");
        holdContainerEl = document.getElementById("asl-hold-container");

        statusBadgeEl = document.getElementById("asl-status-badge");
        handBadgeEl = document.getElementById("asl-hand-badge");
        top5ListEl = document.getElementById("asl-top5-list");
        sentenceBoxEl = document.getElementById("asl-sentence-box");

        spaceBtn = document.getElementById("asl-space-btn");
        undoBtn = document.getElementById("asl-undo-btn");
        clearBtn = document.getElementById("asl-clear-btn");
        sendToSentixBtn = document.getElementById("asl-send-to-sentix-btn");
    }

    function bindEvents() {
        // ASL toggle button in the chat input dock
        const toggleBtn = document.getElementById("asl-toggle-btn");
        if (toggleBtn) {
            toggleBtn.addEventListener("click", () => openAslModal());
        }

        // Close modal button & backdrop click
        const closeBtn = document.getElementById("asl-close-modal-btn");
        if (closeBtn) {
            closeBtn.addEventListener("click", () => closeAslModal());
        }
        if (backdropEl) {
            backdropEl.addEventListener("click", (e) => {
                if (e.target === backdropEl) closeAslModal();
            });
        }

        // Mode switches
        if (modeAlphaBtn) {
            modeAlphaBtn.addEventListener("click", () => setMode("alphabet"));
        }
        if (modeWordBtn) {
            modeWordBtn.addEventListener("click", () => setMode("word"));
        }

        // Session Controls
        if (startBtn) startBtn.addEventListener("click", () => startRecognition());
        if (stopBtn) stopBtn.addEventListener("click", () => stopRecognition());

        // Sentence Actions
        if (spaceBtn) spaceBtn.addEventListener("click", () => doSpace());
        if (undoBtn) undoBtn.addEventListener("click", () => doUndo());
        if (clearBtn) clearBtn.addEventListener("click", () => doClear());
        if (sendToSentixBtn) sendToSentixBtn.addEventListener("click", () => sendToSentix());

        // Keyboard navigation inside modal
        document.addEventListener("keydown", (e) => {
            if (!modalEl || modalEl.classList.contains("hidden")) return;
            if (e.key === "Escape") {
                closeAslModal();
            } else if (e.code === "Space" && e.target === document.body) {
                e.preventDefault();
                doSpace();
            }
        });
    }

    // ── Backend Readiness Check ──────────────────────────────────────────────
    async function checkBackendReadiness() {
        try {
            const base = await getApiBase();
            const res = await fetch(`${base}/asl/status`);
            if (res.ok) {
                const data = await res.json();
                if (data.ready) {
                    setStatusBadge("ready", `ASL Ready (${data.device || "GPU/CPU"})`);
                } else {
                    setStatusBadge("warning", "Models loading...");
                }
            } else {
                setStatusBadge("disconnected", "ASL Backend Offline");
            }
        } catch (e) {
            setStatusBadge("disconnected", "Backend unreachable");
        }
    }

    // ── Modal Visibility ─────────────────────────────────────────────────────
    function openAslModal() {
        if (!modalEl || !backdropEl) return;
        modalEl.classList.remove("hidden");
        backdropEl.classList.remove("hidden");

        // Automatically start webcam and recognition session when opening
        startRecognition();
    }

    function closeAslModal() {
        if (!modalEl || !backdropEl) return;
        modalEl.classList.add("hidden");
        backdropEl.classList.add("hidden");

        // Stop frame processing loop and release resources
        stopRecognition();
    }

    // ── Mode Selection ───────────────────────────────────────────────────────
    function setMode(mode) {
        if (mode !== "alphabet" && mode !== "word") return;
        currentMode = mode;

        if (mode === "alphabet") {
            modeAlphaBtn.classList.add("active");
            modeWordBtn.classList.remove("active");
            if (holdContainerEl) holdContainerEl.style.display = "block";
            if (top5ListEl) top5ListEl.style.display = "none";
            if (spaceBtn) spaceBtn.style.display = "inline-flex";
        } else {
            modeAlphaBtn.classList.remove("active");
            modeWordBtn.classList.add("active");
            if (holdContainerEl) holdContainerEl.style.display = "none";
            if (top5ListEl) top5ListEl.style.display = "block";
            if (spaceBtn) spaceBtn.style.display = "none";
        }

        // Reset live prediction text
        if (livePredEl) livePredEl.textContent = "—";
        if (confValEl) confValEl.textContent = "0%";
        if (confBarEl) confBarEl.style.width = "0%";
    }

    // ── Recognition Lifecycle ────────────────────────────────────────────────
    async function startRecognition() {
        if (isRunning) return;

        setStatusBadge("connecting", "Starting camera...");
        try {
            // Coordinate shared webcam stream
            let stream = null;
            if (typeof window.startWebcam === "function") {
                stream = await window.startWebcam();
            } else {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
            }

            if (previewVideo && stream) {
                previewVideo.srcObject = stream;
                previewVideo.play().catch(() => {});
            }

            if (typeof window.setAslActive === "function") {
                window.setAslActive(true);
            }

            // Call backend /asl/start
            const base = await getApiBase();
            await fetch(`${base}/asl/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, mode: currentMode })
            });

            isRunning = true;
            if (startBtn) startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            setStatusBadge("active", "Live Recognizer Active");

            // Start single-flight throttled frame pump
            if (frameLoopTimer) clearInterval(frameLoopTimer);
            frameLoopTimer = setInterval(processFramePump, FRAME_RATE_MS);

        } catch (err) {
            console.error("Failed to start ASL recognition:", err);
            setStatusBadge("disconnected", "Camera access denied");
            if (typeof window.setAslActive === "function") {
                window.setAslActive(false);
            }
        }
    }

    async function stopRecognition() {
        isRunning = false;
        if (frameLoopTimer) {
            clearInterval(frameLoopTimer);
            frameLoopTimer = null;
        }

        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        setStatusBadge("ready", "Stopped");

        // Inform script.js that ASL is no longer using the camera
        if (typeof window.setAslActive === "function") {
            window.setAslActive(false);
        }

        // Inform backend to clean up session
        try {
            const base = await getApiBase();
            await fetch(`${base}/asl/stop`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });
        } catch (e) {
            // Ignore network teardown errors
        }
    }

    // ── Single-Flight Frame Pump ─────────────────────────────────────────────
    async function processFramePump() {
        if (!isRunning || isProcessingFrame) return;

        // Grab base64 frame from canvas or shared video
        const frameB64 = captureFrame();
        if (!frameB64) return;

        isProcessingFrame = true;
        try {
            const base = await getApiBase();
            const res = await fetch(`${base}/asl/frame`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    image: frameB64,
                    mode: currentMode
                })
            });

            if (res.ok) {
                const data = await res.json();
                renderInferenceResult(data);
            }
        } catch (err) {
            console.warn("ASL frame transmission error:", err);
        } finally {
            isProcessingFrame = false;
        }
    }

    function captureFrame() {
        if (!previewVideo || previewVideo.videoWidth === 0) {
            // Fallback to script.js capture if previewVideo is pending
            if (typeof window.captureWebcamImage === "function") {
                return window.captureWebcamImage();
            }
            return null;
        }

        if (!previewCanvas) {
            previewCanvas = document.createElement("canvas");
        }
        previewCanvas.width = 320; // 320x240 is optimal for fast MediaPipe landmark extraction
        previewCanvas.height = 240;
        const ctx = previewCanvas.getContext("2d");
        ctx.drawImage(previewVideo, 0, 0, 320, 240);
        return previewCanvas.toDataURL("image/jpeg", 0.7);
    }

    // ── Render Recognition Output ────────────────────────────────────────────
    function renderInferenceResult(data) {
        if (!data || !data.success) return;

        // Hand presence status
        if (handBadgeEl) {
            if (data.has_hand) {
                handBadgeEl.className = "asl-badge badge-hand-detected";
                handBadgeEl.textContent = "✋ Hand Detected";
            } else {
                handBadgeEl.className = "asl-badge badge-no-hand";
                handBadgeEl.textContent = "✋ No Hand Detected";
            }
        }

        // Live Prediction & Confidence
        const pred = data.prediction || "—";
        const confPercent = Math.round((data.confidence || 0) * 100);

        if (livePredEl) {
            livePredEl.textContent = pred;
            if (data.accepted) {
                livePredEl.classList.add("flash-accepted");
                setTimeout(() => livePredEl.classList.remove("flash-accepted"), 250);
            }
        }

        if (confValEl) confValEl.textContent = `${confPercent}%`;
        if (confBarEl) {
            confBarEl.style.width = `${confPercent}%`;
            confBarEl.className = "asl-progress-fill " + (confPercent > 70 ? "conf-high" : confPercent > 40 ? "conf-med" : "conf-low");
        }

        // Hold Progress for Alphabet Fingerspelling
        if (holdBarEl && typeof data.hold_progress === "number") {
            const holdPercent = Math.round(data.hold_progress * 100);
            holdBarEl.style.width = `${holdPercent}%`;
        }

        // Word Mode Top-5 list
        if (currentMode === "word" && top5ListEl && Array.isArray(data.top_predictions)) {
            top5ListEl.innerHTML = "";
            data.top_predictions.slice(0, 5).forEach((item) => {
                const li = document.createElement("div");
                li.className = "asl-top-item";
                const pct = Math.round((item.confidence || 0) * 100);
                li.innerHTML = `
                    <span class="asl-candidate-label">${item.label}</span>
                    <div class="asl-candidate-bar-wrapper">
                        <div class="asl-candidate-bar" style="width: ${pct}%"></div>
                    </div>
                    <span class="asl-candidate-pct">${pct}%</span>
                `;
                top5ListEl.appendChild(li);
            });
        }

        // Recognized Sentence Box
        if (sentenceBoxEl) {
            const fullSentence = data.sentence || "";
            sentenceBoxEl.textContent = fullSentence || "Recognized sentence will appear here...";
            if (fullSentence) {
                sentenceBoxEl.classList.add("has-content");
            } else {
                sentenceBoxEl.classList.remove("has-content");
            }
        }
    }

    // ── Sentence Operations ──────────────────────────────────────────────────
    async function doSpace() {
        try {
            const base = await getApiBase();
            const res = await fetch(`${base}/asl/commit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });
            if (res.ok) {
                const data = await res.json();
                if (sentenceBoxEl && data.sentence) {
                    sentenceBoxEl.textContent = data.sentence;
                }
            }
        } catch (e) {
            console.error("Space commit error:", e);
        }
    }

    async function doUndo() {
        try {
            const base = await getApiBase();
            const res = await fetch(`${base}/asl/undo`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });
            if (res.ok) {
                const data = await res.json();
                if (sentenceBoxEl) {
                    sentenceBoxEl.textContent = data.sentence || "Recognized sentence will appear here...";
                }
            }
        } catch (e) {
            console.error("Undo error:", e);
        }
    }

    async function doClear() {
        try {
            const base = await getApiBase();
            const res = await fetch(`${base}/asl/clear`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });
            if (res.ok) {
                if (sentenceBoxEl) {
                    sentenceBoxEl.textContent = "Recognized sentence will appear here...";
                    sentenceBoxEl.classList.remove("has-content");
                }
                if (livePredEl) livePredEl.textContent = "—";
            }
        } catch (e) {
            console.error("Clear error:", e);
        }
    }

    // ── Send to Sentix Chatbot Handoff ───────────────────────────────────────
    async function sendToSentix() {
        if (!sentenceBoxEl) return;
        const text = sentenceBoxEl.textContent.trim();
        if (!text || text === "Recognized sentence will appear here...") {
            return;
        }

        const chatInput = document.getElementById("input");
        if (chatInput) {
            chatInput.value = text;
        }

        // Close the ASL modal so the user sees the Sentix chat stream
        closeAslModal();

        // Trigger existing Sentix chat send mechanism with source="asl"
        if (typeof window.send === "function") {
            window.send("asl");
        }

        // Reset the ASL sentence buffer for next use
        doClear();
    }

    function setStatusBadge(state, text) {
        if (!statusBadgeEl) return;
        statusBadgeEl.className = `asl-badge badge-${state}`;
        statusBadgeEl.textContent = text;
    }

    // Export controller API to window
    window.ASLController = {
        openModal: openAslModal,
        closeModal: closeAslModal,
        start: startRecognition,
        stop: stopRecognition,
        setMode: setMode,
        sendToSentix: sendToSentix,
        clear: doClear,
    };
})();

