const STORAGE_KEY = "mika_chats_v1";
const THEME_KEY = "mika_theme_v1";
const ZOOM_KEY = "mika_zoom_v1";
const LANG_KEY = "mika_lang_v1";
let chats = [];
let activeChatId = null;
let isSending = false;
let typingNode = null;
let apiBase = null;
let currentZoom = 1;

let webcamStream = null;
const WEBCAM_KEY = "mika_webcam_v1";

const API_BASE_CANDIDATES = [
    "http://127.0.0.1:5000",
    "http://localhost:5000"
];

// BCP-47 speech codes mapped to each supported language
const SPEECH_LANG_MAP = {
    "English":    "en-US",
    "Hindi":      "hi-IN",
    "Bengali":    "bn-IN",
    "Tamil":      "ta-IN",
    "Telugu":     "te-IN",
    "Marathi":    "mr-IN",
    "Gujarati":   "gu-IN",
    "Kannada":    "kn-IN",
    "Malayalam":  "ml-IN",
    "Punjabi":    "pa-IN",
    "Odia":       "or-IN",
    "Assamese":   "as-IN",
};

window.onload = () => {
    wireUi();
    wireWebcam();
    loadChats();
    renderChatList();
    if (!activeChatId) {
        createNewChat();
    } else {
        setActiveChat(activeChatId);
    }

    checkServerStatus();
    setInterval(checkServerStatus, 5000);
};

function getSelectedLanguage() {
    const sel = document.getElementById("language-select");
    return sel ? sel.value : "English";
}

// [UPGRADE 6] Speech error toast notification
function showSpeechError(msg) {
    const existing = document.querySelector('.speech-error-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'speech-error-toast';
    toast.textContent = msg;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function wireWebcam() {
    const webcamSwitch = document.getElementById("webcam-switch");
    if (!webcamSwitch) return;

    const savedWebcam = localStorage.getItem(WEBCAM_KEY) === "true";
    webcamSwitch.checked = savedWebcam;
    if (savedWebcam) {
        startWebcam();
    }

    webcamSwitch.addEventListener("change", () => {
        const isEnabled = webcamSwitch.checked;
        localStorage.setItem(WEBCAM_KEY, isEnabled.toString());
        if (isEnabled) {
            startWebcam();
        } else {
            stopWebcam();
        }
    });

    // Ensure video elements exist
    let video = document.getElementById("webcam-feed");
    if (!video) {
        video = document.createElement("video");
        video.id = "webcam-feed";
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;
        video.style.display = "none";
        document.body.appendChild(video);
    }
    let canvas = document.getElementById("webcam-canvas");
    if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.id = "webcam-canvas";
        canvas.style.display = "none";
        document.body.appendChild(canvas);
    }
}

async function startWebcam() {
    try {
        const video = document.getElementById("webcam-feed");
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = webcamStream;
    } catch (err) {
        console.error("Camera access denied or unavailable", err);
        const webcamSwitch = document.getElementById("webcam-switch");
        if (webcamSwitch) webcamSwitch.checked = false;
        localStorage.setItem(WEBCAM_KEY, "false");
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    const video = document.getElementById("webcam-feed");
    if (video) video.srcObject = null;
}

function captureWebcamImage() {
    if (!webcamStream) return null;
    const video = document.getElementById("webcam-feed");
    const canvas = document.getElementById("webcam-canvas");
    if (!video || !canvas || video.videoWidth === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8);
}

function wireUi() {
    const newChatBtn = document.getElementById("new-chat");
    const toggleBtn = document.getElementById("toggle-sidebar");
    const themeSwitch = document.getElementById("theme-switch");

    newChatBtn.addEventListener("click", createNewChat);
    toggleBtn.addEventListener("click", () => {
        const isCollapsed = document.body.classList.toggle("sidebar-collapsed");
        toggleBtn.textContent = isCollapsed ? ">" : "<";
    });
    
    // Settings and Zoom UI
    const settingsBtn = document.getElementById("settings-btn");
    const settingsMenu = document.getElementById("settings-menu");
    const zoomInBtn = document.getElementById("zoom-in-btn");
    const zoomOutBtn = document.getElementById("zoom-out-btn");

    if (settingsBtn && settingsMenu) {
        settingsBtn.addEventListener("click", () => {
            settingsMenu.classList.toggle("hidden");
        });
        
        document.addEventListener("click", (e) => {
            if (!settingsBtn.contains(e.target) && !settingsMenu.contains(e.target)) {
                settingsMenu.classList.add("hidden");
            }
        });
    }

    const savedZoom = localStorage.getItem(ZOOM_KEY);
    if (savedZoom) {
        currentZoom = parseFloat(savedZoom);
        applyZoom();
    }

    if (zoomInBtn && zoomOutBtn) {
        zoomInBtn.addEventListener("click", () => {
            if (currentZoom < 1.5) {
                currentZoom = Math.min(1.5, currentZoom + 0.1);
                applyZoom();
            }
        });

        zoomOutBtn.addEventListener("click", () => {
            if (currentZoom > 0.5) {
                currentZoom = Math.max(0.5, currentZoom - 0.1);
                applyZoom();
            }
        });
    }

    const savedTheme = localStorage.getItem(THEME_KEY) || "dark";
    applyTheme(savedTheme, themeSwitch);

    themeSwitch.addEventListener("change", () => {
        const nextTheme = themeSwitch.checked ? "dark" : "light";
        applyTheme(nextTheme, themeSwitch);
        localStorage.setItem(THEME_KEY, nextTheme);
    });

    // ── Language selector persistence ────────────────────────────
    const langSelect = document.getElementById("language-select");
    if (langSelect) {
        const savedLang = localStorage.getItem(LANG_KEY);
        if (savedLang) {
            langSelect.value = savedLang;
        }
        langSelect.addEventListener("change", () => {
            localStorage.setItem(LANG_KEY, langSelect.value);
            // Update speech recognition language when dropdown changes
            if (typeof updateRecognitionLang === "function") {
                updateRecognitionLang(langSelect.value);
            }
        });
    }

    // ── Voice-to-text ──────────────────────────────────────────────
    const micBtn = document.getElementById("mic-btn");
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        if (micBtn) {
            micBtn.title = "Voice input not supported in this browser";
            micBtn.style.opacity = "0.4";
            micBtn.style.cursor = "not-allowed";
        }
    } else {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;

        // Set initial language from dropdown
        const initLang = getSelectedLanguage();
        recognition.lang = SPEECH_LANG_MAP[initLang] || "en-US";

        let isListening = false;
        let baseText = "";   // text already in the input before mic was pressed
        let speechTimeout = null;  // [UPGRADE 6] safety timeout

        // Expose a function so the language dropdown can update recognition.lang
        window.updateRecognitionLang = (langName) => {
            recognition.lang = SPEECH_LANG_MAP[langName] || "en-US";
        };

        micBtn.addEventListener("click", () => {
            if (isListening) {
                isListening = false;
                recognition.stop();
                if (speechTimeout) { clearTimeout(speechTimeout); speechTimeout = null; }
            } else {
                const currentLang = getSelectedLanguage();
                recognition.lang = SPEECH_LANG_MAP[currentLang] || "en-US";
                baseText = document.getElementById("input").value;
                isListening = true;
                try {
                    recognition.start();
                } catch (err) {
                    console.error("Failed to start speech recognition:", err);
                    isListening = false;
                    showSpeechError("Could not start voice input. Please type instead.");
                    return;
                }
                // [UPGRADE 6] Auto-stop after 60 seconds
                speechTimeout = setTimeout(() => {
                    if (isListening) {
                        isListening = false;
                        recognition.stop();
                        showSpeechError("Voice input timed out. Click mic to try again.");
                    }
                }, 60000);
            }
        });

        recognition.onstart = () => {
            micBtn.classList.add("mic-active");
            micBtn.title = "Click to stop";
        };

        recognition.onresult = (e) => {
            let interim = "";
            let final = "";
            for (let i = e.resultIndex; i < e.results.length; i++) {
                const t = e.results[i][0].transcript;
                if (e.results[i].isFinal) {
                    final += t;
                } else {
                    interim += t;
                }
            }
            document.getElementById("input").value = baseText + final + interim;
            if (final) baseText += final;
        };

        recognition.onend = () => {
            if (isListening) {
                try { recognition.start(); } catch (_) { /* browser may block rapid restarts */ }
            } else {
                micBtn.classList.remove("mic-active");
                micBtn.title = "Voice Input";
                if (speechTimeout) { clearTimeout(speechTimeout); speechTimeout = null; }
            }
        };

        // [UPGRADE 6] Robust speech error handling with user feedback
        recognition.onerror = (e) => {
            if (e.error === "no-speech" || e.error === "audio-capture") return;
            console.error("Speech recognition error:", e.error);
            isListening = false;
            micBtn.classList.remove("mic-active");
            micBtn.title = "Voice Input";
            if (speechTimeout) { clearTimeout(speechTimeout); speechTimeout = null; }

            let errorMsg = "Voice input failed. Please type your message.";
            if (e.error === "not-allowed") {
                errorMsg = "Microphone access denied. Please allow mic in browser settings.";
            } else if (e.error === "network") {
                errorMsg = "Network error during voice input. Check your connection.";
            } else if (e.error === "language-not-supported") {
                errorMsg = `Voice not available for ${getSelectedLanguage()}. Please type instead.`;
            }
            showSpeechError(errorMsg);
        };
    }
}

function applyTheme(theme, themeSwitch) {
    const isDark = theme === "dark";
    document.body.classList.toggle("theme-light", !isDark);
    if (themeSwitch) {
        themeSwitch.checked = isDark;
    }
}

function applyZoom() {
    document.body.style.zoom = currentZoom;
    const zoomLevelText = document.getElementById("zoom-level-text");
    if (zoomLevelText) {
        zoomLevelText.textContent = `${Math.round(currentZoom * 100)}%`;
    }
    localStorage.setItem(ZOOM_KEY, currentZoom);
}

function defaultChat() {
    const id = `chat_${Date.now()}`;
    return {
        id,
        title: "Mike",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [
            {
                role: "bot",
                text: "Hi! I'm Mike. How are you feeling today?",
                ts: Date.now()
            }
        ]
    };
}

function loadChats() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            chats = Array.isArray(parsed.chats) ? parsed.chats : [];
            activeChatId = parsed.activeChatId || null;
        }
    } catch (err) {
        chats = [];
        activeChatId = null;
    }

    // Migrate old titles
    let migrated = false;
    chats = chats.map((chat) => {
        if (chat.title === "Chat with Mika") {
            migrated = true;
            return { ...chat, title: "Mike" };
        }
        return chat;
    });
    if (migrated) {
        saveChats();
    }
}

function saveChats() {
    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ chats, activeChatId })
    );
}

function createNewChat() {
    const chat = defaultChat();
    chats.unshift(chat);
    activeChatId = chat.id;
    saveChats();
    renderChatList();
    setActiveChat(chat.id);
}

function deleteChat(chatId) {
    const chat = chats.find((item) => item.id === chatId);
    if (!chat) return;

    const ok = window.confirm(`Delete "${chat.title}"? This cannot be undone.`);
    if (!ok) return;

    chats = chats.filter((item) => item.id !== chatId);
    if (activeChatId === chatId) {
        activeChatId = chats.length ? chats[0].id : null;
    }
    saveChats();
    renderChatList();
    if (activeChatId) {
        setActiveChat(activeChatId);
    } else {
        createNewChat();
    }
}

function setActiveChat(chatId) {
    activeChatId = chatId;
    saveChats();
    renderChatList();
    renderMessages();
    updateHeader();
}

function updateHeader() {
    const title = document.getElementById("chat-title");
    const chat = chats.find((item) => item.id === activeChatId);
    title.textContent = chat ? chat.title : "Mike";
}

async function checkServerStatus() {
    const statusEl = document.querySelector(".status");
    const sidebarDot = document.querySelector(".brand-dot");
    if (!statusEl) return;

    try {
        const base = await resolveApiBase();
        const res = await fetch(`${base}/status`);
        const data = await res.json();
        const isOnline = data && data.status === "online";

        if (isOnline) {
            statusEl.classList.remove("offline");
            statusEl.textContent = "Active now";
            if (sidebarDot) sidebarDot.style.background = "#22c55e";
        } else {
            statusEl.classList.add("offline");
            statusEl.textContent = "Offline";
            if (sidebarDot) sidebarDot.style.background = "#ef4444";
        }
    } catch (err) {
        console.error("Failed to fetch server status", err);
        statusEl.classList.add("offline");
        statusEl.textContent = "Offline";
        if (sidebarDot) sidebarDot.style.background = "#ef4444";
    }
}

async function resolveApiBase() {
    if (apiBase) return apiBase;

    for (const candidate of API_BASE_CANDIDATES) {
        const url = `${candidate}/status`;
        try {
            const res = await fetch(url);
            if (!res.ok) continue;
            const data = await res.json();
            if (data && typeof data.status === "string") {
                apiBase = candidate;
                return apiBase;
            }
        } catch (_err) {
            // Try next candidate.
        }
    }
    throw new Error("No reachable backend API base URL.");
}

function renderChatList() {
    const list = document.getElementById("chat-list");
    list.innerHTML = "";

    const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);
    sorted.forEach((chat) => {
        const item = document.createElement("div");
        item.className = `chat-item${chat.id === activeChatId ? " active" : ""}`;

        const img = document.createElement("img");
        img.src = "https://i.pinimg.com/564x/8d/ff/49/8dff49985d0d8afa53751d9ba8907aed.jpg";
        img.alt = "Avatar";

        const content = document.createElement("div");
        content.className = "chat-item-content";

        const title = document.createElement("h4");
        title.textContent = chat.title;

        const preview = document.createElement("p");
        preview.textContent = getPreview(chat);

        content.appendChild(title);
        content.appendChild(preview);

        const time = document.createElement("span");
        time.className = "time";
        time.textContent = formatTime(chat.updatedAt);

        const del = document.createElement("button");
        del.className = "delete-btn";
        del.type = "button";
        del.title = "Delete chat";
        del.textContent = "✕";
        del.addEventListener("click", (event) => {
            event.stopPropagation();
            deleteChat(chat.id);
        });

        item.appendChild(img);
        item.appendChild(content);
        item.appendChild(time);
        item.appendChild(del);

        item.addEventListener("click", () => setActiveChat(chat.id));
        list.appendChild(item);
    });
}

function getPreview(chat) {
    const last = chat.messages[chat.messages.length - 1];
    if (!last) return "No messages yet";
    if (last.text.length <= 36) return last.text;
    return `${last.text.slice(0, 36)}...`;
}

function formatTime(ts) {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;

    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const isYesterday =
        date.getFullYear() === yesterday.getFullYear() &&
        date.getMonth() === yesterday.getMonth() &&
        date.getDate() === yesterday.getDate();
    if (isYesterday) return "Yesterday";

    if (diffDays < 7) {
        return date.toLocaleDateString(undefined, { weekday: "short" });
    }

    if (date.getFullYear() === now.getFullYear()) {
        return date.toLocaleDateString(undefined, {
            month: "short",
            day: "numeric"
        });
    }

    return date.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric"
    });
}

function renderMessages() {
    const chat = document.getElementById("chat");
    chat.innerHTML = "";

    const active = chats.find((item) => item.id === activeChatId);
    if (!active) return;

    active.messages.forEach((msg) => {
        const bubble = document.createElement("div");
        bubble.className = `message ${msg.role === "user" ? "user" : "bot"}`;
        bubble.textContent = msg.text;
        chat.appendChild(bubble);
    });

    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
}

function showTypingIndicator() {
    const chat = document.getElementById("chat");
    if (!chat) return;

    // Remove any existing indicator before adding a new one
    hideTypingIndicator();

    typingNode = document.createElement("div");
    typingNode.className = "typing-indicator";

    const lang = getSelectedLanguage();
    if (lang !== "English") {
        typingNode.textContent = "Translating & thinking...";
    } else {
        typingNode.textContent = "Mike is typing...";
    }

    chat.appendChild(typingNode);
    chat.scrollTop = chat.scrollHeight;
}

function hideTypingIndicator() {
    if (typingNode && typingNode.parentNode) {
        typingNode.parentNode.removeChild(typingNode);
    }
    typingNode = null;
}

function setSendingState(sending) {
    const sendBtn = document.getElementById("send-btn");
    if (sendBtn) {
        sendBtn.disabled = sending;
        sendBtn.classList.toggle("is-loading", sending);
    }
    isSending = sending;
}

async function send() {
    const input = document.getElementById("input");
    const chat = document.getElementById("chat");
    const text = input.value.trim();

    if (!text || isSending) return;
    if (!activeChatId) createNewChat();

    const active = chats.find((item) => item.id === activeChatId);
    if (!active) return;

    const language = getSelectedLanguage();
    const imageBase64 = captureWebcamImage();

    setSendingState(true);

    active.messages.push({
        role: "user",
        text,
        ts: Date.now()
    });
    active.updatedAt = Date.now();
    saveChats();
    renderMessages();
    renderChatList();

    input.value = "";
    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
    showTypingIndicator();

    try {
        // Backend call — now includes language and image
        const base = await resolveApiBase();
        const history = active.messages.slice(-8).map((msg) => ({
            role: msg.role,
            text: msg.text
        }));
        const res = await fetch(`${base}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, history, language, image: imageBase64 })
        });

        if (!res.ok) {
            throw new Error(`Backend returned HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        
        let botMessage = { role: "bot", text: "", ts: Date.now() };
        active.messages.push(botMessage);

        // Bot bubble is created lazily on first chunk; typing indicator stays until then
        const chatEl = document.getElementById("chat");
        let botBubble = null;

        let buffer = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            let newlineIdx;
            while ((newlineIdx = buffer.indexOf('\n')) >= 0) {
                const line = buffer.slice(0, newlineIdx);
                buffer = buffer.slice(newlineIdx + 1);
                
                if (line.trim()) {
                    try {
                        const parsed = JSON.parse(line);

                        if (parsed.type === "chunk" || parsed.type === "full_response") {
                            // Unified handler: works for English word-by-word
                            // AND non-English sentence-level streaming
                            if (!botBubble) {
                                hideTypingIndicator();
                                botBubble = document.createElement("div");
                                botBubble.className = "message bot";
                                chatEl.appendChild(botBubble);
                            }
                            botMessage.text += parsed.data;
                            botBubble.textContent = botMessage.text;
                            chatEl.scrollTop = chatEl.scrollHeight;
                        }
                    } catch (e) {
                        console.error("Stream parse error", e, line);
                    }
                }
            }
        }
        
        active.updatedAt = Date.now();
        saveChats();
        renderChatList();
    } catch (err) {
        console.error("Failed to reach Mika backend", err);
        active.messages.push({
            role: "bot",
            text: "Sorry, backend is unreachable. Run start_project.bat, wait 10 seconds, then refresh this page.",
            ts: Date.now()
        });
        active.updatedAt = Date.now();
        saveChats();
        renderMessages();
        renderChatList();
    } finally {
        hideTypingIndicator();
        setSendingState(false);
    }
}
