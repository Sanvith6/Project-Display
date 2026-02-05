const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadStage = document.getElementById('upload-stage');
const processStage = document.getElementById('process-stage');
const resultStage = document.getElementById('result-stage');

const statusMessage = document.getElementById('status-message');
const stepCount = document.getElementById('step-count');
const progressFill = document.getElementById('progress-fill');
const logContent = document.getElementById('log-content');
const finalVideo = document.getElementById('final-video');
const downloadLink = document.getElementById('download-link');

// Chat Elements
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('btn-send');
const chatBox = document.getElementById('chat-box');

const API_BASE = window.location.origin;
let lastLoggedMessage = "";

// --- Drag & Drop ---
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--neon-green)';
    dropZone.style.backgroundColor = 'rgba(0, 255, 157, 0.05)';
});
dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--glass-border)';
    dropZone.style.backgroundColor = 'transparent';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
    dropZone.style.backgroundColor = 'transparent';
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// --- Chat Interactions ---
sendBtn.addEventListener('click', sendChat);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChat();
});

async function sendChat() {
    const text = chatInput.value.trim();
    if (!text) return;

    // User Message
    appendMessage(text, 'user');
    chatInput.value = '';

    // Typing indicator
    const typingId = appendMessage("Analysing match data...", 'ai typing');

    try {
        const timestamp = finalVideo.currentTime || 0;
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: text,
                timestamp: timestamp
            })
        });

        const data = await res.json();

        // Remove typing indicator
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        // AI Response
        if (data.answer) {
            appendMessage(data.answer, 'ai');
        } else {
            appendMessage("Signal interference. Could not process.", 'ai error');
        }

    } catch (err) {
        console.error(err);
        appendMessage("System Error: Connection lost.", 'ai error');
    }
}

function appendMessage(text, type) {
    const id = 'msg-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = `msg ${type}`;
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

// --- Upload & Process Logic ---

async function handleFile(file) {
    // Relaxed check: Allow video/* OR file names ending in .mp4
    if (!file.type.startsWith('video/') && !file.name.toLowerCase().endsWith('.mp4')) {
        alert("Please upload a video file (MP4 recommended).");
        return;
    }

    switchStage(uploadStage, processStage);
    statusMessage.innerText = "UPLOADING FOOTAGE...";
    log("Starting upload...");

    const formData = new FormData();
    formData.append('file', file);

    try {
        const upResp = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        if (!upResp.ok) throw new Error("Upload failed");
        const upData = await upResp.json();
        const filename = upData.filename;

        log("Upload complete. Initializing pipeline...");
        startProcessing(filename);

    } catch (err) {
        showError(err.message);
    }
}

function switchStage(current, next) {
    current.classList.remove('active');
    // small delay for transition
    setTimeout(() => {
        next.classList.add('active');
    }, 100);
}

async function startProcessing(filename) {
    statusMessage.innerText = "MATCH IN PROGRESS...";

    // Get options
    const hasScorecard = document.querySelector('input[name="scorecard"]:checked').value === 'yes';

    try {
        await fetch(`${API_BASE}/process?filename=${encodeURIComponent(filename)}&has_scorecard=${hasScorecard}`, { method: 'POST' });
        pollStatus();
    } catch (err) {
        showError("Failed to start processing");
    }
}

async function pollStatus() {
    let interval = setInterval(async () => {
        try {
            const resp = await fetch(`${API_BASE}/status`);
            const data = await resp.json();

            if (data.logs && data.logs.length) {
                // Only show last log if different
                const lastLog = data.logs[data.logs.length - 1];
                if (lastLog !== lastLoggedMessage) {
                    log(lastLog);
                    lastLoggedMessage = lastLog;
                }
            }

            const steps = data.step || 0;
            const total = data.total_steps || 7;
            stepCount.innerText = `${steps}/${total}`;
            const percent = (steps / total) * 100;
            progressFill.style.width = `${percent}%`;

            if (data.status === 'completed') {
                clearInterval(interval);
                showResult();
            } else if (data.status === 'error') {
                clearInterval(interval);
                showError(data.message);
            }

        } catch (err) {
            console.error("Polling error", err);
        }
    }, 1000);
}

async function showResult() {
    switchStage(processStage, resultStage);

    const timestamp = new Date().getTime();
    finalVideo.src = `${API_BASE}/result?t=${timestamp}`;
    downloadLink.href = `${API_BASE}/result`;

    // Auto-play
    finalVideo.play().catch(e => console.log("Autoplay prevented"));
}

function showError(msg) {
    alert("Error: " + msg);
    switchStage(processStage, uploadStage);
}

function log(msg) {
    const entry = document.createElement('div');
    entry.className = 'log-line';
    entry.innerText = msg;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}
