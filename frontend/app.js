const API = "http://localhost:9091";

const els = {
  docStatus: document.getElementById("docStatus"),
  statusDot: document.getElementById("statusDot"),
  processBtn: document.getElementById("processBtn"),
  messages: document.getElementById("messages"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
};

let pollTimer = null;
let ready = false;
let waitingForBackend = false;
let pollAttempts = 0;
const MAX_POLL_ATTEMPTS = 45;
const MAX_HISTORY = 5;
const chatHistory = [];

function addBubble(html, role = "bot", className = "") {
  const div = document.createElement("div");
  div.className = `bubble ${role} ${className}`.trim();
  div.innerHTML = html;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function addTyping() {
  const div = document.createElement("div");
  div.className = "bubble bot";
  div.dataset.typing = "1";
  div.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function setUiStatus({ status, message, documents }) {
  els.statusDot.dataset.status = status || "idle";
  if (status === "ready") {
    els.docStatus.textContent = `${documents?.length || 0} papers ready`;
  } else if (status === "processing") {
    els.docStatus.textContent = "Indexing…";
  } else if (status === "error") {
    els.docStatus.textContent = "Unavailable";
  } else {
    els.docStatus.textContent = documents?.length ? `${documents.length} papers · not indexed` : "Not indexed";
  }
  ready = status === "ready";
  els.sendBtn.disabled = !ready || !els.questionInput.value.trim();
  els.processBtn.disabled = status === "processing";
}

async function fetchStatus() {
  const res = await fetch(`${API}/status`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error("Could not reach backend");
  return res.json();
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    try {
      const data = await fetchStatus();
      waitingForBackend = false;
      pollAttempts = 0;
      setUiStatus(data);
      if (data.status === "ready") {
        stopPolling();
      } else if (data.status === "error") {
        stopPolling();
        addBubble(data.message || "Indexing failed.", "system");
      }
    } catch (_) {
      pollAttempts += 1;
      if (pollAttempts >= MAX_POLL_ATTEMPTS) {
        stopPolling();
        setUiStatus({ status: "error", message: "Backend offline" });
        return;
      }
      if (!waitingForBackend) {
        waitingForBackend = true;
        setUiStatus({ status: "idle", message: "Connecting…" });
      }
    }
  }, 3000);
}

async function startProcess() {
  chatHistory.length = 0;
  addBubble("Indexing research papers…", "system");
  setUiStatus({ status: "processing", message: "Indexing…" });
  try {
    const res = await fetch(`${API}/process`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Process failed");
    setUiStatus(data);
    startPolling();
  } catch (err) {
    addBubble(err.message, "system");
    setUiStatus({ status: "error", message: err.message });
  }
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function pushHistory(question, minilmAnswer, bgeAnswer) {
  chatHistory.push({
    question,
    minilm_answer: minilmAnswer || "",
    bge_answer: bgeAnswer || "",
  });
  while (chatHistory.length > MAX_HISTORY) chatHistory.shift();
}

function setCompareLoading(loading) {
  els.sendBtn.disabled = loading || !ready || !els.questionInput.value.trim();
  els.questionInput.disabled = loading;
  els.processBtn.disabled = loading || els.docStatus.textContent.includes("Indexing");
}

async function sendQuestion(question) {
  if (!ready) {
    const status = els.statusDot.dataset.status;
    if (status === "processing") {
      addBubble("Papers are still being indexed. Please wait…", "system");
    } else if (status === "error" || waitingForBackend) {
      addBubble("Service unavailable. Try again in a moment.", "system");
    } else {
      addBubble("Index the library first using Rebuild index.", "system");
    }
    return;
  }

  addBubble(escapeHtml(question), "user");
  const typing = addTyping();
  setCompareLoading(true);

  try {
    const res = await fetch(`${API}/chat/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: 8,
        history: chatHistory,
      }),
    });
    const data = await res.json().catch(() => ({}));
    typing.remove();

    if (!res.ok) {
      const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || "Request failed.");
      addBubble(escapeHtml(msg), "bot");
      return;
    }

    const minilmText = (data.minilm?.answer || "").trim() || "No answer returned.";
    const bgeText = (data.bge?.answer || "").trim() || "No answer returned.";

    addBubble(
      `<h3 class="compare-title">MiniLM</h3><p>${escapeHtml(minilmText)}</p>`,
      "bot",
      "compare-minilm"
    );
    addBubble(
      `<h3 class="compare-title">BGE</h3><p>${escapeHtml(bgeText)}</p>`,
      "bot",
      "compare-bge"
    );
    pushHistory(question, data.minilm?.answer, data.bge?.answer);
  } catch (err) {
    typing.remove();
    addBubble(escapeHtml(err.message || "Network error."), "bot");
  } finally {
    setCompareLoading(false);
  }
}

els.processBtn.addEventListener("click", startProcess);

els.questionInput.addEventListener("input", () => {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(els.questionInput.scrollHeight, 140)}px`;
  els.sendBtn.disabled = !ready || !els.questionInput.value.trim();
});

els.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.chatForm.requestSubmit();
  }
});

els.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = els.questionInput.value.trim();
  if (!q) return;
  els.questionInput.value = "";
  els.questionInput.style.height = "auto";
  els.sendBtn.disabled = true;
  await sendQuestion(q);
});

(async function init() {
  try {
    const data = await fetchStatus();
    if (data.status === "ready") {
      setUiStatus({ status: "ready", message: `${data.documents?.length || 0} papers ready` });
    } else {
      setUiStatus(data);
      if (data.status === "processing") startPolling();
    }
  } catch (_) {
    startPolling();
  }
})();
