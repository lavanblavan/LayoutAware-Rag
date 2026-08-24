const API = "http://localhost:9091";

const els = {
  docStatus: document.getElementById("docStatus"),
  statusDot: document.getElementById("statusDot"),
  messages: document.getElementById("messages"),
  chatForm: document.getElementById("chatForm"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
};

let ready = false;
let modelReady = false;
let pollTimer = null;
let pendingQuestion = null;
let sending = false;
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
  div.className = "bubble bot compare-finetuned";
  div.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function setLoading(loading) {
  sending = loading;
  els.sendBtn.disabled = loading || !ready || !modelReady || !els.questionInput.value.trim();
  els.questionInput.disabled = loading;
}

function pushHistory(question, answer) {
  chatHistory.push({
    question,
    base_answer: "",
    finetuned_answer: answer || "",
  });
  while (chatHistory.length > MAX_HISTORY) chatHistory.shift();
}

async function fetchLibraryStatus() {
  const res = await fetch(`${API}/status`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error("Could not reach backend");
  return res.json();
}

async function fetchModelStatus() {
  const res = await fetch(`${API}/chat/finetuned/status`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error("Fine-tuned status unavailable");
  return res.json();
}

function updateModelUi(data) {
  modelReady = !!data?.ready;
  if (data?.error) {
    els.docStatus.textContent = "Unavailable";
    els.statusDot.dataset.status = "error";
    return;
  }
  if (modelReady) {
    els.docStatus.textContent = "Fine-tuned · ready";
    els.statusDot.dataset.status = "ready";
  } else {
    els.docStatus.textContent = "Loading model…";
    els.statusDot.dataset.status = "processing";
  }
  els.sendBtn.disabled = sending || !ready || !modelReady || !els.questionInput.value.trim();
}

function startModelPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const data = await fetchModelStatus();
      updateModelUi(data);
      if (data.ready) {
        clearInterval(pollTimer);
        pollTimer = null;
        if (pendingQuestion) {
          const q = pendingQuestion;
          pendingQuestion = null;
          sendQuestion(q);
        }
      }
    } catch (_) {
      /* keep polling */
    }
  }, 3000);
}

async function sendQuestion(question) {
  if (!ready) {
    addBubble("Research library not ready. Build the index on the main page first.", "system");
    return;
  }

  if (!modelReady) {
    pendingQuestion = question;
    addBubble(escapeHtml(question), "user");
    addBubble("Fine-tuned model is loading — your question will run automatically.", "system");
    startModelPolling();
    return;
  }

  addBubble(escapeHtml(question), "user");
  const typing = addTyping();
  setLoading(true);

  try {
    const res = await fetch(`${API}/chat/finetuned/ask`, {
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
      const msg =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail || "Request failed.");
      addBubble(escapeHtml(msg), "bot");
      if (res.status === 503) startModelPolling();
      return;
    }

    const answer = (data.answer || "").trim() || "No answer returned.";
    addBubble(`<p>${escapeHtml(answer)}</p>`, "bot", "compare-finetuned");
    pushHistory(question, data.answer);
  } catch (err) {
    typing.remove();
    addBubble(escapeHtml(err.message || "Network error."), "bot");
  } finally {
    setLoading(false);
  }
}

els.questionInput.addEventListener("input", () => {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(els.questionInput.scrollHeight, 140)}px`;
  els.sendBtn.disabled = sending || !ready || !modelReady || !els.questionInput.value.trim();
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
  if (!q || sending) return;
  els.questionInput.value = "";
  els.questionInput.style.height = "auto";
  await sendQuestion(q);
});

(async function init() {
  try {
    const lib = await fetchLibraryStatus();
    ready = lib.status === "ready";
    if (!ready) {
      els.docStatus.textContent = "Library not ready";
      els.statusDot.dataset.status = "error";
      return;
    }
    const model = await fetchModelStatus();
    updateModelUi(model);
    if (!model.ready) startModelPolling();
  } catch (_) {
    els.statusDot.dataset.status = "error";
    els.docStatus.textContent = "Offline";
  }
})();
