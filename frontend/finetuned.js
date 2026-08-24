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
let finetunedReady = false;
let finetunedPollTimer = null;
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

function setTypingMessage(node, html) {
  if (node) node.innerHTML = html;
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addTyping(message = "") {
  const div = document.createElement("div");
  div.className = "bubble bot system";
  div.innerHTML = message
    ? `<p>${message}</p><span class="typing"><i></i><i></i><i></i></span>`
    : '<span class="typing"><i></i><i></i><i></i></span>';
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

function buildCompareHtml(side, answerText) {
  const title =
    side === "base"
      ? "Base BGE"
      : `<span class="compare-title-tuned">Fine-tuned BGE</span>`;
  const answerBlock = answerText
    ? `<p class="answer-text">${escapeHtml(answerText)}</p>`
    : `<p class="answer-pending"><em>Generating answer…</em></p>`;
  return `<h3 class="compare-title">${title}</h3>${answerBlock}`;
}

function setLoading(loading) {
  sending = loading;
  els.sendBtn.disabled = loading || !ready || !els.questionInput.value.trim();
  els.questionInput.disabled = loading;
}

function pushHistory(question, baseAnswer, finetunedAnswer) {
  chatHistory.push({
    question,
    base_answer: baseAnswer || "",
    finetuned_answer: finetunedAnswer || "",
  });
  while (chatHistory.length > MAX_HISTORY) chatHistory.shift();
}

async function postJson(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  return { res, data };
}

async function fetchLibraryStatus() {
  const res = await fetch(`${API}/status`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error("Could not reach backend");
  return res.json();
}

async function fetchFinetunedStatus() {
  const res = await fetch(`${API}/chat/finetuned/status`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error("Fine-tuned status unavailable");
  return res.json();
}

function updateFinetunedUi(data) {
  finetunedReady = !!data?.ready;
  if (data?.error) {
    els.docStatus.textContent = "Unavailable";
    els.statusDot.dataset.status = "error";
    return;
  }
  if (finetunedReady) {
    els.docStatus.textContent = "Ready";
    els.statusDot.dataset.status = "ready";
  } else {
    els.docStatus.textContent = "Loading model…";
    els.statusDot.dataset.status = "processing";
  }
  els.sendBtn.disabled = sending || !ready || !els.questionInput.value.trim();
}

function startFinetunedPolling() {
  if (finetunedPollTimer) clearInterval(finetunedPollTimer);
  finetunedPollTimer = setInterval(async () => {
    try {
      const data = await fetchFinetunedStatus();
      updateFinetunedUi(data);
      if (data.ready) {
        clearInterval(finetunedPollTimer);
        finetunedPollTimer = null;
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
    addBubble("Research library not ready. Open the main page and rebuild the index.", "system");
    return;
  }

  if (!finetunedReady) {
    pendingQuestion = question;
    addBubble(escapeHtml(question), "user");
    addBubble("Model loading — your question will run automatically.", "system");
    startFinetunedPolling();
    return;
  }

  addBubble(escapeHtml(question), "user");
  const typing = addTyping("Finding relevant passages…");
  setLoading(true);

  let baseBubble = null;
  let tunedBubble = null;

  try {
    const payload = { question, top_k: 8, history: chatHistory, with_answers: false };
    const { res: retRes, data: retData } = await postJson("/chat/finetuned/retrieve", payload);
    typing.remove();

    if (!retRes.ok) {
      const msg =
        typeof retData.detail === "string"
          ? retData.detail
          : JSON.stringify(retData.detail || "Retrieval failed.");
      addBubble(escapeHtml(msg), "bot");
      if (retRes.status === 503) startFinetunedPolling();
      return;
    }

    baseBubble = addBubble(buildCompareHtml("base", ""), "bot", "compare-base");
    tunedBubble = addBubble(buildCompareHtml("tuned", ""), "bot", "compare-finetuned");

    const answerTyping = addTyping("Writing answers…");

    const { res: ansRes, data: ansData } = await postJson("/chat/finetuned", {
      ...payload,
      with_answers: true,
    });
    answerTyping.remove();

    if (!ansRes.ok) {
      const msg =
        typeof ansData.detail === "string"
          ? ansData.detail
          : JSON.stringify(ansData.detail || "Answer generation failed.");
      addBubble(escapeHtml(msg), "bot");
      return;
    }

    const baseText = (ansData.base_bge?.answer || "").trim() || "No answer returned.";
    const tunedText = (ansData.finetuned_bge?.answer || "").trim() || "No answer returned.";

    baseBubble.innerHTML = buildCompareHtml("base", baseText);
    tunedBubble.innerHTML = buildCompareHtml("tuned", tunedText);
    pushHistory(question, ansData.base_bge?.answer, ansData.finetuned_bge?.answer);
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
  els.sendBtn.disabled = sending || !ready || !els.questionInput.value.trim();
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
    const ft = await fetchFinetunedStatus();
    updateFinetunedUi(ft);
    if (!ft.ready) startFinetunedPolling();
  } catch (_) {
    els.statusDot.dataset.status = "error";
    els.docStatus.textContent = "Offline";
  }
})();
