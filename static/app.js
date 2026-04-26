const API = "http://localhost:8000";
const SESSION_ID = crypto.randomUUID();

function renderDecisionPrompt(label, confirmText, confirmClass, confirmAction, cancelAction) {
  return `
    <div class="save-prompt">
      <span class="prompt-label">${label}</span>
      <button class="${confirmClass}" onclick="${confirmAction}">${confirmText}</button>
      <button class="btn-secondary" onclick="${cancelAction}">No</button>
    </div>`;
}

function renderToolChips(toolsCalled) {
  return toolsCalled.map(t =>
    `<span class="tool-chip">${t.name}(${JSON.stringify(t.args)})</span>`
  ).join(" ");
}

async function clearMemory() {
  await fetch(`${API}/memory/${SESSION_ID}`, { method: "DELETE" });
  document.getElementById("memory-status").textContent = "Memory cleared.";
  setTimeout(() => document.getElementById("memory-status").textContent = "", 2000);
}

async function saveSourcesToDB(btn) {
  const container = btn.parentElement;
  const urls = (window._pendingSources || []).map(s => s.url);
  if (!urls.length) return;
  container.innerHTML = `<span class="status-msg loading">Ingesting ${urls.length} URLs...</span>`;
  let saved = 0;
  for (const url of urls) {
    try {
      await fetch(`${API}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      saved++;
    } catch (e) { /* skip failed */ }
  }
  container.innerHTML = `<span class="status-msg success">Saved ${saved} of ${urls.length} sources to DB.</span>`;
}

async function uploadPdf() {
  const fileInput = document.getElementById("pdf-file");
  const el = document.getElementById("upload-result");
  if (!fileInput.files.length) { el.textContent = "Please select a PDF."; el.className = "status-msg error"; return; }
  el.textContent = "Uploading and extracting text, tables, and visuals..."; el.className = "status-msg loading";
  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    el.textContent = `${data.chunks_stored || 0} text chunks, ${data.tables_stored || 0} tables, ${data.visuals_stored || 0} visual summaries stored from ${data.file}`;
    el.className = "status-msg success";
  } catch (e) {
    el.textContent = `Error: ${e.message}`; el.className = "status-msg error";
  }
}

async function uploadAudio() {
  const fileInput = document.getElementById("audio-file");
  const langSelect = document.getElementById("audio-lang");
  const el = document.getElementById("audio-result");
  if (!fileInput.files.length) { el.textContent = "Please select an audio file."; el.className = "status-msg error"; return; }
  const file = fileInput.files[0];
  const sizeMb = (file.size / 1024 / 1024).toFixed(1);
  el.textContent = `Transcribing ${file.name} (${sizeMb} MB) — this may take 30s to a few minutes...`;
  el.className = "status-msg loading";
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API}/ingest-audio?language=${langSelect.value}`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Transcription failed");
    const dur = Math.round(data.audio_duration || 0);
    const mins = Math.floor(dur / 60), secs = dur % 60;
    el.innerHTML = `
      <div class="status-msg success">
        ${data.chunks_stored} chunks stored from ${data.source}
        (${mins}m${secs}s audio, ${data.language})
      </div>
      <details style="margin-top:8px;">
        <summary>Transcript preview</summary>
        <div class="result-content" style="margin-top:6px;">${data.preview || "(empty)"}</div>
      </details>`;
  } catch (e) {
    el.textContent = `Error: ${e.message}`; el.className = "status-msg error";
  }
}

async function ingest() {
  const url = document.getElementById("ingest-url").value.trim();
  const el = document.getElementById("ingest-result");
  el.textContent = "Ingesting..."; el.className = "status-msg loading";
  try {
    const res = await fetch(`${API}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    el.textContent = `${data.chunks_stored} chunks stored from ${data.url}`;
    el.className = "status-msg success";
  } catch (e) {
    el.textContent = `Error: ${e.message}`; el.className = "status-msg error";
  }
}

function getQueryParams() {
  const q = document.getElementById("query-input").value.trim();
  const nInput = document.getElementById("n-results");
  const raw = parseInt(nInput.value);
  const el = document.getElementById("query-results");
  if (isNaN(raw) || raw < 1 || raw > 20) {
    nInput.value = Math.min(Math.max(raw || 3, 1), 20);
    el.innerHTML = `<div class="status-msg error">Please enter a number between 1 and 20.</div>`;
    return null;
  }
  return { q, n: raw, el };
}

function renderSources(sources) {
  return `
    <details>
      <summary>View ${sources.length} sources</summary>
      ${sources.map((r, i) => `
        <div class="result-card">
          <div class="result-meta">#${i+1} &middot; score: ${r.score} &middot; type: ${r.content_type || "text"}${r.page ? ` &middot; page: ${r.page}` : ""} &middot; <a href="${r.url}" target="_blank">${r.url}</a></div>
          <div class="result-content">${r.content}</div>
        </div>
      `).join("")}
    </details>`;
}

async function query() {
  const p = getQueryParams(); if (!p) return;
  p.el.innerHTML = `<div class="status-msg loading">Searching...</div>`;
  try {
    const res = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: p.q, n_results: p.n }),
    });
    const results = await res.json();
    p.el.innerHTML = results.map((r, i) => `
      <div class="result-card">
        <div class="result-meta">#${i+1} &middot; score: ${r.score} &middot; type: ${r.content_type || "text"}${r.chunk_index !== undefined ? ` &middot; chunk: ${r.chunk_index}` : ""}${r.page ? ` &middot; page: ${r.page}` : ""} &middot; <a href="${r.url}" target="_blank">${r.url}</a></div>
        <div class="result-content">${r.content}</div>
      </div>
    `).join("");
  } catch (e) {
    p.el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

async function answer() {
  const p = getQueryParams(); if (!p) return;
  p.el.innerHTML = `<div class="status-msg loading">Generating answer from DB...</div>`;
  try {
    const res = await fetch(`${API}/ask-db`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: p.q, n_results: p.n }),
    });
    const data = await res.json();
    p.el.innerHTML = `
      <div class="answer-card db">
        <strong>AI Answer (DB)</strong>
        <div>${marked.parse(data.answer)}</div>
      </div>
      ${renderSources(data.sources)}`;
  } catch (e) {
    p.el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

async function webSearch() {
  const p = getQueryParams(); if (!p) return;
  p.el.innerHTML = `<div class="status-msg loading">Searching the web...</div>`;
  try {
    const res = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: p.q, n_results: p.n }),
    });
    const data = await res.json();
    const webUrls = data.sources.filter(s => s.url && s.url.startsWith("http"));
    const saveBtn = webUrls.length
      ? renderDecisionPrompt(
          "Save these web sources to DB?",
          "Yes",
          "btn-agent",
          "saveSourcesToDB(this)",
          "this.parentElement.innerHTML='<span class=\\'prompt-note\\'>Skipped.</span>'",
        )
      : "";
    window._pendingSources = webUrls;
    p.el.innerHTML = `
      <div class="answer-card web">
        <strong>AI Answer (Web)</strong>
        <div>${marked.parse(data.answer)}</div>
      </div>
      ${renderSources(data.sources)}
      ${saveBtn}`;
  } catch (e) {
    p.el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

// ── Agent with Human-in-the-loop ──
// Agent 每執行完一輪工具就暫停，顯示中間結果
// 用戶按 Continue 讓 Agent 繼續，按 Stop 強制生成最終答案

function renderReflection(reflectionText) {
  try {
    const ref = JSON.parse(reflectionText);
    const boxClass = ref.sufficient ? "reflection-box success" : "reflection-box";
    return `
      <div class="${boxClass}">
        <strong class="reflection-title">Reflection:</strong>
        <span>${ref.reason}</span>
        ${ref.suggested_query ? `<div class="reflection-meta">Next query: "${ref.suggested_query}"</div>` : ""}
        <div class="reflection-meta">Decision: ${ref.next_action} | Sufficient: ${ref.sufficient}</div>
      </div>`;
  } catch (e) {
    return reflectionText ? `<div class="reflection-box"><span>${reflectionText}</span></div>` : "";
  }
}

function renderAgentThinking(data, el) {
  const toolsSummary = renderToolChips(data.tools_called);
  const reflectionHtml = data.reflection ? renderReflection(data.reflection) : "";
  el.innerHTML = `
    <div class="answer-card agent">
      <strong>Agent Round ${data.round}</strong>
      <div class="tool-row">Tools called: ${toolsSummary}</div>
      ${reflectionHtml}
    </div>
    ${data.sources.length ? renderSources(data.sources) : ""}
    ${renderDecisionPrompt("Continue or stop here?", "Continue", "btn-agent", "agentContinue()", "agentStop()")}`;
}

function renderAgentDone(data, el) {
  const webUrls = data.sources.filter(s => s.url && s.url.startsWith("http"));
  const saveBtn = webUrls.length
    ? renderDecisionPrompt(
        "Save these web sources to DB?",
        "Yes",
        "btn-agent",
        "saveSourcesToDB(this)",
        "this.parentElement.innerHTML='<span class=\\'prompt-note\\'>Skipped.</span>'",
      )
    : "";
  window._pendingSources = webUrls;
  el.innerHTML = `
    <div class="answer-card agent">
      <strong>Agent Answer</strong>
      <div>${marked.parse(data.answer)}</div>
    </div>
    ${data.sources.length ? renderSources(data.sources) : ""}
    ${saveBtn}`;
}

async function agent() {
  const p = getQueryParams(); if (!p) return;
  p.el.innerHTML = `<div class="status-msg loading">Agent is thinking...</div>`;
  try {
    const res = await fetch(`${API}/agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: p.q, n_results: p.n, session_id: SESSION_ID }),
    });
    const data = await res.json();
    if (data.status === "thinking") {
      renderAgentThinking(data, p.el);
    } else {
      renderAgentDone(data, p.el);
    }
  } catch (e) {
    p.el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

async function agentContinue() {
  const el = document.getElementById("query-results");
  el.innerHTML = `<div class="status-msg loading">Agent is continuing...</div>`;
  try {
    const res = await fetch(`${API}/agent/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
    const data = await res.json();
    if (data.status === "thinking") {
      renderAgentThinking(data, el);
    } else {
      renderAgentDone(data, el);
    }
  } catch (e) {
    el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

// ── Agent v2 (LangGraph with Reflection) ──

function renderAgentV2Thinking(data, el) {
  const toolsSummary = renderToolChips(data.tools_called);
  const reflectionHtml = data.reflection ? renderReflection(data.reflection) : "";
  el.innerHTML = `
    <div class="answer-card v2">
      <strong>Agent v2 — Round ${data.round}</strong>
      <div class="tool-row">Tools called: ${toolsSummary}</div>
      ${reflectionHtml}
    </div>
    ${data.sources.length ? renderSources(data.sources) : ""}
    ${renderDecisionPrompt("Continue or stop here?", "Continue", "btn-v2", "agentV2Continue()", "agentV2Stop()")}`;
}

function renderAgentV2Done(data, el) {
  const webUrls = data.sources.filter(s => s.url && s.url.startsWith("http"));
  const saveBtn = webUrls.length
    ? renderDecisionPrompt(
        "Save these web sources to DB?",
        "Yes",
        "btn-agent",
        "saveSourcesToDB(this)",
        "this.parentElement.innerHTML='<span class=\\'prompt-note\\'>Skipped.</span>'",
      )
    : "";
  window._pendingSources = webUrls;
  el.innerHTML = `
    <div class="answer-card v2">
      <strong>Agent v2 Answer</strong>
      <div>${marked.parse(data.answer)}</div>
    </div>
    ${data.sources.length ? renderSources(data.sources) : ""}
    ${saveBtn}`;
}

async function agentV2() {
  const p = getQueryParams(); if (!p) return;
  p.el.innerHTML = `<div class="status-msg loading">Agent v2 is thinking (with reflection)...</div>`;
  try {
    const res = await fetch(`${API}/agent-v2`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: p.q, n_results: p.n, session_id: SESSION_ID }),
    });
    const data = await res.json();
    if (data.status === "thinking") {
      renderAgentV2Thinking(data, p.el);
    } else {
      renderAgentV2Done(data, p.el);
    }
  } catch (e) {
    p.el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

async function agentV2Continue() {
  const el = document.getElementById("query-results");
  el.innerHTML = `<div class="status-msg loading">Agent v2 is continuing...</div>`;
  try {
    const res = await fetch(`${API}/agent-v2/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
    const data = await res.json();
    if (data.status === "thinking") {
      renderAgentV2Thinking(data, el);
    } else {
      renderAgentV2Done(data, el);
    }
  } catch (e) {
    el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

async function agentV2Stop() {
  const el = document.getElementById("query-results");
  el.innerHTML = `<div class="status-msg loading">Generating final answer...</div>`;
  try {
    const res = await fetch(`${API}/agent-v2/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
    const data = await res.json();
    renderAgentV2Done(data, el);
  } catch (e) {
    el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

// ── Structured (Vectorless) Query ──

async function queryStructured() {
  const p = getQueryParams(); if (!p) return;
  p.el.innerHTML = `<div class="status-msg loading">Routing query to document sections...</div>`;
  try {
    const res = await fetch(`${API}/query-structured`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: p.q, n_results: p.n, session_id: SESSION_ID }),
    });
    const data = await res.json();

    if (!data.sections || data.sections.length === 0) {
      p.el.innerHTML = `<div class="status-msg">No structured documents indexed. Ingest a document first.</div>`;
      return;
    }

    const sectionsHtml = data.sections.map(s => `
      <div class="answer-card structured">
        <div>
          <strong class="section-path">${s.section_path}</strong>
          ${s.page ? `<span class="page-tag">Page ${s.page}</span>` : ""}
        </div>
        <div>${marked.parse(s.content)}</div>
        <div class="source-tag">Source: ${s.source}</div>
      </div>
    `).join("");

    const tocHtml = data.toc ? `
      <details>
        <summary>Document TOC (${data.toc.length} sections)</summary>
        <ul class="toc-list">
          ${data.toc.map(t => `<li>${t}</li>`).join("")}
        </ul>
      </details>` : "";

    p.el.innerHTML = sectionsHtml + tocHtml;
  } catch (e) {
    p.el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}

async function agentStop() {
  const el = document.getElementById("query-results");
  el.innerHTML = `<div class="status-msg loading">Generating final answer...</div>`;
  try {
    const res = await fetch(`${API}/agent/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
    const data = await res.json();
    renderAgentDone(data, el);
  } catch (e) {
    el.innerHTML = `<div class="status-msg error">Error: ${e.message}</div>`;
  }
}
