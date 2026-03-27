const API = "http://localhost:8000";
const SESSION_ID = crypto.randomUUID();

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
  el.textContent = "Uploading..."; el.className = "status-msg loading";
  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    el.textContent = `${data.chunks_stored} chunks stored from ${data.file}`;
    el.className = "status-msg success";
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
          <div class="result-meta">#${i+1} &middot; score: ${r.score} &middot; <a href="${r.url}" target="_blank">${r.url}</a></div>
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
        <div class="result-meta">#${i+1} &middot; score: ${r.score} &middot; chunk: ${r.chunk_index} &middot; <a href="${r.url}" target="_blank">${r.url}</a></div>
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
    const saveBtn = webUrls.length ? `
      <div class="save-prompt" style="margin-top: 12px;">
        <span style="font-size: 13px; color: #ccc;">Save these web sources to DB?</span>
        <button class="btn-agent" style="padding: 6px 14px; font-size: 12px; margin-left: 8px;" onclick="saveSourcesToDB(this)">Yes</button>
        <button class="btn-secondary" style="padding: 6px 14px; font-size: 12px; margin-left: 4px;" onclick="this.parentElement.innerHTML='<span style=\\'font-size:13px;color:#888;\\'>Skipped.</span>'">No</button>
      </div>` : "";
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
    const color = ref.sufficient ? "#22c55e" : "#f59e0b";
    return `
      <div style="margin: 8px 0; padding: 8px; background: #1a1a2e; border-left: 3px solid ${color}; border-radius: 4px;">
        <strong style="color:${color};">Reflection:</strong>
        <span style="color:#ccc;"> ${ref.reason}</span>
        ${ref.suggested_query ? `<div style="margin-top:4px;font-size:12px;color:#888;">Next query: "${ref.suggested_query}"</div>` : ""}
        <div style="margin-top:4px;font-size:12px;color:#888;">Decision: ${ref.next_action} | Sufficient: ${ref.sufficient}</div>
      </div>`;
  } catch (e) {
    return reflectionText ? `<div style="margin:8px 0;padding:8px;background:#1a1a2e;border-radius:4px;color:#ccc;font-size:13px;">${reflectionText}</div>` : "";
  }
}

function renderAgentThinking(data, el) {
  const toolsSummary = data.tools_called.map(t =>
    `<span style="background:#2d2d2d;padding:2px 8px;border-radius:4px;font-size:12px;">${t.name}(${JSON.stringify(t.args)})</span>`
  ).join(" ");
  const reflectionHtml = data.reflection ? renderReflection(data.reflection) : "";
  el.innerHTML = `
    <div class="answer-card" style="border-left: 3px solid #f59e0b;">
      <strong>Agent Round ${data.round}</strong>
      <div style="margin: 8px 0;">Tools called: ${toolsSummary}</div>
      ${reflectionHtml}
    </div>
    ${data.sources.length ? renderSources(data.sources) : ""}
    <div class="save-prompt" style="margin-top: 12px;">
      <span style="font-size: 13px; color: #ccc;">Continue or stop here?</span>
      <button class="btn-agent" style="padding: 6px 14px; font-size: 12px; margin-left: 8px;" onclick="agentContinue()">Continue</button>
      <button class="btn-secondary" style="padding: 6px 14px; font-size: 12px; margin-left: 4px;" onclick="agentStop()">Stop & Answer</button>
    </div>`;
}

function renderAgentDone(data, el) {
  const webUrls = data.sources.filter(s => s.url && s.url.startsWith("http"));
  const saveBtn = webUrls.length ? `
    <div class="save-prompt" style="margin-top: 12px;">
      <span style="font-size: 13px; color: #ccc;">Save these web sources to DB?</span>
      <button class="btn-agent" style="padding: 6px 14px; font-size: 12px; margin-left: 8px;" onclick="saveSourcesToDB(this)">Yes</button>
      <button class="btn-secondary" style="padding: 6px 14px; font-size: 12px; margin-left: 4px;" onclick="this.parentElement.innerHTML='<span style=\\'font-size:13px;color:#888;\\'>Skipped.</span>'">No</button>
    </div>` : "";
  window._pendingSources = webUrls;
  el.innerHTML = `
    <div class="answer-card" style="border-left: 3px solid #f59e0b;">
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
  const toolsSummary = data.tools_called.map(t =>
    `<span style="background:#2d2d2d;padding:2px 8px;border-radius:4px;font-size:12px;">${t.name}(${JSON.stringify(t.args)})</span>`
  ).join(" ");

  const reflectionHtml = data.reflection ? renderReflection(data.reflection) : "";

  el.innerHTML = `
    <div class="answer-card" style="border-left: 3px solid #7c3aed;">
      <strong>Agent v2 — Round ${data.round}</strong>
      <div style="margin: 8px 0;">Tools called: ${toolsSummary}</div>
      ${reflectionHtml}
    </div>
    ${data.sources.length ? renderSources(data.sources) : ""}
    <div class="save-prompt" style="margin-top: 12px;">
      <span style="font-size: 13px; color: #ccc;">Continue or stop here?</span>
      <button class="btn-agent" style="padding: 6px 14px; font-size: 12px; margin-left: 8px; background:#7c3aed;" onclick="agentV2Continue()">Continue</button>
      <button class="btn-secondary" style="padding: 6px 14px; font-size: 12px; margin-left: 4px;" onclick="agentV2Stop()">Stop & Answer</button>
    </div>`;
}

function renderAgentV2Done(data, el) {
  const webUrls = data.sources.filter(s => s.url && s.url.startsWith("http"));
  const saveBtn = webUrls.length ? `
    <div class="save-prompt" style="margin-top: 12px;">
      <span style="font-size: 13px; color: #ccc;">Save these web sources to DB?</span>
      <button class="btn-agent" style="padding: 6px 14px; font-size: 12px; margin-left: 8px;" onclick="saveSourcesToDB(this)">Yes</button>
      <button class="btn-secondary" style="padding: 6px 14px; font-size: 12px; margin-left: 4px;" onclick="this.parentElement.innerHTML='<span style=\\'font-size:13px;color:#888;\\'>Skipped.</span>'">No</button>
    </div>` : "";
  window._pendingSources = webUrls;
  el.innerHTML = `
    <div class="answer-card" style="border-left: 3px solid #7c3aed;">
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
