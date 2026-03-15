from __future__ import annotations

import json
from textwrap import dedent
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse


def _page() -> str:
    config = json.dumps({"apiBase": "/v1"}, ensure_ascii=False)
    template = """\
<!doctype html><html lang="pl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Czat MG</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1215;--panel:#11181dcc;--line:#fff4e21f;--text:#f4ede1;--muted:#baa996;--gold:#d4b06a;--fire:#ef8c52;--ok:#70c098;--bad:#ea6b57}
*{box-sizing:border-box} body{margin:0;min-height:100vh;color:var(--text);font-family:"Space Grotesk",sans-serif;background:radial-gradient(circle at top left,#ef8c5230,transparent 24%),linear-gradient(180deg,#091014,#10171b 55%,#0b1114)}
.shell{display:grid;grid-template-columns:320px 1fr;min-height:100vh} .side,.main{padding:24px} .side{border-right:1px solid var(--line);background:#080c0ebd}
.brand h1,.hero h2,.msg h3{font-family:"Fraunces",serif} .brand h1{margin:0;font-size:34px} .brand p,.muted{color:var(--muted)}
.card,.chat,.composer{border:1px solid var(--line);background:var(--panel);border-radius:20px;box-shadow:0 22px 52px #0007} .card{padding:16px;margin-top:16px}
.hero{padding:20px 22px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,#d4b06a24,#ef8c5214)} .hero h2{margin:0 0 8px;font-size:28px} .hero p{margin:0;color:var(--muted)}
.label{display:block;margin-bottom:8px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em} .stack{display:grid;gap:12px} .row{display:flex;flex-wrap:wrap;gap:10px}
.chip{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;background:#fff4e20f;border-radius:999px;font-size:13px} .chip::before{content:"";width:8px;height:8px;border-radius:999px;background:var(--muted)} .chip.ok::before{background:var(--ok)} .chip.warn::before{background:var(--fire)} .chip.bad::before{background:var(--bad)}
button,textarea,input{font:inherit} button{border:0;padding:10px 14px;border-radius:999px;cursor:pointer;font-weight:700;background:linear-gradient(135deg,var(--gold),var(--fire));color:#24150d} button.secondary{background:#fff4e20f;color:var(--text);border:1px solid var(--line)} button:disabled{opacity:.45;cursor:not-allowed}
.list{display:grid;gap:10px;max-height:36vh;overflow:auto} .list button{text-align:left;width:100%;background:#fff4e20a;color:var(--text);border:1px solid transparent;border-radius:16px;padding:12px} .list button.active{border-color:#d4b06a8f;background:#d4b06a1f}
.main{display:grid;grid-template-rows:auto 1fr auto;gap:18px} .chat{min-height:0;overflow:hidden} .messages{height:100%;overflow:auto;display:grid;gap:14px;padding:18px;align-content:start}
.msg{max-width:min(900px,94%);border-radius:18px;padding:16px;background:#fff4e20d;border:1px solid var(--line)} .msg.user{justify-self:end;background:#ef8c521a} .msg.pending{border-style:dashed;opacity:.82} .head{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)} .body{margin:0;white-space:pre-wrap;line-height:1.6;word-break:break-word} .meta{margin-top:12px;padding:12px;border-radius:14px;background:#fff4e20a} .meta h4{margin:0 0 8px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em} .meta ul{margin:0;padding-left:18px}
.composer{padding:16px;display:grid;gap:12px} .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px} .full{grid-column:1/-1} textarea,input{width:100%;border-radius:14px;border:1px solid var(--line);background:#fff4e20a;color:var(--text);padding:12px 14px} textarea{min-height:120px;resize:vertical}
.empty{color:var(--muted);border:1px dashed var(--line);border-radius:18px;padding:24px} a{color:var(--gold)} details{border:1px solid var(--line);border-radius:16px;padding:12px;background:#fff4e208} summary{cursor:pointer;color:var(--muted)}
@media (max-width:1100px){.shell{grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid var(--line)}}
@media (max-width:720px){.grid{grid-template-columns:1fr}.side,.main{padding:18px}.msg{max-width:100%}}
</style></head><body>
<div class="shell"><aside class="side">
<div class="brand"><h1>AI Czat MG</h1><p>Jedno wejscie do rozmow, kanonu i zapisow na Drive.</p></div>
<section class="card stack"><span class="label">Konto Google</span><div id="auth" class="chip warn">Nie zalogowano</div><div id="authNote" class="muted">Zaloguj konto Google, aby uzywac czatu i zapisywac na Drive.</div><div class="row"><button id="loginBtn">Polacz konto Google</button><button id="signOut" class="secondary" disabled>Wyloguj</button></div></section>
<section class="card stack"><span class="label">Drive</span><div id="drive" class="chip warn">Brak danych</div><div id="driveEmail" class="muted">Brak aktywnego polaczenia.</div><div class="row"><button id="connectDrive" disabled>Polacz Drive</button><button id="disconnectDrive" class="secondary" disabled>Odlacz</button></div></section>
<section class="card stack"><div class="row"><button id="newConv" disabled>Nowa rozmowa</button><button id="refreshConv" class="secondary" disabled>Odswiez</button></div><span class="label">Rozmowy</span><div id="convList" class="list"></div></section>
</aside><main class="main">
<header class="hero"><h2 id="heroTitle">Konsola MG</h2><p id="heroSub">Napisz normalnym jezykiem. Agent sam wybierze tryb pracy i przed trwala zmiana poprosi o potwierdzenie.</p></header>
<section class="chat"><div id="messages" class="messages"><div class="empty">Brak aktywnej rozmowy.</div></div></section>
<section class="composer">
<label class="full"><span class="label">Wiadomosc</span><textarea id="message" placeholder="Np. Co mamy ciekawego w 1 rozdziale kampanii?"></textarea></label>
<details>
<summary>Opcje dodatkowe</summary>
<div class="grid" style="margin-top:12px">
<label class="full"><span class="label">Dodatkowy tekst do analizy</span><textarea id="candidate" placeholder="Wklej tu tekst, jesli chcesz zeby agent sprawdzil go z kanonem albo przerobil."></textarea></label>
<label><span class="label">Tytul dokumentu</span><input id="outputTitle" type="text" placeholder="Opcjonalna nazwa w Google Docs"></label>
<label style="align-self:end"><span class="label">Zapisz wynik do Drive</span><input id="saveOutput" type="checkbox"></label>
</div>
</details>
<div class="row"><button id="send" disabled>Wyslij</button><button id="clear" class="secondary">Wyczysc</button><div id="stream" class="chip ok">Streaming gotowy</div></div>
<div class="muted">"Tytul dokumentu" ma sens tylko wtedy, gdy zapisujesz wynik. Jesli go nie podasz, agent nada sensowna nazwe sam.</div>
</section>
</main></div>
<script id="cfg" type="application/json">__CONFIG__</script>
<script>
(() => {
  const cfg = JSON.parse(document.getElementById("cfg").textContent);
  const s = { authenticated: false, email: "", convs: [], cid: null, title: "", msgs: [], pending: -1 };
  const byId = (id) => document.getElementById(id);
  const e = {
    auth: byId("auth"),
    authNote: byId("authNote"),
    loginBtn: byId("loginBtn"),
    signOut: byId("signOut"),
    connectDrive: byId("connectDrive"),
    disconnectDrive: byId("disconnectDrive"),
    drive: byId("drive"),
    driveEmail: byId("driveEmail"),
    convList: byId("convList"),
    newConv: byId("newConv"),
    refreshConv: byId("refreshConv"),
    heroTitle: byId("heroTitle"),
    heroSub: byId("heroSub"),
    messages: byId("messages"),
    candidate: byId("candidate"),
    message: byId("message"),
    outputTitle: byId("outputTitle"),
    saveOutput: byId("saveOutput"),
    send: byId("send"),
    clear: byId("clear"),
    stream: byId("stream"),
  };
  const esc = (v) => String(v || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const chip = (n, t, x) => { n.className = `chip ${t}`; n.textContent = x; };
  async function request(path, opt = {}) {
    const headers = new Headers(opt.headers || {});
    if (opt.json !== undefined) headers.set("Content-Type", "application/json; charset=utf-8");
    const response = await fetch(`${cfg.apiBase}${path}`, { method: opt.method || "GET", headers, body: opt.json !== undefined ? JSON.stringify(opt.json) : opt.body, credentials: "same-origin" });
    if (!response.ok) {
      let payload = null;
      try { payload = await response.json(); } catch (_err) { payload = null; }
      const error = new Error(payload?.detail?.message || payload?.message || response.statusText);
      error.status = response.status;
      throw error;
    }
    return response;
  }
  async function loadSession() {
    const response = await request("/auth/session/status");
    const payload = await response.json();
    s.authenticated = Boolean(payload.authenticated);
    s.email = payload.email || "";
    renderAuth();
    if (!s.authenticated) {
      s.cid = null; s.title = ""; s.convs = []; s.msgs = [];
      renderConvs(); renderMsgs(); renderDriveDisconnected();
      return;
    }
    await loadDrive();
    await loadConvs();
  }
  function renderAuth() {
    chip(e.auth, s.authenticated ? "ok" : "warn", s.authenticated ? `Zalogowano: ${s.email || "uzytkownik"}` : "Nie zalogowano");
    e.authNote.textContent = s.authenticated ? "Czat i akcje po lewej sa aktywne." : "Kliknij 'Polacz konto Google', aby wejsc do czatu.";
    e.loginBtn.disabled = false;
    e.signOut.disabled = !s.authenticated;
    e.connectDrive.disabled = !s.authenticated;
    e.newConv.disabled = !s.authenticated;
    e.refreshConv.disabled = !s.authenticated;
    e.send.disabled = !s.authenticated;
  }
  function renderDriveDisconnected() {
    chip(e.drive, "warn", "Drive niepolaczony");
    e.driveEmail.textContent = s.authenticated ? "Zapis wymaga polaczenia Twojego Google Drive." : "Najpierw zaloguj konto Google.";
    e.disconnectDrive.disabled = true;
  }
  async function loadDrive() {
    if (!s.authenticated) { renderDriveDisconnected(); return; }
    try {
      const response = await request("/auth/google-drive/status");
      const payload = await response.json();
      chip(e.drive, payload.connected ? "ok" : "warn", payload.connected ? "Drive polaczony" : "Drive niepolaczony");
      e.driveEmail.textContent = payload.connected ? `Zapis jako: ${payload.subject_email}` : "Mozesz polaczyc Drive ponownie z lewego panelu.";
      e.disconnectDrive.disabled = !payload.connected;
    } catch (err) {
      if (err.status === 401) { s.authenticated = false; renderAuth(); renderDriveDisconnected(); return; }
      chip(e.drive, "bad", "Blad Drive");
      e.driveEmail.textContent = err.message;
    }
  }
  function msgFromResponse(r) { return { role: "assistant", kind: r.kind, title: r.title || "", content: r.reply_markdown || r.reply || "", actions: r.next_actions || [], warnings: r.warnings || [], citations: r.citations || [], continuity: r.continuity || null, output: r.output || null, stream: r.stream_debug || null }; }
  function msgFromRecord(r) { return { role: r.role, kind: r.kind || "", title: r.artifact_type || "", content: r.content || "", actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null }; }
  function renderConvs() {
    if (!s.authenticated) { e.convList.innerHTML = '<div class="empty">Zaloguj konto, aby zobaczyc rozmowy.</div>'; return; }
    if (!s.convs.length) { e.convList.innerHTML = '<div class="empty">Brak zapisanych rozmow.</div>'; return; }
    e.convList.innerHTML = s.convs.map(c => `<button type="button" class="${c.conversation_id === s.cid ? "active" : ""}" data-id="${esc(c.conversation_id)}"><strong>${esc(c.title || "Nowa rozmowa")}</strong><br><span class="muted">${esc(String(c.message_count || 0))} msg</span></button>`).join("");
    e.convList.querySelectorAll("[data-id]").forEach(b => b.addEventListener("click", () => selectConv(b.getAttribute("data-id"))));
  }
  function renderMsgs() {
    if (!s.msgs.length) { e.messages.innerHTML = '<div class="empty">Brak wiadomosci w tej rozmowie.</div>'; return; }
    e.messages.innerHTML = s.msgs.map((m, i) => {
      const meta = [];
      if (m.warnings?.length) meta.push(`<div class="meta"><h4>Warnings</h4><ul>${m.warnings.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`);
      if (m.citations?.length) meta.push(`<div class="meta"><h4>Zrodla</h4><ul>${m.citations.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`);
      if (m.continuity?.issues?.length) meta.push(`<div class="meta"><h4>Continuity</h4><ul>${m.continuity.issues.map(x => `<li>[${esc(x.severity)}] ${esc(x.message)}</li>`).join("")}</ul></div>`);
      if (m.output?.doc_id) meta.push(`<div class="meta"><h4>Output</h4><a target="_blank" rel="noopener noreferrer" href="https://docs.google.com/document/d/${encodeURIComponent(m.output.doc_id)}/edit">${esc(m.output.title || "Otworz dokument")}</a></div>`);
      if (m.stream) meta.push(`<div class="meta"><h4>Streaming</h4><div>${esc(m.stream.selected_mode || "")} / ${esc(m.stream.reason || "")}</div></div>`);
      const actions = (m.actions || []).map(a => `<button class="secondary" type="button" data-action="${esc(a.type)}" data-payload="${encodeURIComponent(JSON.stringify(a.payload || {}))}" data-index="${i}">${esc(a.label)}</button>`).join("");
      return `<article class="msg ${esc(m.role)} ${m.pending ? "pending" : ""}"><div class="head"><span>${esc(m.role === "user" ? "MG" : "Agent")}</span><span>${esc(m.kind || "")}</span></div>${m.title ? `<h3>${esc(m.title)}</h3>` : ""}<pre class="body">${esc(m.content || "")}</pre>${meta.join("")}<div class="row" style="margin-top:12px">${actions}</div></article>`;
    }).join("");
    e.messages.querySelectorAll("[data-action]").forEach(b => b.addEventListener("click", onAction));
    e.messages.scrollTop = e.messages.scrollHeight;
  }
  async function loadConvs() {
    if (!s.authenticated) return;
    const response = await request("/conversations");
    const payload = await response.json();
    s.convs = payload.items || [];
    renderConvs();
    if (s.cid) await loadMsgs(s.cid);
  }
  async function newConversation() {
    const response = await request("/conversations", { method: "POST", json: { title: "Nowa rozmowa" } });
    const payload = await response.json();
    s.cid = payload.conversation.conversation_id;
    s.title = payload.conversation.title || "";
    s.msgs = [];
    e.heroTitle.textContent = s.title || "Nowa rozmowa";
    renderMsgs();
    await loadConvs();
  }
  async function loadMsgs(id) {
    const response = await request(`/conversations/${id}/messages`);
    const payload = await response.json();
    s.msgs = (payload.items || []).map(msgFromRecord);
    s.cid = id;
    const current = s.convs.find(x => x.conversation_id === id);
    s.title = current?.title || "Rozmowa";
    e.heroTitle.textContent = s.title;
    e.heroSub.textContent = "Agent sam wybiera sposob pracy. Trwale operacje wymagaja potwierdzenia.";
    renderConvs();
    renderMsgs();
  }
  async function selectConv(id) { s.cid = id; await loadMsgs(id); }
  async function startGoogleFlow() {
    const response = await request("/auth/google-drive/start", { method: "POST" });
    const payload = await response.json();
    window.location.href = payload.authorization_url;
  }
  async function disconnectDriveFlow() { await request("/auth/google-drive/disconnect", { method: "POST" }); await loadDrive(); }
  async function logoutFlow() {
    await request("/auth/session/logout", { method: "POST" });
    s.authenticated = false; s.email = ""; s.cid = null; s.title = ""; s.convs = []; s.msgs = [];
    renderAuth(); renderConvs(); renderMsgs(); renderDriveDisconnected();
  }
  async function sendMessage() {
    const text = e.message.value.trim();
    const extraText = e.candidate.value.trim();
    if (!text) return;
    const req = { message: text, mode: "auto", stream: true, save_output: e.saveOutput.checked, output_title: e.outputTitle.value.trim() || null };
    if (extraText) req.candidate_text = extraText;
    const endpoint = s.cid ? `/conversations/${s.cid}/messages` : "/chat";
    s.msgs.push({ role: "user", kind: "input", title: "", content: text, actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null });
    s.pending = s.msgs.length;
    s.msgs.push({ role: "assistant", kind: "stream", title: "Agent pisze", content: "", actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null, pending: true });
    renderMsgs();
    e.message.value = "";
    chip(e.stream, "ok", "Streaming odpowiedzi...");
    try {
      const response = await request(endpoint, { method: "POST", json: req });
      await consume(response);
      await loadSession();
      chip(e.stream, "ok", "Streaming gotowy");
    } catch (err) {
      s.msgs[s.pending] = { role: "assistant", kind: "error", title: "Blad", content: err.message, actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null };
      renderMsgs();
      chip(e.stream, "bad", "Blad streamingu");
    }
  }
  async function consume(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const part = await reader.read();
      buffer += decoder.decode(part.value || new Uint8Array(), { stream: !part.done });
      let cut = buffer.indexOf("\\n\\n");
      while (cut !== -1) {
        handle(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        cut = buffer.indexOf("\\n\\n");
      }
      if (part.done) break;
    }
  }
  function handle(raw) {
    if (!raw.trim()) return;
    let type = "message";
    const data = [];
    for (const line of raw.split("\\n")) {
      if (line.startsWith("event:")) type = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    const payload = data.length ? JSON.parse(data.join("\\n")) : {};
    if (type === "start") {
      if (payload.conversation_id) s.cid = payload.conversation_id;
      if (s.pending >= 0) s.msgs[s.pending].stream = payload.stream_debug || null;
      renderMsgs();
      return;
    }
    if (type === "delta") {
      if (s.pending >= 0) s.msgs[s.pending].content += payload.text || "";
      renderMsgs();
      return;
    }
    if (type === "complete" && s.pending >= 0) {
      s.msgs[s.pending] = msgFromResponse(payload);
      s.pending = -1;
      s.cid = payload.conversation_id || s.cid;
      s.title = payload.conversation_title || s.title;
      if (s.title) e.heroTitle.textContent = s.title;
      renderMsgs();
    }
  }
  async function onAction(ev) {
    const type = ev.currentTarget.getAttribute("data-action");
    const payload = JSON.parse(decodeURIComponent(ev.currentTarget.getAttribute("data-payload") || "%7B%7D"));
    const idx = Number(ev.currentTarget.getAttribute("data-index"));
    if (type === "accept_world_change" || type === "reject_world_change" || type === "confirm_inferred_action") {
      const response = await request("/assistant/actions", { method: "POST", json: { action_type: type, ...payload } });
      const result = await response.json();
      if (result.chat) {
        s.msgs.push(msgFromResponse(result.chat));
      } else {
        s.msgs.push({ role: "assistant", kind: "answer", title: type === "accept_world_change" ? "Zmiana zaakceptowana" : type === "reject_world_change" ? "Zmiana odrzucona" : "Potwierdzone", content: result.summary || "Akcja wykonana.", actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null });
      }
      renderMsgs();
      await loadSession();
      return;
    }
    if (type === "revise") { e.message.focus(); return; }
    if (type === "review_continuity") {
      const message = s.msgs[idx];
      if (message?.continuity?.issues?.length) alert(message.continuity.issues.map(x => `[${x.severity}] ${x.message}`).join("\\n"));
      return;
    }
    if (type === "open_output_doc" && payload.doc_id) window.open(`https://docs.google.com/document/d/${encodeURIComponent(payload.doc_id)}/edit`, "_blank", "noopener");
  }
  e.loginBtn.addEventListener("click", startGoogleFlow);
  e.signOut.addEventListener("click", logoutFlow);
  e.connectDrive.addEventListener("click", startGoogleFlow);
  e.disconnectDrive.addEventListener("click", disconnectDriveFlow);
  e.newConv.addEventListener("click", newConversation);
  e.refreshConv.addEventListener("click", loadConvs);
  e.send.addEventListener("click", sendMessage);
  e.clear.addEventListener("click", () => { e.message.value = ""; e.candidate.value = ""; e.outputTitle.value = ""; e.saveOutput.checked = false; });
  renderAuth();
  renderConvs();
  renderMsgs();
  renderDriveDisconnected();
  loadSession().catch(err => { e.heroSub.textContent = err.message; });
})();
</script></body></html>
"""
    return dedent(template).replace("__CONFIG__", config)


def build_web_router(*, google_client_id: Optional[str] = None) -> APIRouter:
    router = APIRouter(tags=["web"])

    @router.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/gm", status_code=307)

    @router.get("/gm", response_class=HTMLResponse, include_in_schema=False)
    def gm_console():
        return HTMLResponse(_page())

    return router
