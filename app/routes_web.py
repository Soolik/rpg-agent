from __future__ import annotations

import json
from html import escape
from textwrap import dedent
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse


def _page(client_id: Optional[str]) -> str:
    config = escape(json.dumps({"apiBase": "/v1", "googleClientId": client_id or ""}, ensure_ascii=False))
    note = escape("Zaloguj się przez Google, żeby używać czatu." if client_id else "Brak GOOGLE_OAUTH_CLIENT_ID.")
    template = """\
<!doctype html><html lang="pl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Czat MG</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://accounts.google.com/gsi/client" async defer></script>
<style>
:root{--bg:#0d1215;--panel:#11181dcc;--line:#fff4e21f;--text:#f4ede1;--muted:#baa996;--gold:#d4b06a;--fire:#ef8c52;--ok:#70c098;--bad:#ea6b57}
*{box-sizing:border-box} body{margin:0;min-height:100vh;color:var(--text);font-family:"Space Grotesk",sans-serif;background:radial-gradient(circle at top left,#ef8c5230,transparent 24%),linear-gradient(180deg,#091014,#10171b 55%,#0b1114)}
.shell{display:grid;grid-template-columns:320px 1fr;min-height:100vh} .side,.main{padding:24px} .side{border-right:1px solid var(--line);background:#080c0ebd}
.brand h1,.hero h2,.msg h3{font-family:"Fraunces",serif} .brand h1{margin:0;font-size:34px} .brand p,.muted{color:var(--muted)}
.card,.chat,.composer{border:1px solid var(--line);background:var(--panel);border-radius:20px;box-shadow:0 22px 52px #0007} .card{padding:16px;margin-top:16px}
.hero{padding:20px 22px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,#d4b06a24,#ef8c5214)} .hero h2{margin:0 0 8px;font-size:28px} .hero p{margin:0;color:var(--muted)}
.label{display:block;margin-bottom:8px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em} .stack{display:grid;gap:12px} .row{display:flex;flex-wrap:wrap;gap:10px}
.chip{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;background:#fff4e20f;border-radius:999px;font-size:13px} .chip::before{content:"";width:8px;height:8px;border-radius:999px;background:var(--muted)} .chip.ok::before{background:var(--ok)} .chip.warn::before{background:var(--fire)} .chip.bad::before{background:var(--bad)}
button,select,textarea,input{font:inherit} button{border:0;padding:10px 14px;border-radius:999px;cursor:pointer;font-weight:700;background:linear-gradient(135deg,var(--gold),var(--fire));color:#24150d} button.secondary{background:#fff4e20f;color:var(--text);border:1px solid var(--line)} button:disabled{opacity:.45;cursor:not-allowed}
.list{display:grid;gap:10px;max-height:36vh;overflow:auto} .list button{text-align:left;width:100%;background:#fff4e20a;color:var(--text);border:1px solid transparent;border-radius:16px;padding:12px} .list button.active{border-color:#d4b06a8f;background:#d4b06a1f}
.main{display:grid;grid-template-rows:auto 1fr auto;gap:18px} .chat{min-height:0;overflow:hidden} .messages{height:100%;overflow:auto;display:grid;gap:14px;padding:18px;align-content:start}
.msg{max-width:min(900px,94%);border-radius:18px;padding:16px;background:#fff4e20d;border:1px solid var(--line)} .msg.user{justify-self:end;background:#ef8c521a} .msg.pending{border-style:dashed;opacity:.82} .head{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)} .body{margin:0;white-space:pre-wrap;line-height:1.6;word-break:break-word} .meta{margin-top:12px;padding:12px;border-radius:14px;background:#fff4e20a} .meta h4{margin:0 0 8px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em} .meta ul{margin:0;padding-left:18px}
.composer{padding:16px;display:grid;gap:12px} .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px} .full{grid-column:1/-1} textarea,input,select{width:100%;border-radius:14px;border:1px solid var(--line);background:#fff4e20a;color:var(--text);padding:12px 14px} textarea{min-height:120px;resize:vertical}
.empty{color:var(--muted);border:1px dashed var(--line);border-radius:18px;padding:24px} a{color:var(--gold)}
@media (max-width:1100px){.shell{grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid var(--line)}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:720px){.grid{grid-template-columns:1fr}.side,.main{padding:18px}.msg{max-width:100%}}
</style></head><body>
<div class="shell"><aside class="side">
<div class="brand"><h1>AI Czat MG</h1><p>Rozmowy, continuity i workflow zmian kanonu w jednym panelu.</p></div>
<section class="card stack"><span class="label">Logowanie</span><div id="auth" class="chip warn">Nie zalogowano</div><div id="googleMount"></div><div class="row"><button id="signOut" class="secondary" disabled>Wyloguj</button></div><div id="loginNote">__NOTE__</div></section>
<section class="card stack"><span class="label">Drive</span><div id="drive" class="chip warn">Brak danych</div><div id="driveEmail" class="muted">Brak aktywnego połączenia.</div><div class="row"><button id="connectDrive" disabled>Podłącz Drive</button><button id="disconnectDrive" class="secondary" disabled>Odłącz</button></div></section>
<section class="card stack"><div class="row"><button id="newConv" disabled>Nowa rozmowa</button><button id="refreshConv" class="secondary" disabled>Odśwież</button></div><span class="label">Rozmowy</span><div id="convList" class="list"></div></section>
</aside><main class="main">
<header class="hero"><h2 id="heroTitle">Konsola MG</h2><p id="heroSub">Zaloguj się i zacznij rozmowę z agentem.</p></header>
<section class="chat"><div id="messages" class="messages"><div class="empty">Brak aktywnej rozmowy.</div></div></section>
<section class="composer"><div class="grid">
<label><span class="label">Tryb</span><select id="mode"><option value="create">create</option><option value="guard">guard</option><option value="editor">editor</option></select></label>
<label id="artifactWrap"><span class="label">Artefakt</span><select id="artifact"><option value="">brak</option><option value="session_hooks">session_hooks</option><option value="npc_brief">npc_brief</option><option value="pre_session_brief">pre_session_brief</option><option value="gm_brief">gm_brief</option><option value="scene_seed">scene_seed</option><option value="twist_pack">twist_pack</option><option value="player_summary">player_summary</option><option value="session_report">session_report</option></select></label>
<label><span class="label">Tytuł outputu</span><input id="outputTitle" type="text" placeholder="Opcjonalny tytuł"></label>
<label style="align-self:end"><span class="label">Zapis</span><input id="saveOutput" type="checkbox"></label>
<label id="candidateWrap" class="full" hidden><span class="label">Tekst do guarda</span><textarea id="candidate" placeholder="Wklej tekst do sprawdzenia continuity."></textarea></label>
<label class="full"><span class="label">Wiadomość</span><textarea id="message" placeholder="Np. Przygotuj 3 hooki na kolejną sesję o Red Blade i Captain Mira."></textarea></label>
</div><div class="row"><button id="send" disabled>Wyślij</button><button id="clear" class="secondary">Wyczyść</button><div id="stream" class="chip ok">Streaming gotowy</div></div></section>
</main></div>
<script id="cfg" type="application/json">__CONFIG__</script>
<script>
(() => {
  const cfg = JSON.parse(document.getElementById("cfg").textContent);
  const s = { token: localStorage.getItem("gm_id_token") || "", email: "", convs: [], cid: null, title: "", msgs: [], pending: -1 };
  const e = { auth: auth, signOut: signOut, connectDrive: connectDrive, disconnectDrive: disconnectDrive, drive: drive, driveEmail: driveEmail, convList: convList, newConv: newConv, refreshConv: refreshConv, heroTitle: heroTitle, heroSub: heroSub, messages: messages, mode: mode, artifactWrap: artifactWrap, artifact: artifact, candidateWrap: candidateWrap, candidate: candidate, message: message, outputTitle: outputTitle, saveOutput: saveOutput, send: send, clear: clear, stream: stream };
  const esc = (v) => String(v || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const chip = (n, t, x) => { n.className = `chip ${t}`; n.textContent = x; };
  const jwt = (t) => { try { return JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))); } catch (_err) { return null; } };
  function setToken(t) { s.token = t || ""; if (s.token) { localStorage.setItem("gm_id_token", s.token); s.email = jwt(s.token)?.email || ""; } else { localStorage.removeItem("gm_id_token"); s.email = ""; } renderAuth(); }
  function renderAuth() { const ok = Boolean(s.token); chip(e.auth, ok ? "ok" : "warn", ok ? `Zalogowano: ${s.email || "użytkownik"}` : "Nie zalogowano"); e.signOut.disabled = !ok; e.connectDrive.disabled = !ok; e.newConv.disabled = !ok; e.refreshConv.disabled = !ok; e.send.disabled = !ok; }
  async function api(path, opt = {}) { if (!s.token) throw new Error("Zaloguj się przez Google."); const h = new Headers(opt.headers || {}); h.set("Authorization", `Bearer ${s.token}`); if (opt.json !== undefined) h.set("Content-Type", "application/json; charset=utf-8"); const r = await fetch(`${cfg.apiBase}${path}`, { method: opt.method || "GET", headers: h, body: opt.json !== undefined ? JSON.stringify(opt.json) : opt.body }); if (!r.ok) { let p = null; try { p = await r.json(); } catch (_err) { p = null; } if (r.status === 401) setToken(""); throw new Error(p?.detail?.message || p?.message || r.statusText); } return r; }
  function msgFromResponse(r) { return { role: "assistant", kind: r.kind, title: r.title || "", content: r.reply_markdown || r.reply || "", actions: r.next_actions || [], warnings: r.warnings || [], citations: r.citations || [], continuity: r.continuity || null, output: r.output || null, stream: r.stream_debug || null }; }
  function msgFromRecord(r) { return { role: r.role, kind: r.kind || "", title: r.artifact_type || "", content: r.content || "", actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null }; }
  function renderConvs() { if (!s.convs.length) { e.convList.innerHTML = '<div class="empty">Brak zapisanych rozmów.</div>'; return; } e.convList.innerHTML = s.convs.map(c => `<button type="button" class="${c.conversation_id === s.cid ? "active" : ""}" data-id="${esc(c.conversation_id)}"><strong>${esc(c.title || "Nowa rozmowa")}</strong><br><span class="muted">${esc(String(c.message_count || 0))} msg</span></button>`).join(""); e.convList.querySelectorAll("[data-id]").forEach(b => b.addEventListener("click", () => selectConv(b.getAttribute("data-id")))); }
  function renderMsgs() { if (!s.msgs.length) { e.messages.innerHTML = '<div class="empty">Brak wiadomości w tej rozmowie.</div>'; return; } e.messages.innerHTML = s.msgs.map((m, i) => { const meta = []; if (m.warnings?.length) meta.push(`<div class="meta"><h4>Warnings</h4><ul>${m.warnings.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`); if (m.citations?.length) meta.push(`<div class="meta"><h4>Źródła</h4><ul>${m.citations.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`); if (m.continuity?.issues?.length) meta.push(`<div class="meta"><h4>Continuity</h4><ul>${m.continuity.issues.map(x => `<li>[${esc(x.severity)}] ${esc(x.message)}</li>`).join("")}</ul></div>`); if (m.output?.doc_id) meta.push(`<div class="meta"><h4>Output</h4><a target="_blank" rel="noopener noreferrer" href="https://docs.google.com/document/d/${encodeURIComponent(m.output.doc_id)}/edit">${esc(m.output.title || "Otwórz dokument")}</a></div>`); if (m.stream) meta.push(`<div class="meta"><h4>Streaming</h4><div>${esc(m.stream.selected_mode || "")} / ${esc(m.stream.reason || "")}</div></div>`); const actions = (m.actions || []).map(a => `<button class="secondary" type="button" data-action="${esc(a.type)}" data-payload="${encodeURIComponent(JSON.stringify(a.payload || {}))}" data-index="${i}">${esc(a.label)}</button>`).join(""); return `<article class="msg ${esc(m.role)} ${m.pending ? "pending" : ""}"><div class="head"><span>${esc(m.role === "user" ? "MG" : "Agent")}</span><span>${esc(m.kind || "")}</span></div>${m.title ? `<h3>${esc(m.title)}</h3>` : ""}<pre class="body">${esc(m.content || "")}</pre>${meta.join("")}<div class="row" style="margin-top:12px">${actions}</div></article>`; }).join(""); e.messages.querySelectorAll("[data-action]").forEach(b => b.addEventListener("click", onAction)); e.messages.scrollTop = e.messages.scrollHeight; }
  function updateMode() { const g = e.mode.value === "guard"; e.candidateWrap.hidden = !g; e.artifactWrap.hidden = g; }
  async function loadDrive() { if (!s.token) { chip(e.drive, "warn", "Wymaga logowania"); e.driveEmail.textContent = "Zaloguj się, aby sprawdzić status Drive."; e.disconnectDrive.disabled = true; return; } const r = await api("/auth/google-drive/status"); const p = await r.json(); chip(e.drive, p.connected ? "ok" : "warn", p.connected ? "Drive połączony" : "Drive niepołączony"); e.driveEmail.textContent = p.connected ? `Zapis jako: ${p.subject_email}` : "Zapis wymaga podłączenia Twojego Drive."; e.disconnectDrive.disabled = !p.connected; }
  async function loadConvs() { if (!s.token) return; const r = await api("/conversations"); const p = await r.json(); s.convs = p.items || []; renderConvs(); if (s.cid) await loadMsgs(s.cid); }
  async function newConversation() { const r = await api("/conversations", { method: "POST", json: { title: "Nowa rozmowa" } }); const p = await r.json(); s.cid = p.conversation.conversation_id; s.title = p.conversation.title || ""; s.msgs = []; renderMsgs(); await loadConvs(); }
  async function loadMsgs(id) { const r = await api(`/conversations/${id}/messages`); const p = await r.json(); s.msgs = (p.items || []).map(msgFromRecord); s.cid = id; const hit = s.convs.find(x => x.conversation_id === id); s.title = hit?.title || "Rozmowa"; e.heroTitle.textContent = s.title; e.heroSub.textContent = "Ta rozmowa korzysta z pamięci konwersacji i tego samego kanonu."; renderConvs(); renderMsgs(); }
  async function selectConv(id) { s.cid = id; await loadMsgs(id); }
  async function connectDriveFlow() { const r = await api("/auth/google-drive/start", { method: "POST" }); const p = await r.json(); window.location.href = p.authorization_url; }
  async function disconnectDriveFlow() { await api("/auth/google-drive/disconnect", { method: "POST" }); await loadDrive(); }
  async function sendMessage() { const text = e.message.value.trim(); if (!text) return; const modeValue = e.mode.value; const req = { message: text, mode: modeValue, stream: true, save_output: e.saveOutput.checked, output_title: e.outputTitle.value.trim() || null }; if (modeValue !== "guard" && e.artifact.value) req.artifact_type = e.artifact.value; if (modeValue === "guard" && e.candidate.value.trim()) req.candidate_text = e.candidate.value.trim(); const endpoint = s.cid ? `/conversations/${s.cid}/messages` : "/chat"; s.msgs.push({ role: "user", kind: "input", title: "", content: text, actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null }); s.pending = s.msgs.length; s.msgs.push({ role: "assistant", kind: "stream", title: "Agent pisze", content: "", actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null, pending: true }); renderMsgs(); e.message.value = ""; chip(e.stream, "ok", "Streaming odpowiedzi…"); try { const r = await api(endpoint, { method: "POST", json: req }); await consume(r); await loadConvs(); await loadDrive(); chip(e.stream, "ok", "Streaming gotowy"); } catch (err) { s.msgs[s.pending] = { role: "assistant", kind: "error", title: "Błąd", content: err.message, actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null }; renderMsgs(); chip(e.stream, "bad", "Błąd streamingu"); } }
  async function consume(r) { const reader = r.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; while (true) { const part = await reader.read(); buffer += decoder.decode(part.value || new Uint8Array(), { stream: !part.done }); let cut = buffer.indexOf("\\n\\n"); while (cut !== -1) { handle(buffer.slice(0, cut)); buffer = buffer.slice(cut + 2); cut = buffer.indexOf("\\n\\n"); } if (part.done) break; } }
  function handle(raw) { if (!raw.trim()) return; let type = "message"; const data = []; for (const line of raw.split("\\n")) { if (line.startsWith("event:")) type = line.slice(6).trim(); if (line.startsWith("data:")) data.push(line.slice(5).trim()); } const p = data.length ? JSON.parse(data.join("\\n")) : {}; if (type === "start") { if (p.conversation_id) s.cid = p.conversation_id; if (s.pending >= 0) s.msgs[s.pending].stream = p.stream_debug || null; renderMsgs(); return; } if (type === "delta") { if (s.pending >= 0) s.msgs[s.pending].content += p.text || ""; renderMsgs(); return; } if (type === "complete" && s.pending >= 0) { s.msgs[s.pending] = msgFromResponse(p); s.pending = -1; s.cid = p.conversation_id || s.cid; s.title = p.conversation_title || s.title; renderMsgs(); } }
  async function onAction(ev) { const type = ev.currentTarget.getAttribute("data-action"); const payload = JSON.parse(decodeURIComponent(ev.currentTarget.getAttribute("data-payload") || "%7B%7D")); const idx = Number(ev.currentTarget.getAttribute("data-index")); if (type === "accept_world_change" || type === "reject_world_change") { const r = await api("/assistant/actions", { method: "POST", json: { action_type: type, proposal_id: payload.proposal_id } }); const p = await r.json(); s.msgs.push({ role: "assistant", kind: "answer", title: type === "accept_world_change" ? "Zmiana zaakceptowana" : "Zmiana odrzucona", content: p.summary || "Akcja wykonana.", actions: [], warnings: [], citations: [], continuity: null, output: null, stream: null }); renderMsgs(); return; } if (type === "revise") { e.mode.value = "create"; e.artifact.value = payload.artifact_type || ""; updateMode(); e.message.focus(); return; } if (type === "review_continuity") { const m = s.msgs[idx]; if (m?.continuity?.issues?.length) alert(m.continuity.issues.map(x => `[${x.severity}] ${x.message}`).join("\\n")); return; } if (type === "open_output_doc" && payload.doc_id) window.open(`https://docs.google.com/document/d/${encodeURIComponent(payload.doc_id)}/edit`, "_blank", "noopener"); }
  function setupGoogle() { if (!cfg.googleClientId) return; const init = () => { if (!window.google?.accounts?.id) return window.setTimeout(init, 120); window.google.accounts.id.initialize({ client_id: cfg.googleClientId, callback: async (resp) => { setToken(resp.credential); await loadDrive(); await loadConvs(); } }); window.google.accounts.id.renderButton(document.getElementById("googleMount"), { theme: "filled_black", size: "large", shape: "pill" }); }; init(); }
  e.mode.addEventListener("change", updateMode); e.send.addEventListener("click", sendMessage); e.clear.addEventListener("click", () => { e.message.value = ""; e.candidate.value = ""; e.outputTitle.value = ""; e.saveOutput.checked = false; }); e.newConv.addEventListener("click", newConversation); e.refreshConv.addEventListener("click", loadConvs); e.connectDrive.addEventListener("click", connectDriveFlow); e.disconnectDrive.addEventListener("click", disconnectDriveFlow); e.signOut.addEventListener("click", () => { setToken(""); s.cid = null; s.title = ""; s.convs = []; s.msgs = []; renderConvs(); renderMsgs(); loadDrive(); });
  updateMode(); renderAuth(); setupGoogle(); if (s.token) { s.email = jwt(s.token)?.email || ""; renderAuth(); loadDrive().then(loadConvs).catch(err => { e.heroSub.textContent = err.message; }); }
})();
</script></body></html>
"""
    return dedent(template).replace("__CONFIG__", config).replace("__NOTE__", note)


def build_web_router(*, google_client_id: Optional[str]) -> APIRouter:
    router = APIRouter(tags=["web"])

    @router.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/gm", status_code=307)

    @router.get("/gm", response_class=HTMLResponse, include_in_schema=False)
    def gm_console():
        return HTMLResponse(_page(google_client_id))

    return router
