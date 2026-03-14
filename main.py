from __future__ import annotations

import json
import os
import re
import uuid
import hashlib
import google.auth
import psycopg
import requests
import time
import jwt

from cryptography.hazmat.primitives import serialization
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from app.drive_store import DriveStore, decode_google_export_text
from app.models_v2 import DocumentRef, SessionPatchPayload, SyncSessionPatchRequest, SyncSessionPatchResponse, WorldDocInfo
from app.planner import PlannerService
from app.routes_v2 import build_v2_router
from app.world_model_store import WorldModelStore
from app.workflow_store import WorkflowStore
from googleapiclient.discovery import build
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field, ValidationError
from html import unescape

APP_NAME = "rpg-agent"
app = FastAPI(title=APP_NAME)

# ---- env ----
CAMPAIGN_ID = os.getenv("CAMPAIGN_ID", "kng")
BIBLE_DOC_ID = os.getenv("BIBLE_DOC_ID")
RULES_DOC_ID = os.getenv("RULES_DOC_ID")
GLOSSARY_DOC_ID = os.getenv("GLOSSARY_DOC_ID")
THREADS_DOC_ID = os.getenv("THREADS_DOC_ID")
DB_URL = os.getenv("DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-pro")

# Debug/visibility
REVISION = os.getenv("K_REVISION", "unknown")

WORLD_FOLDER_ENV_MAP = {
    "00 Admin": os.getenv("ADMIN_FOLDER_ID", ""),
    "01 Bible": os.getenv("BIBLE_FOLDER_ID", ""),
    "02 Sessions": os.getenv("SESSIONS_FOLDER_ID", ""),
    "03 NPC": os.getenv("NPC_FOLDER_ID", ""),
    "04 Locations": os.getenv("LOCATIONS_FOLDER_ID", ""),
    "05 Factions": os.getenv("FACTIONS_FOLDER_ID", ""),
    "06 Threads": os.getenv("THREADS_FOLDER_ID", ""),
    "07 Secrets": os.getenv("SECRETS_FOLDER_ID", ""),
    "08 Outputs": os.getenv("OUTPUTS_FOLDER_ID", ""),
}

CORE_WORLD_DOC_MAP = {
    "Campaign Bible": BIBLE_DOC_ID or "",
    "Rules And Tone": RULES_DOC_ID or "",
    "Glossary": GLOSSARY_DOC_ID or "",
    "Thread Tracker": THREADS_DOC_ID or "",
}

# -------------------------
# Models (API)
# -------------------------

AskMode = Literal["auto", "campaign", "general", "scene"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    include_sources: bool = False
    mode: AskMode = "auto"


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []


class ReindexRequest(BaseModel):
    clean: bool = False


class IngestSessionRequest(BaseModel):
    raw_notes: str = Field(..., min_length=1)
    campaign_id: Optional[str] = None


class ThreadPatch(BaseModel):
    thread_id: Optional[str] = None
    title: str
    status: Optional[str] = None
    change: str


class EntityPatch(BaseModel):
    kind: Literal["npc", "location", "faction", "item", "other"] = "other"
    name: str
    description: str
    tags: List[str] = []


class SessionPatch(BaseModel):
    session_summary: str
    thread_tracker_patch: List[ThreadPatch] = []
    entities_patch: List[EntityPatch] = []
    rag_additions: List[str] = []


class IngestAndSyncSessionRequest(IngestSessionRequest):
    source_doc_id: Optional[str] = None
    source_title: Optional[str] = None


class IngestAndSyncSessionResponse(BaseModel):
    patch: SessionPatch
    sync: SyncSessionPatchResponse


class CampaignOut(BaseModel):
    # twardy output dla trybu kampanii (zero dopowiedzeń)
    format: Literal["bullets", "table"] = "bullets"
    bullets: List[str] = []
    table: Optional[Dict[str, Any]] = None
    used_context: List[int] = []


# -------------------------
# Helpers
# -------------------------

def html_table_to_tsv(html: str) -> str:
    if not html:
        return ""

    h = unescape(html)

    # wytnij style/script zanim zrobimy strip tagów (CSS to był wasz problem)
    h = re.sub(r"(?is)<style.*?</style>", " ", h)
    h = re.sub(r"(?is)<script.*?</script>", " ", h)

    # wiersze i komórki
    h = re.sub(r"(?is)</tr\s*>", "\n", h)
    h = re.sub(r"(?is)</t[dh]\s*>", " | ", h)

    # usuń tagi
    h = re.sub(r"(?is)<[^>]+>", " ", h)

    # normalizacja whitespace
    h = re.sub(r"[ \t]{2,}", " ", h)

    lines = [ln.strip() for ln in h.splitlines()]
    lines = [ln.strip(" |") for ln in lines if ln and len(ln) > 3]

    norm: List[str] = []
    for ln in lines:
        ln = ln.replace("\t", " | ")
        ln = re.sub(r"\s*\|\s*", " | ", ln)
        ln = re.sub(r"[ \t]{2,}", " ", ln).strip()

        # wiersz tabeli = co najmniej 3 kolumny
        if ln.count("|") >= 2:
            norm.append(ln)

    # startuj od nagłówka tabeli jeśli istnieje
    for i, ln in enumerate(norm):
        if "Thread ID" in ln and "|" in ln:
            norm = norm[i:]
            break

    # jak nie ma nagłówka, to od pierwszego Txx |
    for i, ln in enumerate(norm):
        if re.match(r"^T\d+\s*\|", ln):
            norm = norm[i:]
            break

    return "\n".join(norm).strip()


def require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise HTTPException(status_code=400, detail=f"Missing env: {name}")
    return value


def sanitize_for_rag(text: str) -> str:
    """
    Wywala oczywiste śmieci z terminala i logów.
    Cel: żeby "cat > rpg.sh", prompty, gcloud itp. nie miały szans wejść do embeddingów.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out: List[str] = []

    bad_prefix = (
        "$ ",
        "gcloud ",
        "curl ",
        "cat >",
        "cat <<",
        "EOF",
        "kubectl ",
        "docker ",
        "pip ",
        "python ",
        "export ",
    )

    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if any(s.startswith(p) for p in bad_prefix):
            continue
        if s.endswith("$") and len(s) <= 80:
            continue
        out.append(ln)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def chunk_text(text: str, max_chars: int = 2400, overlap: int = 400) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(0, end - overlap)
    return chunks


def chunk_threads(text: str) -> List[str]:
    """
    Thread Tracker zwykle jest liniowy/tabelkowy.
    Chunkowanie po liniach, żeby nie ucinało wątku typu "T05 | ...".
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: List[str] = []
    buf: List[str] = []
    for ln in lines:
        buf.append(ln)
        if len(buf) >= 40:
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def gemini_embed(texts: List[str]) -> List[List[float]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("Brak GEMINI_API_KEY")
    out: List[List[float]] = []
    for t in texts:
        url = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:embedContent"
        payload = {"content": {"parts": [{"text": t}]}}
        r = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Gemini embeddings error: {r.status_code} {r.text}")
        data = r.json()
        out.append(data["embedding"]["values"])
    return out


def gemini_generate(
    prompt: str,
    *,
    response_mime_type: str = "text/plain",
    temperature: float = 0.6,
    max_output_tokens: int = 2500,
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Brak GEMINI_API_KEY")

    url = f"https://generativelanguage.googleapis.com/v1beta/{GEN_MODEL}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": response_mime_type,
        },
    }
    r = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini generate error: {r.status_code} {r.text}")
    data = r.json()

    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        out = "\n".join([t for t in texts if t.strip()]).strip()
        if out:
            return out
    except Exception:
        pass
    return str(data)


def get_drive_service():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def export_google_doc_as_text(drive, doc_id: str) -> str:
    data = drive.files().export(fileId=doc_id, mimeType="text/plain").execute()
    return decode_google_export_text(data)


def export_google_doc(drive, doc_id: str, mime_type: str) -> str:
    data = drive.files().export(fileId=doc_id, mimeType=mime_type).execute()
    return decode_google_export_text(data)


def db_conn():
    require_env("DB_URL", DB_URL)
    conn = psycopg.connect(DB_URL)  # type: ignore[arg-type]
    register_vector(conn)
    return conn


def delete_chunks_for_doc(doc_id: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "delete from chunks where campaign_id = %s and doc_id = %s",
                (CAMPAIGN_ID, doc_id),
            )
        conn.commit()


def delete_all_chunks_for_campaign():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("delete from chunks where campaign_id = %s", (CAMPAIGN_ID,))
        conn.commit()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync_world_doc_state(
    doc: WorldDocInfo,
    *,
    doc_type: str,
    raw_content: str,
    chunk_count: int,
) -> bool:
    if not doc.doc_id:
        return False

    doc_hash = content_hash(raw_content)
    entity_type = getattr(doc.entity_type, "value", str(doc.entity_type))
    metadata = json.dumps(
        {
            "path_hint": doc.path_hint,
            "doc_type": doc_type,
            "chunk_count": chunk_count,
        }
    )

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into world_docs (
                    campaign_id,
                    doc_id,
                    folder,
                    title,
                    entity_type,
                    content_hash,
                    last_synced_at,
                    metadata
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                on conflict (campaign_id, doc_id)
                do update set
                    folder = excluded.folder,
                    title = excluded.title,
                    entity_type = excluded.entity_type,
                    content_hash = excluded.content_hash,
                    last_synced_at = excluded.last_synced_at,
                    metadata = excluded.metadata
                """,
                (
                    CAMPAIGN_ID,
                    doc.doc_id,
                    doc.folder,
                    doc.title,
                    entity_type or "other",
                    doc_hash,
                    datetime.now(timezone.utc),
                    metadata,
                ),
            )
            cur.execute(
                """
                select 1
                from doc_snapshots
                where campaign_id = %s and doc_id = %s and content_hash = %s
                limit 1
                """,
                (CAMPAIGN_ID, doc.doc_id, doc_hash),
            )
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(
                    """
                    insert into doc_snapshots (campaign_id, doc_id, content_hash, content_text)
                    values (%s, %s, %s, %s)
                    """,
                    (CAMPAIGN_ID, doc.doc_id, doc_hash, raw_content),
                )
        conn.commit()

    return not exists


def upsert_chunks(doc: WorldDocInfo, doc_type: str, chunks: List[str], embeddings: List[List[float]]):
    """
    Insert-only (tak jak wcześniej). Dla MVP OK.
    Przy reindex warto używać clean=true.
    """
    now = datetime.now(timezone.utc)
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "delete from chunks where campaign_id = %s and doc_id = %s",
                (CAMPAIGN_ID, doc.doc_id),
            )
            for t, emb in zip(chunks, embeddings):
                cid = str(uuid.uuid4())
                cur.execute(
                    """
                    insert into chunks (id, campaign_id, doc_id, doc_type, chunk_text, embedding, metadata, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        cid,
                        CAMPAIGN_ID,
                        doc.doc_id,
                        doc_type,
                        t,
                        emb,
                        json.dumps(
                            {
                                "source": doc_type,
                                "title": doc.title,
                                "folder": doc.folder,
                                "path_hint": doc.path_hint,
                            }
                        ),
                        now,
                    ),
                )
        conn.commit()


def vector_search(question: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = gemini_embed([question])[0]
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    c.id,
                    c.doc_id,
                    c.doc_type,
                    c.chunk_text,
                    (c.embedding <-> %s::vector) as distance,
                    coalesce(wd.title, c.metadata->>'title', '') as title,
                    coalesce(wd.folder, c.metadata->>'folder', '') as folder,
                    coalesce(wd.metadata->>'path_hint', c.metadata->>'path_hint', '') as path_hint
                from chunks c
                left join world_docs wd
                    on wd.campaign_id = c.campaign_id
                   and wd.doc_id = c.doc_id
                where c.campaign_id = %s
                order by c.embedding <-> %s::vector
                limit %s
                """,
                (q_emb, CAMPAIGN_ID, q_emb, top_k),
            )
            rows = cur.fetchall()
    return [
        {
            "chunk_id": cid,
            "doc_id": d,
            "doc_type": t,
            "chunk_text": c,
            "distance": float(dist),
            "title": title,
            "folder": folder,
            "path_hint": path_hint,
        }
        for (cid, d, t, c, dist, title, folder, path_hint) in rows
    ]


def is_campaign_question(text: str) -> bool:
    t = text.lower()
    keywords = [
        "kampania",
        "krew na gwiazdach",
        "premis",
        "wątk",
        "thread",
        "npc",
        "mg",
        "sesj",
        "fabuł",
        "scen",
        "lokac",
        "frakc",
        "bible",
        "glossary",
        "rules",
        "timeline",
        "akt",
        "prolog",
    ]
    return any(k in t for k in keywords)


def extract_json_object(text: str) -> str:
    """
    Gemini czasem dorzuci coś dookoła. Wyciągamy pierwsze sensowne {...}.
    """
    text = (text or "").strip()
    if not text:
        return ""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else ""


def render_campaign_out(out: CampaignOut) -> str:
    if out.format == "table" and out.table:
        cols = out.table.get("columns") or []
        rows = out.table.get("rows") or []
        if not cols or not isinstance(cols, list):
            return "brak w notatkach"
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body_lines = []
        for r in rows:
            if isinstance(r, list):
                body_lines.append("| " + " | ".join(str(x) for x in r) + " |")
        md = "\n".join([header, sep] + body_lines).strip()
        return md or "brak w notatkach"

    bullets = [b.strip() for b in (out.bullets or []) if b and b.strip()]
    if not bullets:
        return "brak w notatkach"
    return "\n".join([f"1. {b}" for b in bullets])


def build_campaign_prompt(question: str, context: str) -> str:
    return f"""
Jesteś asystentem MG kampanii "Krew Na Gwiazdach". Odpowiadasz po polsku.

ZASADY (twarde):
1) Używaj wyłącznie faktów z KONTEKSTU.
2) Jeśli czegoś nie ma w kontekście, zwróć JSON z format="bullets" i bullets=["brak w notatkach"].
3) Nie dopowiadaj, nie spekuluj, nie twórz "trzeciej drogi", nie dodawaj żadnych sekcji ponad to co trzeba.
4) Zwróć wyłącznie JSON. Żadnego markdown, żadnego tekstu dookoła.

Dozwolony JSON (dokładnie te pola):
{{
  "format": "bullets" | "table",
  "bullets": ["..."],
  "table": {{"columns": ["..."], "rows": [["..."]]}},
  "used_context": [1,2,3]
}}

KONTEKST:
{context}

PYTANIE:
{question}

ODPOWIEDŹ (tylko JSON):
""".strip()


def build_general_prompt(question: str) -> str:
    return f"""
Odpowiedz po polsku dokładnie na pytanie użytkownika.
Nie dodawaj sekcji, których nie wymaga pytanie.

PYTANIE:
{question}

ODPOWIEDŹ:
""".strip()


def ctx_slice(h: Dict[str, Any]) -> str:
    doc_type = h.get("doc_type", "")
    text = h.get("chunk_text", "") or ""
    if not text:
        return ""

    limit = 2400 if doc_type == "threads" else 1800
    if len(text) <= limit:
        return text

    head = text[:900].rstrip()
    anchor = ""
    match = re.search(r"#{1,6}\s+[^\n]+", text)
    if match:
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 400)
        anchor = text[start:end].strip()
    if not anchor:
        midpoint = len(text) // 2
        start = max(0, midpoint - 300)
        end = min(len(text), midpoint + 300)
        anchor = text[start:end].strip()
    tail = text[-900:].lstrip()
    parts = [head]
    if anchor and anchor not in head and anchor not in tail:
        parts.extend(["...", anchor])
    parts.extend(["...", tail])
    return "\n".join(parts)


def build_campaign_context(hits: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, hit in enumerate(hits, start=1):
        title = hit.get("title") or "unknown"
        folder = hit.get("folder") or "unknown"
        doc_type = hit.get("doc_type") or "other"
        path_hint = hit.get("path_hint") or ""
        label = f"[{idx}] title={title} | folder={folder} | doc_type={doc_type}"
        if path_hint:
            label += f" | path={path_hint}"
        blocks.append(f"{label}\n{ctx_slice(hit)}")
    return "\n\n".join(blocks)


def build_campaign_prompt(question: str, context: str) -> str:
    return f"""
Jestes asystentem MG kampanii "Krew Na Gwiazdach". Odpowiadasz po polsku.

ZASADY:
1) Uzywaj wylacznie faktow z KONTEKSTU.
2) Jesli odpowiedz jest w kontekscie, podaj ja wprost i mozliwie doslownie.
3) Szczegolnie zwracaj uwage na zgodnosc pytania z title, folder i trescia chunku.
4) Jesli czegos nie ma w kontekscie, zwroc JSON z format="bullets" i bullets=["brak w notatkach"].
5) Nie dopowiadaj, nie spekuluj i nie lacz faktow spoza kontekstu.
6) W used_context podaj numery blokow, z ktorych skorzystales.
7) Zwracaj wylacznie JSON. Bez markdown i bez tekstu dookola.

Dozwolony JSON:
{{
  "format": "bullets" | "table",
  "bullets": ["..."],
  "table": {{"columns": ["..."], "rows": [["..."]]}},
  "used_context": [1,2,3]
}}

KONTEKST (kazdy blok zawiera title, folder, doc_type i tresc chunku):
{context}

PYTANIE:
{question}

ODPOWIEDZ (tylko JSON):
""".strip()


GITHUB_API = "https://api.github.com"
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "Soolik")
GITHUB_REPO = os.getenv("GITHUB_REPO", "rpg-agent")


def github_installation_token() -> str:
    app_id = require_env("GITHUB_APP_ID", os.getenv("GITHUB_APP_ID"))
    installation_id = require_env("GITHUB_INSTALLATION_ID", os.getenv("GITHUB_INSTALLATION_ID"))
    private_key_pem = require_env("GITHUB_PRIVATE_KEY_PEM", os.getenv("GITHUB_PRIVATE_KEY_PEM")).encode("utf-8")

    private_key = serialization.load_pem_private_key(private_key_pem, password=None)

    now = int(time.time())
    payload = {"iat": now - 30, "exp": now + 9 * 60, "iss": app_id}
    jwt_token = jwt.encode(payload, private_key, algorithm="RS256")

    r = requests.post(
        f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
        headers={"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"},
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub access_tokens error: {r.status_code} {r.text}")
    return r.json()["token"]


def gh_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


@app.get("/debug_github_auth", response_class=PlainTextResponse)
def debug_github_auth():
    try:
        token = github_installation_token()
        r = requests.get(
            f"{GITHUB_API}/repos/{GITHUB_OWNER}/{GITHUB_REPO}",
            headers=gh_headers(token),
            timeout=30,
        )
        if r.status_code != 200:
            return f"ERR repo_read {r.status_code}: {r.text}\n"
        data = r.json()
        return f"OK token_len={len(token)} repo={data.get('full_name')} default_branch={data.get('default_branch')}\n"
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}\n"


def count_indexed_chunks() -> Optional[int]:
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from chunks where campaign_id = %s", (CAMPAIGN_ID,))
                row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None


def planner_generate_json(prompt: str) -> str:
    return gemini_generate(
        prompt,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=3000,
    ).strip()


def planner_generate_text(prompt: str) -> str:
    return gemini_generate(
        prompt,
        response_mime_type="text/plain",
        temperature=0.3,
        max_output_tokens=2000,
    ).strip()


def build_drive_store() -> DriveStore:
    folder_map = {k: v for k, v in WORLD_FOLDER_ENV_MAP.items() if v}
    core_doc_map = {k: v for k, v in CORE_WORLD_DOC_MAP.items() if v}
    return DriveStore(folder_map=folder_map, core_doc_map=core_doc_map)


def build_workflow_store() -> Optional[WorkflowStore]:
    if not DB_URL:
        return None
    return WorkflowStore(campaign_id=CAMPAIGN_ID, connection_factory=db_conn)


def build_world_model_store() -> Optional[WorldModelStore]:
    if not DB_URL:
        return None
    return WorldModelStore(campaign_id=CAMPAIGN_ID, connection_factory=db_conn)


def doc_type_for_indexing(doc: WorldDocInfo) -> str:
    if doc.title == "Thread Tracker":
        return "threads"
    entity_type = getattr(doc.entity_type, "value", str(doc.entity_type))
    return entity_type or "other"


def export_world_doc_for_indexing(drive, doc: WorldDocInfo) -> str:
    if not doc.doc_id:
        return ""

    if doc.title == "Thread Tracker":
        raw_html = export_google_doc(drive, doc.doc_id, "text/html")
        raw = html_table_to_tsv(raw_html)
        if raw.strip():
            return raw

    return export_google_doc(drive, doc.doc_id, "text/plain")


def index_world_docs(docs: List[WorldDocInfo], *, clean: bool = False) -> Dict[str, Any]:
    if not docs:
        raise HTTPException(status_code=400, detail="No world docs configured for indexing")

    drive = get_drive_service()
    if clean:
        delete_all_chunks_for_campaign()

    total_chunks = 0
    indexed_docs = 0
    skipped_docs = 0
    changed_docs = 0
    for doc in docs:
        if not doc.doc_id:
            skipped_docs += 1
            continue

        doc_type = doc_type_for_indexing(doc)
        raw = export_world_doc_for_indexing(drive, doc)

        if doc.title == "Thread Tracker":
            cleaned = raw.strip()
            chunks = chunk_threads(cleaned)
            if len(cleaned) < 50 or not chunks:
                sync_world_doc_state(
                    doc,
                    doc_type=doc_type,
                    raw_content=cleaned,
                    chunk_count=0,
                )
                delete_chunks_for_doc(doc.doc_id)
                skipped_docs += 1
                continue
        else:
            cleaned = sanitize_for_rag(raw)
            chunks = chunk_text(cleaned)
            if not chunks:
                sync_world_doc_state(
                    doc,
                    doc_type=doc_type,
                    raw_content=cleaned,
                    chunk_count=0,
                )
                delete_chunks_for_doc(doc.doc_id)
                skipped_docs += 1
                continue

        embs = gemini_embed(chunks)
        upsert_chunks(doc=doc, doc_type=doc_type, chunks=chunks, embeddings=embs)
        if sync_world_doc_state(
            doc,
            doc_type=doc_type,
            raw_content=cleaned,
            chunk_count=len(chunks),
        ):
            changed_docs += 1
        total_chunks += len(chunks)
        indexed_docs += 1

    return {
        "ok": True,
        "indexed_docs": indexed_docs,
        "indexed_chunks": total_chunks,
        "skipped_docs": skipped_docs,
        "changed_docs": changed_docs,
        "clean": clean,
    }


def resolve_reindex_docs(targets: List[DocumentRef]) -> List[WorldDocInfo]:
    docs: List[WorldDocInfo] = []
    seen_doc_ids = set()

    for target in targets:
        found = drive_store_v2.find_doc(
            folder=target.folder if target.folder else None,
            title=target.title if target.title else None,
            doc_id=target.doc_id,
        )
        if not found or not found.doc_id:
            continue
        if found.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(found.doc_id)
        docs.append(found)

    return docs


def reindex_after_apply_default(targets: List[DocumentRef]) -> Dict[str, Any]:
    docs = resolve_reindex_docs(targets)
    if not docs:
        return {
            "ok": True,
            "indexed_docs": 0,
            "indexed_chunks": 0,
            "skipped_docs": 0,
            "changed_docs": 0,
            "clean": False,
            "mode": "partial",
        }
    result = index_world_docs(docs, clean=False)
    result["mode"] = "partial"
    return result


drive_store_v2 = build_drive_store()
workflow_store_v2 = build_workflow_store()
world_model_store_v2 = build_world_model_store()
planner_v2 = PlannerService(generate_text_fn=planner_generate_json)
consistency_planner_v2 = PlannerService(generate_text_fn=planner_generate_text)
app.include_router(
    build_v2_router(
        drive_store=drive_store_v2,
        planner=planner_v2,
        reindex_fn=reindex_after_apply_default,
        indexed_chunks_fn=count_indexed_chunks,
        campaign_id=CAMPAIGN_ID,
        workflow_store=workflow_store_v2,
        world_model_store=world_model_store_v2,
    )
)


# -------------------------
# Endpoints
# -------------------------

@app.get("/debug_env", response_class=PlainTextResponse)
def debug_env():
    keys = [
        "GITHUB_APP_ID",
        "GITHUB_INSTALLATION_ID",
        "GITHUB_PRIVATE_KEY_PEM",
    ]
    out = []
    for k in keys:
        v = os.getenv(k)
        out.append(f"{k}={'SET' if v else 'MISSING'} len={len(v) if v else 0}")
    return "\n".join(out) + "\n"


@app.get("/health")
def health():
    return {"ok": True, "campaign_id": CAMPAIGN_ID, "revision": REVISION}


@app.post("/reindex")
def reindex(body: ReindexRequest = ReindexRequest()):
    try:
        docs = drive_store_v2.list_world_docs()
        result = index_world_docs(docs, clean=body.clean)
        result["mode"] = "full"
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def ctx_slice(h: Dict[str, Any]) -> str:
    doc_type = h.get("doc_type", "")
    text = h.get("chunk_text", "") or ""
    if not text:
        return ""
    limit = 2400 if doc_type == "threads" else 1800
    if len(text) <= limit:
        return text
    head = text[:900].rstrip()
    anchor = ""
    match = re.search(r"#{1,6}\s+[^\n]+", text)
    if match:
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 400)
        anchor = text[start:end].strip()
    if not anchor:
        midpoint = len(text) // 2
        start = max(0, midpoint - 300)
        end = min(len(text), midpoint + 300)
        anchor = text[start:end].strip()
    tail = text[-900:].lstrip()
    parts = [head]
    if anchor and anchor not in head and anchor not in tail:
        parts.extend(["...", anchor])
    parts.extend(["...", tail])
    return "\n".join(parts)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        q = req.question.strip()

        if req.mode in ("campaign", "scene"):
            campaign_mode = True
        elif req.mode == "general":
            campaign_mode = False
        else:
            campaign_mode = is_campaign_question(q)

        if not campaign_mode:
            answer = gemini_generate(build_general_prompt(q)).strip()
            return AskResponse(answer=answer, sources=[])

        hits = vector_search(q, req.top_k)
        context = build_campaign_context(hits)
        prompt = build_campaign_prompt(q, context)

        raw = gemini_generate(
            prompt,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=2000,
        ).strip()

        parsed: Optional[CampaignOut] = None
        try:
            obj = json.loads(extract_json_object(raw))
            parsed = CampaignOut.model_validate(obj)
        except Exception:
            fix = f"""
Napraw output. Masz zwrócić wyłącznie JSON zgodny z tym schematem i nic więcej.
Jeśli brak danych: format="bullets", bullets=["brak w notatkach"].

ZŁY OUTPUT:
{raw}

POPRAWNY OUTPUT (tylko JSON):
""".strip()
            raw2 = gemini_generate(
                fix,
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=1500,
            ).strip()
            obj2 = json.loads(extract_json_object(raw2))
            parsed = CampaignOut.model_validate(obj2)

        answer = render_campaign_out(parsed)

        sources: List[Dict[str, Any]] = []
        if req.include_sources:
            sources = [
                {
                    "doc_type": h["doc_type"],
                    "doc_id": h["doc_id"],
                    "chunk_id": h["chunk_id"],
                    "distance": h["distance"],
                    "title": h.get("title"),
                    "folder": h.get("folder"),
                    "path_hint": h.get("path_hint"),
                }
                for h in hits
            ]

        return AskResponse(answer=answer, sources=sources)

    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Output validation error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_text", response_class=PlainTextResponse)
def ask_text(req: AskRequest):
    resp = ask(req)
    return resp.answer + "\n"


def generate_session_patch(raw_notes: str) -> SessionPatch:
    cleaned_notes = raw_notes.strip()
    if not cleaned_notes:
        raise HTTPException(status_code=400, detail="raw_notes is empty")

    notes = sanitize_for_rag(cleaned_notes)

    prompt = f"""
Jestes asystentem MG kampanii "Krew Na Gwiazdach".
Masz z surowych notatek wygenerowac PATCH do dokumentow kampanii.
Nie zmyslaj. Jesli czegos nie ma w notatkach, pomin to.
Zwroc wylacznie JSON zgodny z tym schematem:
{{
  "session_summary": "krotkie podsumowanie (max 8 zdan)",
  "thread_tracker_patch": [{{"thread_id": "Txx (opcjonalnie)", "title": "...", "status": "...", "change": "co dopisac/zmienic"}}],
  "entities_patch": [{{"kind": "npc|location|faction|item|other", "name": "...", "description": "...", "tags": ["..."]}}],
  "rag_additions": ["krotkie fakty warte wejscia do indeksu, bez smieci z terminala"]
}}

NOTATKI:
{notes}

PATCH (tylko JSON):
""".strip()

    raw = gemini_generate(
        prompt,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=2500,
    ).strip()

    obj = json.loads(extract_json_object(raw))
    return SessionPatch.model_validate(obj)


def sync_generated_session_patch(req: IngestAndSyncSessionRequest, patch: SessionPatch) -> SyncSessionPatchResponse:
    if req.campaign_id and req.campaign_id != CAMPAIGN_ID:
        raise HTTPException(status_code=400, detail="campaign_id does not match configured campaign")
    if not world_model_store_v2:
        raise HTTPException(status_code=503, detail="World model store is not configured")

    sync_request = SyncSessionPatchRequest(
        patch=SessionPatchPayload.model_validate(patch.model_dump(mode="json")),
        raw_notes=req.raw_notes,
        campaign_id=req.campaign_id,
        source_doc_id=req.source_doc_id,
        source_title=req.source_title,
    )
    response = world_model_store_v2.sync_session_patch(sync_request)
    if response is None:
        raise HTTPException(status_code=503, detail="World model store is not configured")
    return response


@app.post("/ingest_session", response_model=SessionPatch)
def ingest_session(req: IngestSessionRequest):
    """
    Wklejasz surowe notatki z sesji -> agent generuje patch.
    Niczego nie zapisujemy do Google Docs automatem.
    """
    try:
        return generate_session_patch(req.raw_notes)

        raw_notes = req.raw_notes.strip()
        if not raw_notes:
            raise HTTPException(status_code=400, detail="raw_notes is empty")

        notes = sanitize_for_rag(raw_notes)

        prompt = f"""
Jesteś asystentem MG kampanii "Krew Na Gwiazdach".
Masz z surowych notatek wygenerować PATCH do dokumentów kampanii.
Nie zmyślaj. Jeśli czegoś nie ma w notatkach, pomiń to.
Zwróć wyłącznie JSON zgodny z tym schematem:
{{
  "session_summary": "krótkie podsumowanie (max 8 zdań)",
  "thread_tracker_patch": [{{"thread_id": "Txx (opcjonalnie)", "title": "...", "status": "...", "change": "co dopisać/zmienić"}}],
  "entities_patch": [{{"kind": "npc|location|faction|item|other", "name": "...", "description": "...", "tags": ["..."]}}],
  "rag_additions": ["krótkie fakty warte wejścia do indeksu, bez śmieci z terminala"]
}}

NOTATKI:
{notes}

PATCH (tylko JSON):
""".strip()

        raw = gemini_generate(
            prompt,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=2500,
        ).strip()

        obj = json.loads(extract_json_object(raw))
        patch = SessionPatch.model_validate(obj)
        return patch

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest_session_and_sync", response_model=IngestAndSyncSessionResponse)
def ingest_session_and_sync(req: IngestAndSyncSessionRequest):
    try:
        patch = generate_session_patch(req.raw_notes)
        sync = sync_generated_session_patch(req, patch)
        return IngestAndSyncSessionResponse(patch=patch, sync=sync)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply_patch")
def apply_patch(_: SessionPatch):
    """
    Na start: celowo NIE zapisujemy automatem do Google Docs.
    Ten endpoint to placeholder pod przyszły, świadomy apply.
    """
    return {"ok": True, "applied": False, "reason": "not implemented (manual approval flow first)"}


@app.get("/debug_threads_preview", response_class=PlainTextResponse)
def debug_threads_preview():
    require_env("THREADS_DOC_ID", THREADS_DOC_ID)
    drive = get_drive_service()

    raw_html = export_google_doc(drive, THREADS_DOC_ID, "text/html")
    tsv = html_table_to_tsv(raw_html)
    cleaned = tsv.strip()

    if not cleaned:
        return "EMPTY\n"

    return cleaned[:8000] + "\n"
