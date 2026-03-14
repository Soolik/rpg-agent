from __future__ import annotations

import json
import os
import re
import uuid
import hashlib
from copy import deepcopy
from contextvars import ContextVar
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
from app.models_v2 import (
    ChangeProposal,
    DocumentRef,
    ProposeChangesRequest,
    SessionPatchPayload,
    SyncSessionPatchRequest,
    SyncSessionPatchResponse,
    WorldDocInfo,
    WorldEntityType,
)
from app.planner import PlannerService
from app.routes_v2 import build_context_for_planner, build_v2_router
from app.world_model_store import WorldModelStore
from app.workflow_store import WorkflowStore
from googleapiclient.discovery import build
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field, ValidationError
from html import unescape

APP_NAME = "rpg-agent"
app = FastAPI(title=APP_NAME)

REQUEST_TELEMETRY: ContextVar[Optional[Dict[str, Any]]] = ContextVar("request_telemetry", default=None)

# ---- env ----
CAMPAIGN_ID = os.getenv("CAMPAIGN_ID", "kng")
BIBLE_DOC_ID = os.getenv("BIBLE_DOC_ID")
RULES_DOC_ID = os.getenv("RULES_DOC_ID")
GLOSSARY_DOC_ID = os.getenv("GLOSSARY_DOC_ID")
THREADS_DOC_ID = os.getenv("THREADS_DOC_ID")
DB_URL = os.getenv("DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_ROLLUP_DOC_ID = os.getenv("OUTPUT_ROLLUP_DOC_ID")
OUTPUT_ROLLUP_DOC_TITLE = os.getenv("OUTPUT_ROLLUP_DOC_TITLE")
OUTPUT_ROLLUP_MODE = os.getenv("OUTPUT_ROLLUP_MODE", "replace").strip().lower()

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
ChatIntent = Literal["auto", "answer", "proposal", "session_sync", "creative"]
ArtifactType = Literal[
    "gm_brief",
    "session_report",
    "player_summary",
    "pre_session_brief",
    "session_hooks",
    "scene_seed",
    "npc_brief",
    "twist_pack",
]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    include_sources: bool = False
    mode: AskMode = "auto"


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    intent: ChatIntent = "auto"
    artifact_type: Optional[ArtifactType] = None
    source_title: Optional[str] = None
    include_sources: bool = False
    include_telemetry: bool = False
    save_output: bool = False
    output_title: Optional[str] = None


class ChatResponse(BaseModel):
    kind: Literal["answer", "proposal", "session_sync", "creative"]
    reply: str
    artifact_type: Optional[ArtifactType] = None
    artifact_text: Optional[str] = None
    proposal_id: Optional[int] = None
    session_id: Optional[int] = None
    references: List[str] = []
    warnings: List[str] = []
    output_doc_id: Optional[str] = None
    output_title: Optional[str] = None
    output_path: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None


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


def start_request_telemetry(enabled: bool):
    if not enabled:
        return None
    return REQUEST_TELEMETRY.set(
        {
            "gemini_calls": [],
            "sections": [],
            "artifacts": [],
        }
    )


def reset_request_telemetry(token) -> None:
    if token is not None:
        REQUEST_TELEMETRY.reset(token)


def record_telemetry(bucket: str, event: Dict[str, Any]) -> None:
    telemetry = REQUEST_TELEMETRY.get()
    if telemetry is None:
        return
    telemetry.setdefault(bucket, []).append(event)


def current_request_telemetry() -> Optional[Dict[str, Any]]:
    telemetry = REQUEST_TELEMETRY.get()
    if telemetry is None:
        return None
    return deepcopy(telemetry)


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
    telemetry_label: Optional[str] = None,
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
        record_telemetry(
            "gemini_calls",
            {
                "label": telemetry_label or "gemini_generate",
                "status_code": r.status_code,
                "error": r.text[:500],
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": response_mime_type,
                "prompt_chars": len(prompt),
            },
        )
        raise RuntimeError(f"Gemini generate error: {r.status_code} {r.text}")
    data = r.json()

    out = ""
    finish_reason = None
    try:
        candidate = data["candidates"][0]
        finish_reason = candidate.get("finishReason")
        parts = candidate["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        out = "\n".join([t for t in texts if t.strip()]).strip()
    except Exception:
        out = ""

    usage = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
    record_telemetry(
        "gemini_calls",
        {
            "label": telemetry_label or "gemini_generate",
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": response_mime_type,
            "prompt_chars": len(prompt),
            "response_chars": len(out),
            "finish_reason": finish_reason,
            "prompt_block_reason": (data.get("promptFeedback") or {}).get("blockReason"),
            "safety_ratings": (data.get("candidates") or [{}])[0].get("safetyRatings"),
            "prompt_token_count": usage.get("promptTokenCount"),
            "candidates_token_count": usage.get("candidatesTokenCount"),
            "total_token_count": usage.get("totalTokenCount"),
            "thoughts_token_count": usage.get("thoughtsTokenCount"),
            "response_preview": out[:200],
        },
    )
    if out:
        return out
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


def detect_chat_intent(message: str) -> Literal["answer", "proposal", "session_sync", "creative"]:
    text = (message or "").strip()
    lowered = text.lower()

    creative_markers = [
        "wymysl ",
        "wymyśl ",
        "pomysl ",
        "pomysł ",
        "zaproponuj ",
        "daj 3 pomysly",
        "daj 3 pomysły",
        "hook",
        "hooki",
        "twist",
        "twisty",
        "seed",
        "scene seed",
        "nowego npc",
        "nowy npc",
        "stworz npc",
        "stwórz npc",
        "npc brief",
    ]
    if any(marker in lowered for marker in creative_markers):
        return "creative"

    proposal_markers = [
        "dodaj ",
        "podmien ",
        "podmień ",
        "zamien ",
        "zamień ",
        "zmien ",
        "zmień ",
        "utworz ",
        "utwórz ",
        "stworz ",
        "stwórz ",
        "uzupelnij ",
        "uzupełnij ",
        "w dokumencie ",
        "sekcje ",
        "sekcję ",
    ]
    if any(marker in lowered for marker in proposal_markers):
        return "proposal"

    line_count = len([line for line in text.splitlines() if line.strip()])
    sentence_count = len(re.findall(r"[.!?]+", text))
    if "?" in text:
        return "answer"
    if line_count >= 3 or sentence_count >= 2 or len(text) >= 180:
        return "session_sync"
    return "answer"


def render_source_labels(sources: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    seen = set()
    for source in sources:
        label = " / ".join(filter(None, [source.get("folder"), source.get("title")])).strip()
        if not label:
            label = source.get("doc_id") or "unknown source"
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def extract_titlecase_phrases(text: str) -> List[str]:
    ignored = {
        "wymysl",
        "przygotuj",
        "stworz",
        "potrzebny",
        "daj",
        "zrob",
        "opisz",
    }
    matches = re.findall(
        r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,3}\b",
        text or "",
    )
    phrases: List[str] = []
    seen = set()
    for match in matches:
        cleaned = match.strip()
        key = normalize_world_model_key(cleaned)
        if not cleaned or key in seen:
            continue
        if " " not in cleaned and key in ignored:
            continue
        seen.add(key)
        phrases.append(cleaned)
    return phrases


def collect_canonical_names(message: str, hits: List[Dict[str, Any]], limit: int = 12) -> List[str]:
    names: List[str] = []
    seen = set()
    normalized_message = normalize_world_model_key(message)

    def add_name(value: Optional[str]) -> None:
        cleaned = (value or "").strip()
        key = normalize_world_model_key(cleaned)
        if not cleaned or not key or key in seen:
            return
        seen.add(key)
        names.append(cleaned)

    try:
        if world_model_store_v2:
            for entity in world_model_store_v2.list_entities(limit=100):
                if normalize_world_model_key(entity.name) in normalized_message:
                    add_name(entity.name)
            for thread in world_model_store_v2.list_threads(limit=100):
                if normalize_world_model_key(thread.title) in normalized_message:
                    add_name(thread.title)
    except Exception:
        pass

    for phrase in extract_titlecase_phrases(message):
        add_name(phrase)

    for hit in hits:
        title = (hit.get("title") or "").strip()
        if title and normalize_world_model_key(title) in normalized_message:
            add_name(title)

    return names[:limit]


def build_canonical_names_context(canonical_names: List[str]) -> str:
    if not canonical_names:
        return "Brak dodatkowych nazw kanonicznych."
    return "\n".join(
        [
            "NAZWY KANONICZNE:",
            *[f"- {name}" for name in canonical_names],
            "ZASADY DLA NAZW KANONICZNYCH:",
            "- Nie tlumacz tych nazw.",
            "- Nie zamieniaj ich na synonimy ani spolszczone odpowiedniki.",
            "- Jesli uzywasz tych nazw, kopiuj je dokladnie w tej formie.",
        ]
    )


def is_creative_artifact_type(artifact_type: Optional[ArtifactType]) -> bool:
    return artifact_type in {"session_hooks", "scene_seed", "npc_brief", "twist_pack"}


def resolved_creative_artifact_type(artifact_type: Optional[ArtifactType]) -> ArtifactType:
    if is_creative_artifact_type(artifact_type):
        return artifact_type  # type: ignore[return-value]
    return "session_hooks"


def infer_artifact_type(message: str, requested: Optional[ArtifactType]) -> Optional[ArtifactType]:
    if requested:
        return requested

    lowered = (message or "").strip().lower()
    if not lowered:
        return None

    if ("briefing" in lowered or "brief" in lowered) and "sesj" in lowered:
        return "pre_session_brief"
    if "raport sesji" in lowered or "raport z sesji" in lowered or "session report" in lowered:
        return "session_report"
    if "player summary" in lowered or "podsumowanie dla graczy" in lowered:
        return "player_summary"
    if "gm brief" in lowered or "briefing mg" in lowered:
        return "gm_brief"
    if "hook" in lowered:
        return "session_hooks"
    if "scene seed" in lowered or ("scene" in lowered and "seed" in lowered):
        return "scene_seed"
    if "npc brief" in lowered or "nowego npc" in lowered or "nowy npc" in lowered:
        return "npc_brief"
    if "twist" in lowered:
        return "twist_pack"
    return None


def render_session_sync_reply(
    patch: SessionPatch,
    sync: SyncSessionPatchResponse,
    *,
    source_title: Optional[str] = None,
) -> str:
    lines = ["Zaktualizowalem model swiata z notatek."]
    if source_title:
        lines.append(f"Zrodlo: {source_title}.")
    lines.append(f"Podsumowanie: {patch.session_summary}")
    if patch.entities_patch:
        lines.append("Encje:")
        for entity in patch.entities_patch[:5]:
            lines.append(f"- {entity.kind}: {entity.name} - {entity.description}")
    if patch.thread_tracker_patch:
        lines.append("Watki:")
        for thread in patch.thread_tracker_patch[:5]:
            prefix = f"{thread.thread_id} / " if thread.thread_id else ""
            lines.append(f"- {prefix}{thread.title}: {thread.change}")
    lines.append(f"Session ID: {sync.session_id}")
    return "\n".join(lines)


def render_proposal_reply(proposal: ChangeProposal) -> str:
    lines = ["Przygotowalem propozycje zmiany.", f"Podsumowanie: {proposal.summary}"]
    if proposal.impacted_docs:
        lines.append("Dokumenty:")
        for doc in proposal.impacted_docs[:5]:
            lines.append(f"- {doc.folder} / {doc.title}")
    if proposal.proposal_id:
        lines.append(f"Proposal ID: {proposal.proposal_id}")
    if proposal.needs_confirmation:
        lines.append("Ta zmiana wymaga akceptacji przed apply.")
    return "\n".join(lines)


def default_output_title(
    kind: Literal["answer", "proposal", "session_sync"],
    artifact_type: Optional[ArtifactType] = None,
) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if artifact_type == "gm_brief":
        return f"GM Brief - {stamp}"
    if artifact_type == "session_report":
        return f"Session Report - {stamp}"
    if artifact_type == "player_summary":
        return f"Player Summary - {stamp}"
    if artifact_type == "pre_session_brief":
        return f"Pre-Session Brief - {stamp}"
    if artifact_type == "session_hooks":
        return f"Session Hooks - {stamp}"
    if artifact_type == "scene_seed":
        return f"Scene Seed - {stamp}"
    if artifact_type == "npc_brief":
        return f"NPC Brief - {stamp}"
    if artifact_type == "twist_pack":
        return f"Twist Pack - {stamp}"
    if kind == "answer":
        return f"Chat Answer - {stamp}"
    if kind == "proposal":
        return f"Chat Proposal - {stamp}"
    return f"Session Sync - {stamp}"


def save_chat_output(
    *,
    kind: Literal["answer", "proposal", "session_sync"],
    content: str,
    requested_title: Optional[str] = None,
    artifact_type: Optional[ArtifactType] = None,
) -> Optional[WorldDocInfo]:
    if not drive_store_v2:
        return None

    title = (requested_title or "").strip() or default_output_title(kind, artifact_type)
    existing = drive_store_v2.find_doc(folder="08 Outputs", title=title)
    doc_ref = DocumentRef(folder="08 Outputs", title=title, doc_id=existing.doc_id if existing else None)

    if existing and existing.doc_id:
        drive_store_v2.replace_doc(doc_ref, content)
        return existing

    return drive_store_v2.create_doc(
        folder="08 Outputs",
        title=title,
        content=content,
        entity_type=WorldEntityType.output,
    )


def format_exception_message(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return message
    return f"{type(error).__name__} without detail"


def is_storage_quota_error(error: Exception) -> bool:
    return "storagequotaexceeded" in format_exception_message(error).lower()


def resolve_output_rollup_doc() -> Optional[WorldDocInfo]:
    if not drive_store_v2:
        return None

    if OUTPUT_ROLLUP_DOC_ID:
        found = drive_store_v2.find_doc(doc_id=OUTPUT_ROLLUP_DOC_ID)
        if found:
            return found
        title = (OUTPUT_ROLLUP_DOC_TITLE or "Output Rollup").strip() or "Output Rollup"
        return WorldDocInfo(
            folder="08 Outputs",
            title=title,
            doc_id=OUTPUT_ROLLUP_DOC_ID,
            path_hint=f"08 Outputs/{title}",
            entity_type=WorldEntityType.output,
        )

    if OUTPUT_ROLLUP_DOC_TITLE:
        found = drive_store_v2.find_doc(folder="08 Outputs", title=OUTPUT_ROLLUP_DOC_TITLE)
        if found:
            return found

    return None


def save_to_output_rollup(content: str) -> Optional[WorldDocInfo]:
    fallback_doc = resolve_output_rollup_doc()
    if not fallback_doc:
        return None

    doc_ref = DocumentRef(
        folder=fallback_doc.folder,
        title=fallback_doc.title,
        doc_id=fallback_doc.doc_id,
        path_hint=fallback_doc.path_hint,
    )
    if OUTPUT_ROLLUP_MODE == "append":
        drive_store_v2.append_doc(doc_ref, content)
    else:
        drive_store_v2.replace_doc(doc_ref, content)
    return fallback_doc


def try_save_chat_output(
    *,
    kind: Literal["answer", "proposal", "session_sync"],
    content: str,
    requested_title: Optional[str] = None,
    artifact_type: Optional[ArtifactType] = None,
) -> tuple[Optional[WorldDocInfo], Optional[str]]:
    try:
        return (
            save_chat_output(
                kind=kind,
                content=content,
                requested_title=requested_title,
                artifact_type=artifact_type,
            ),
            None,
        )
    except Exception as e:
        if is_storage_quota_error(e):
            try:
                fallback_doc = save_to_output_rollup(content)
            except Exception as fallback_error:
                detail = format_exception_message(fallback_error)
                warning = f"Nie udalo sie zapisac outputu do Google Docs: {detail}"
                return None, warning
            if fallback_doc:
                warning = f"Quota zablokowala nowy plik; zapisano do fallback dokumentu {fallback_doc.path_hint}."
                return fallback_doc, warning
        detail = format_exception_message(e)
        warning = f"Nie udalo sie zapisac outputu do Google Docs: {detail}"
        return None, warning


def render_gm_brief(
    *,
    message: str,
    kind: Literal["answer", "proposal", "session_sync"],
    reply: str,
    references: Optional[List[str]] = None,
    proposal: Optional[ChangeProposal] = None,
    patch: Optional[SessionPatch] = None,
    source_title: Optional[str] = None,
    proposal_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> str:
    lines = ["# GM Brief", ""]
    lines.extend(["## Input", message.strip(), "", "## Result", reply.strip()])

    if proposal_id:
        lines.extend(["", "## Proposal", f"Proposal ID: {proposal_id}"])
    if session_id:
        lines.extend(["", "## Session Sync", f"Session ID: {session_id}"])
    if source_title:
        lines.extend(["", "## Source", source_title])
    if proposal and proposal.impacted_docs:
        lines.append("")
        lines.append("## Impacted Documents")
        for doc in proposal.impacted_docs[:10]:
            lines.append(f"- {doc.folder} / {doc.title}")
    if patch and patch.thread_tracker_patch:
        lines.append("")
        lines.append("## Threads")
        for thread in patch.thread_tracker_patch[:10]:
            prefix = f"{thread.thread_id} / " if thread.thread_id else ""
            lines.append(f"- {prefix}{thread.title}: {thread.change}")
    if patch and patch.entities_patch:
        lines.append("")
        lines.append("## Entities")
        for entity in patch.entities_patch[:10]:
            lines.append(f"- {entity.kind}: {entity.name} - {entity.description}")
    if references:
        lines.append("")
        lines.append("## Sources")
        for ref in references[:10]:
            lines.append(f"- {ref}")
    return "\n".join(lines).strip()


def suggest_doc_followups(patch: Optional[SessionPatch]) -> List[str]:
    if not patch:
        return []

    followups: List[str] = []
    seen = set()
    folder_map = {
        "npc": "03 NPC",
        "location": "04 Locations",
        "faction": "05 Factions",
        "item": "00 Admin",
        "other": "00 Admin",
    }

    for entity in patch.entities_patch:
        folder = folder_map.get(entity.kind, "00 Admin")
        item = f"Przejrzyj dokument {folder} / {entity.name}."
        if item not in seen:
            seen.add(item)
            followups.append(item)

    for thread in patch.thread_tracker_patch:
        title = thread.thread_id or thread.title
        item = f"Zweryfikuj wpis watku {title} w Thread Tracker."
        if item not in seen:
            seen.add(item)
            followups.append(item)

    return followups[:8]


def suggest_next_session_prep(patch: Optional[SessionPatch]) -> List[str]:
    if not patch:
        return []

    prep: List[str] = []
    for thread in patch.thread_tracker_patch[:3]:
        prep.append(f"Przygotuj scene pokazujaca konsekwencje watku {thread.title}.")
    for entity in patch.entities_patch[:2]:
        prep.append(f"Zdecyduj, jak {entity.name} zareaguje na nowe informacje.")
    if patch.rag_additions:
        prep.append("Sprawdz, czy nowe fakty sa juz odzwierciedlone w dokumentach swiata.")
    return prep[:6]


def render_session_report(
    *,
    message: str,
    reply: str,
    patch: Optional[SessionPatch] = None,
    source_title: Optional[str] = None,
    session_id: Optional[int] = None,
) -> str:
    lines = ["# Session Report", ""]
    if source_title:
        lines.extend(["## Source", source_title, ""])
    if session_id:
        lines.extend(["## Session ID", str(session_id), ""])

    if patch:
        lines.extend(["## Executive Summary", patch.session_summary, ""])
        if patch.entities_patch:
            lines.append("## World Changes")
            for entity in patch.entities_patch[:10]:
                lines.append(f"- {entity.kind}: {entity.name} - {entity.description}")
            lines.append("")
        if patch.thread_tracker_patch:
            lines.append("## Threads")
            for thread in patch.thread_tracker_patch[:10]:
                prefix = f"{thread.thread_id} / " if thread.thread_id else ""
                lines.append(f"- {prefix}{thread.title}: {thread.change}")
            lines.append("")
        if patch.rag_additions:
            lines.append("## Facts For Retrieval")
            for item in patch.rag_additions[:10]:
                lines.append(f"- {item}")
            lines.append("")
        followups = suggest_doc_followups(patch)
        if followups:
            lines.append("## Suggested Document Follow-ups")
            for item in followups:
                lines.append(f"- {item}")
            lines.append("")
        prep = suggest_next_session_prep(patch)
        if prep:
            lines.append("## Prep For Next Session")
            for item in prep:
                lines.append(f"- {item}")
    else:
        lines.extend(["## Input Notes", message.strip(), "", "## Result", reply.strip()])

    return "\n".join(lines).strip()


def render_player_summary(
    *,
    message: str,
    reply: str,
    source_title: Optional[str] = None,
) -> str:
    lines = ["# Player Summary", "", "## Summary", reply.strip()]
    if source_title:
        lines.extend(["", "## Source", source_title])
    lines.extend(["", "## Note", "Wymaga przegladu MG przed udostepnieniem graczom."])
    return "\n".join(lines).strip()


def build_recent_sessions_context(limit: int = 5) -> str:
    if not world_model_store_v2:
        return "No recent sessions available."
    try:
        sessions = world_model_store_v2.list_sessions(limit=limit)
    except Exception:
        return "Recent sessions unavailable."

    if not sessions:
        return "No recent sessions available."

    lines = ["RECENT SESSIONS:"]
    for session in sessions:
        lines.append(
            f"- session_id={session.id} | source={session.source_title or 'n/a'} | summary={session.session_summary}"
        )
    return "\n".join(lines)


def render_pre_session_brief_placeholder() -> str:
    return """
# Pre-Session Brief

## Campaign State
Do doprecyzowania.

## Active Threads
- Do doprecyzowania.

## Key NPCs and Factions
- Do doprecyzowania.

## Risks and Pressure Points
- Do doprecyzowania.

## Scene Opportunities
- Do doprecyzowania.

## Prep Checklist
- Do doprecyzowania.
""".strip()


def build_chat_artifact(
    *,
    artifact_type: ArtifactType,
    kind: Literal["answer", "proposal", "session_sync"],
    message: str,
    reply: str,
    references: Optional[List[str]] = None,
    proposal: Optional[ChangeProposal] = None,
    patch: Optional[SessionPatch] = None,
    source_title: Optional[str] = None,
    proposal_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> str:
    if artifact_type == "gm_brief":
        return render_gm_brief(
            message=message,
            kind=kind,
            reply=reply,
            references=references,
            proposal=proposal,
            patch=patch,
            source_title=source_title,
            proposal_id=proposal_id,
            session_id=session_id,
        )
    if artifact_type == "session_report":
        return render_session_report(
            message=message,
            reply=reply,
            patch=patch,
            source_title=source_title,
            session_id=session_id,
        )
    if artifact_type == "pre_session_brief":
        return render_pre_session_brief_placeholder()
    return render_player_summary(
        message=message,
        reply=reply,
        source_title=source_title,
    )


def build_creative_artifact_sections(artifact_type: ArtifactType) -> str:
    if artifact_type == "pre_session_brief":
        return """
# Pre-Session Brief

## Campaign State

## Active Threads

## Key NPCs and Factions

## Risks and Pressure Points

## Scene Opportunities

## Prep Checklist
""".strip()
    if artifact_type == "session_hooks":
        return """
Tytul:

Hook 1:

Hook 2:

Hook 3:

Stawki:

Co przygotowac:
""".strip()
    if artifact_type == "scene_seed":
        return """
Tytul sceny:

Cel sceny:

Miejsce:

Zaangazowane postacie:

Przebieg:

Komplikacja:

Mozliwe skutki:
""".strip()
    if artifact_type == "npc_brief":
        return """
Imie:

Rola w kampanii:

Pierwsze wrazenie:

Motywacja:

Sekret:

Relacje:

Jak uzyc tej postaci na sesji:
""".strip()
    return """
Twist 1:

Twist 2:

Twist 3:

Foreshadowing:

Ryzyko dla kampanii:
""".strip()


def artifact_required_markers(artifact_type: ArtifactType) -> List[str]:
    if artifact_type == "pre_session_brief":
        return [
            "# Pre-Session Brief",
            "## Campaign State",
            "## Active Threads",
            "## Key NPCs and Factions",
            "## Risks and Pressure Points",
            "## Scene Opportunities",
            "## Prep Checklist",
        ]
    if artifact_type == "session_hooks":
        return ["Tytul:", "Hook 1:", "Hook 2:", "Hook 3:", "Stawki:", "Co przygotowac:"]
    if artifact_type == "scene_seed":
        return [
            "Tytul sceny:",
            "Cel sceny:",
            "Miejsce:",
            "Zaangazowane postacie:",
            "Przebieg:",
            "Komplikacja:",
            "Mozliwe skutki:",
        ]
    if artifact_type == "npc_brief":
        return [
            "Imie:",
            "Rola w kampanii:",
            "Pierwsze wrazenie:",
            "Motywacja:",
            "Sekret:",
            "Relacje:",
            "Jak uzyc tej postaci na sesji:",
        ]
    if artifact_type == "twist_pack":
        return ["Twist 1:", "Twist 2:", "Twist 3:", "Foreshadowing:", "Ryzyko dla kampanii:"]
    return []


def artifact_style_guidance(artifact_type: ArtifactType) -> str:
    if artifact_type == "pre_session_brief":
        return (
            "- Kazda sekcja ma byc zwiezla i praktyczna.\n"
            "- Uzywaj 2-5 bulletow na sekcje, zamiast dlugich akapitow.\n"
            '- W "Scene Opportunities" dawaj konkretne sceny do zagrania.\n'
            '- W "Prep Checklist" dawaj konkretne rzeczy do przygotowania przez MG.'
        )
    if artifact_type == "session_hooks":
        return (
            "- Daj dokladnie trzy rozne hooki.\n"
            "- Kazdy hook: 2-4 zdania.\n"
            "- Sekcje 'Stawki' i 'Co przygotowac' wypelnij krotszymi bulletami."
        )
    if artifact_type == "scene_seed":
        return "- Kazda sekcja ma byc krotka, konkretna i gotowa do uzycia na sesji."
    if artifact_type == "npc_brief":
        return (
            "- Kazda sekcja ma byc konkretna i zwiezla.\n"
            "- Nie pisz dlugich blokow tekstu; zwykle 2-4 zdania na sekcje.\n"
            "- Sekcje 'Relacje' i 'Jak uzyc tej postaci na sesji' moga byc listami."
        )
    if artifact_type == "twist_pack":
        return "- Kazdy twist ma byc odrebny, konkretny i opisany w 2-4 zdaniach."
    return "- Pisz zwiezle i praktycznie."


def extract_artifact_sections(text: str, artifact_type: ArtifactType) -> Dict[str, str]:
    markers = artifact_required_markers(artifact_type)
    sections: Dict[str, str] = {}
    current_marker: Optional[str] = None
    buffer: List[str] = []

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        matched_marker: Optional[str] = None
        inline_value = ""

        for marker in markers:
            if stripped == marker:
                matched_marker = marker
                break
            if not marker.startswith("#") and stripped.startswith(marker):
                matched_marker = marker
                inline_value = stripped[len(marker):].strip()
                break

        if matched_marker:
            if current_marker is not None:
                sections[current_marker] = "\n".join(buffer).strip()
            current_marker = matched_marker
            buffer = [inline_value] if inline_value else []
            continue

        if current_marker is not None:
            buffer.append(line)

    if current_marker is not None:
        sections[current_marker] = "\n".join(buffer).strip()
    return sections


def normalize_section_body(content: str) -> str:
    normalized = re.sub(r"^[\-\*\d\.\)\s]+", "", (content or "").strip(), flags=re.MULTILINE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def bullet_count(content: str) -> int:
    return sum(1 for line in (content or "").splitlines() if re.match(r"^\s*[\*\-]\s+", line))


def ends_with_sentence_punctuation(content: str) -> bool:
    return bool(re.search(r"[.!?\)\]\"']\s*$", (content or "").strip()))


def sentence_count(content: str) -> int:
    return len(re.findall(r"[.!?](?=(?:[\"')\]]*)?(?:\s|$))", (content or "").strip()))


def extract_bullet_items(content: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []

    for raw_line in (content or "").splitlines():
        line = raw_line.rstrip()
        bullet_match = re.match(r"^\s*[\*\-]\s+(.*)$", line)
        if bullet_match:
            if current:
                items.append(" ".join(part for part in current if part).strip())
            current = [bullet_match.group(1).strip()]
            continue
        if current and line.strip():
            current.append(line.strip())

    if current:
        items.append(" ".join(part for part in current if part).strip())
    return [item for item in items if item]


def complete_bullet_items(content: str, minimum_length: int = 8) -> List[str]:
    items: List[str] = []
    for item in extract_bullet_items(content):
        normalized = normalize_section_body(item)
        if (
            len(normalized) >= minimum_length
            and "do doprecyzowania" not in normalized.lower()
            and ends_with_sentence_punctuation(item)
        ):
            items.append(re.sub(r"\s+", " ", item).strip())
    return items


def complete_bullet_count(content: str, minimum_length: int = 8) -> int:
    return len(complete_bullet_items(content, minimum_length=minimum_length))


def trim_to_complete_sentences(content: str) -> str:
    raw = (content or "").strip()
    if not raw or ends_with_sentence_punctuation(raw):
        return raw

    matches = list(re.finditer(r"[.!?](?=(?:[\"')\]]*)?(?:\s|$))", raw))
    if not matches:
        return raw

    trimmed = raw[:matches[-1].end()].strip()
    if len(normalize_section_body(trimmed)) >= max(20, len(normalize_section_body(raw)) // 2):
        return trimmed
    return raw


def normalize_single_line_section(content: str) -> str:
    lines = [re.sub(r"^\s*[\*\-]\s+", "", line).strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return ""
    first_line = lines[0]
    first_line = re.sub(r"\s+", " ", first_line).strip()
    sentence_parts = re.split(r"(?<=[.!?])\s+", first_line)
    return sentence_parts[0].strip() if sentence_parts else first_line


def sanitize_generated_section(marker: str, content: str) -> str:
    raw = strip_section_marker(content, marker)
    if marker in {"Tytul:", "Tytul sceny:", "Imie:"}:
        return normalize_single_line_section(raw)
    if marker in {"Stawki:", "Co przygotowac:", "Relacje:", "Jak uzyc tej postaci na sesji:"}:
        items = [trim_to_complete_sentences(item) for item in extract_bullet_items(raw)]
        items = complete_bullet_items("\n".join(f"* {item}" for item in items if item.strip()))
        items = [re.sub(r"\s+", " ", item).strip() for item in items if item.strip()]
        return "\n".join(f"* {item}" for item in items)
    if marker.startswith("## "):
        items = [trim_to_complete_sentences(item) for item in extract_bullet_items(raw)]
        items = complete_bullet_items("\n".join(f"* {item}" for item in items if item.strip()))
        items = [re.sub(r"\s+", " ", item).strip() for item in items if item.strip()]
        return "\n".join(f"* {item}" for item in items)
    return trim_to_complete_sentences(raw)


def is_bullet_section_marker(artifact_type: ArtifactType, marker: str) -> bool:
    if marker in {"Stawki:", "Co przygotowac:", "Relacje:", "Jak uzyc tej postaci na sesji:"}:
        return True
    return artifact_type == "pre_session_brief" and marker.startswith("## ")


def section_target_bullet_count(artifact_type: ArtifactType, marker: str) -> int:
    if artifact_type == "session_hooks" and marker in {"Stawki:", "Co przygotowac:"}:
        return 4
    if artifact_type == "npc_brief" and marker in {"Relacje:", "Jak uzyc tej postaci na sesji:"}:
        return 3
    if artifact_type == "pre_session_brief" and marker.startswith("## "):
        return 3
    return 2


def compact_retry_rule(artifact_type: ArtifactType, marker: str) -> str:
    if artifact_type == "session_hooks" and marker in {"Hook 1:", "Hook 2:", "Hook 3:"}:
        return "Zwroc dokladnie 2 krotkie pelne zdania. Kazde zdanie ma byc konkretne i zakonczone kropka."
    if artifact_type == "npc_brief" and marker in {
        "Rola w kampanii:",
        "Pierwsze wrazenie:",
        "Motywacja:",
        "Sekret:",
    }:
        return "Zwroc dokladnie 2 krotkie pelne zdania. Kazde zdanie zakoncz kropka."
    return section_retry_rule(artifact_type, marker)


def section_min_length(artifact_type: ArtifactType, marker: str) -> int:
    if artifact_type == "pre_session_brief":
        return 60 if marker == "## Campaign State" else 35
    if artifact_type == "session_hooks":
        if marker == "Tytul:":
            return 8
        if marker in {"Stawki:", "Co przygotowac:"}:
            return 35
        return 50
    if artifact_type == "scene_seed":
        return 25
    if artifact_type == "npc_brief":
        return 3 if marker == "Imie:" else 45
    if artifact_type == "twist_pack":
        return 40
    return 20


def section_needs_fill(
    *,
    artifact_type: ArtifactType,
    marker: str,
    content: str,
    is_last_marker: bool,
) -> bool:
    raw = (content or "").strip()
    normalized = normalize_section_body(content)
    lowered = normalized.lower()
    if not normalized:
        return True
    if "do doprecyzowania" in lowered:
        return True
    if artifact_type == "session_hooks":
        if marker == "Tytul:":
            return "\n" in raw or len(raw.split()) > 12
        if marker in {"Hook 1:", "Hook 2:", "Hook 3:"}:
            return (
                len(normalized) < section_min_length(artifact_type, marker)
                or not ends_with_sentence_punctuation(raw)
                or sentence_count(raw) < 2
            )
        if marker in {"Stawki:", "Co przygotowac:"}:
            return complete_bullet_count(raw) < 2
    if artifact_type == "npc_brief":
        if marker == "Imie:":
            return "\n" in raw or len(normalized) < 2
        if marker in {"Relacje:", "Jak uzyc tej postaci na sesji:"}:
            return complete_bullet_count(raw) < 2
        return len(normalized) < section_min_length(artifact_type, marker) or not ends_with_sentence_punctuation(raw)
    if artifact_type == "pre_session_brief" and marker.startswith("## "):
        return complete_bullet_count(raw) < 2
    if len(normalized) < section_min_length(artifact_type, marker):
        return True
    if is_last_marker and artifact_type in {"pre_session_brief", "npc_brief"}:
        if len(normalized) >= section_min_length(artifact_type, marker) and not re.search(r"[.!?)]$", normalized):
            return True
    return False


def markers_requiring_fill(text: str, artifact_type: ArtifactType) -> List[str]:
    sections = extract_artifact_sections(text, artifact_type)
    markers = artifact_required_markers(artifact_type)
    required: List[str] = []
    for idx, marker in enumerate(markers):
        if marker.startswith("# ") and not marker.startswith("## "):
            if marker not in sections:
                required.append(marker)
            continue
        content = sections.get(marker, "")
        if section_needs_fill(
            artifact_type=artifact_type,
            marker=marker,
            content=content,
            is_last_marker=idx == len(markers) - 1,
        ):
            required.append(marker)
    return required


def section_retry_rule(artifact_type: ArtifactType, marker: str) -> str:
    if artifact_type == "session_hooks":
        if marker == "Tytul:":
            return "Zwroc dokladnie jeden krotki wiersz tytulu. Bez listy, bez lamanych linii."
        if marker in {"Hook 1:", "Hook 2:", "Hook 3:"}:
            return "Zwroc 2-4 pelne zdania i zakoncz sekcje pelnym zdaniem z kropka, pytajnikiem albo wykrzyknikiem."
        if marker in {"Stawki:", "Co przygotowac:"}:
            return "Zwroc co najmniej 2 osobne bullety zaczynajace sie od '* '. Kazdy bullet zakoncz pelnym zdaniem."
    if artifact_type == "npc_brief":
        if marker == "Imie:":
            return "Zwroc tylko imie albo imie i nazwisko w jednym wierszu."
        if marker in {"Relacje:", "Jak uzyc tej postaci na sesji:"}:
            return "Zwroc co najmniej 2 osobne bullety zaczynajace sie od '* '. Kazdy bullet zakoncz pelnym zdaniem."
        return "Zwroc 2-4 pelne zdania i zakoncz sekcje pelnym zdaniem."
    if artifact_type == "pre_session_brief" and marker.startswith("## "):
        return "Zwroc co najmniej 2 osobne bullety zaczynajace sie od '* '. Kazdy bullet zakoncz pelnym zdaniem."
    return "Uzupelnij sekcje pelna i konkretna trescia."


def missing_artifact_markers(text: str, artifact_type: ArtifactType) -> List[str]:
    lowered = (text or "").lower()
    return [marker for marker in artifact_required_markers(artifact_type) if marker.lower() not in lowered]


def fill_missing_artifact_sections(
    *,
    artifact_type: ArtifactType,
    partial_text: str,
    repair_context: str,
    missing_markers: List[str],
) -> str:
    if not missing_markers:
        return ""

    fill_prompt = f"""
Uzupelnij brakujace sekcje artefaktu tekstowego. Zwracaj tylko brakujace sekcje.
Nie zwracaj JSON. Nie zwracaj code fence. Wszystkie tresci maja byc po polsku.

BRAKUJACE SEKCJE:
{chr(10).join(f"- {marker}" for marker in missing_markers)}

WYMAGANY FORMAT DLA TYCH SEKCJI:
{chr(10).join(marker for marker in missing_markers)}

ZASADY STYLU:
{artifact_style_guidance(artifact_type)}

ISTNIEJACY ARTEFAKT:
{partial_text or '[empty]'}

KONTEKST:
{repair_context}

ZWROC TYLKO BRAKUJACE SEKCJE:
""".strip()

    try:
        return gemini_generate(
            fill_prompt,
            response_mime_type="text/plain",
            temperature=0.4,
            max_output_tokens=2000,
        ).strip()
    except Exception:
        return ""


def render_artifact_section_block(marker: str, content: str) -> List[str]:
    body = (content or "").strip()
    lines: List[str] = []
    if marker.startswith("# ") and not marker.startswith("## "):
        lines.append(marker)
        return lines

    if marker.startswith("#"):
        lines.append(marker)
        lines.append("")
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append("- Do doprecyzowania.")
        return lines

    if marker in {"Tytul:", "Tytul sceny:", "Imie:"} and body and "\n" not in body and not body.lstrip().startswith(("*", "-")):
        lines.append(f"{marker} {body}".rstrip())
        return lines

    lines.append(marker)
    if body:
        lines.extend(body.splitlines())
    else:
        lines.append("Do doprecyzowania.")
    return lines


def merge_artifact_sections(base_text: str, supplement_text: str, artifact_type: ArtifactType) -> str:
    base_sections = extract_artifact_sections(base_text, artifact_type)
    supplement_sections = extract_artifact_sections(supplement_text, artifact_type)
    merged = {**base_sections, **supplement_sections}

    lines: List[str] = []
    for marker in artifact_required_markers(artifact_type):
        if lines:
            lines.append("")
        lines.extend(render_artifact_section_block(marker, merged.get(marker, "")))
    return "\n".join(lines).strip()


def build_placeholder_sections(artifact_type: ArtifactType, markers: List[str]) -> str:
    lines: List[str] = []
    for marker in markers:
        if lines:
            lines.append("")
        if marker.startswith("#"):
            lines.extend([marker, "", "- Do doprecyzowania."])
        else:
            lines.extend([marker, "Do doprecyzowania."])
    return "\n".join(lines).strip()


def creative_section_specs(artifact_type: ArtifactType) -> List[Dict[str, Any]]:
    if artifact_type == "pre_session_brief":
        return [
            {
                "marker": "## Campaign State",
                "instruction": "Daj 3 krotkie bullety o aktualnym stanie kampanii i ostatnim zwrocie sytuacji.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Active Threads",
                "instruction": "Daj 3 krotkie bullety o najwazniejszych aktywnych watkach i ich aktualnym stanie.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Key NPCs and Factions",
                "instruction": "Daj 3 krotkie bullety o kluczowych NPC i frakcjach istotnych przed kolejna sesja.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Risks and Pressure Points",
                "instruction": "Daj 3 krotkie bullety o ryzykach, presji i potencjalnej eskalacji.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Scene Opportunities",
                "instruction": "Daj 3 krotkie bullety z konkretnymi scenami do rozegrania na sesji.",
                "require_canonical_name": True,
            },
            {
                "marker": "## Prep Checklist",
                "instruction": "Daj 3 konkretne bullety rzeczy do przygotowania przez MG przed sesja.",
                "require_canonical_name": True,
            },
        ]
    if artifact_type == "session_hooks":
        return [
            {"marker": "Tytul:", "instruction": "Jedna krotka linia, 4-8 slow.", "require_canonical_name": False},
            {
                "marker": "Hook 1:",
                "instruction": "Napisz 2-4 zdania. To ma byc konkretny incydent otwierajacy sesje.",
                "require_canonical_name": True,
            },
            {
                "marker": "Hook 2:",
                "instruction": "Napisz 2-4 zdania. Ten hook ma byc wyraznie inny od poprzedniego.",
                "require_canonical_name": True,
            },
            {
                "marker": "Hook 3:",
                "instruction": "Napisz 2-4 zdania. Ten hook ma byc wyraznie inny od poprzednich.",
                "require_canonical_name": True,
            },
            {
                "marker": "Stawki:",
                "instruction": "Daj 4-6 krotkich bulletow zaczynajacych sie od '* '.",
                "require_canonical_name": False,
            },
            {
                "marker": "Co przygotowac:",
                "instruction": "Daj 4-6 krotkich bulletow zaczynajacych sie od '* '.",
                "require_canonical_name": False,
            },
        ]
    if artifact_type == "npc_brief":
        return [
            {"marker": "Imie:", "instruction": "Podaj imie albo imie i nazwisko nowej postaci.", "require_canonical_name": False},
            {
                "marker": "Rola w kampanii:",
                "instruction": "Napisz 2-4 zdania o roli tej postaci w kampanii.",
                "require_canonical_name": True,
            },
            {
                "marker": "Pierwsze wrazenie:",
                "instruction": "Napisz 2-4 zdania o wygladzie i aurze postaci.",
                "require_canonical_name": False,
            },
            {
                "marker": "Motywacja:",
                "instruction": "Napisz 2-4 zdania o motywacji i presji tej postaci.",
                "require_canonical_name": False,
            },
            {
                "marker": "Sekret:",
                "instruction": "Napisz 2-4 zdania o sekrecie lub ukrytym koszcie tej postaci.",
                "require_canonical_name": True,
            },
            {
                "marker": "Relacje:",
                "instruction": "Daj 3-5 bulletow zaczynajacych sie od '* ' o relacjach postaci.",
                "require_canonical_name": True,
            },
            {
                "marker": "Jak uzyc tej postaci na sesji:",
                "instruction": "Daj 3-5 bulletow zaczynajacych sie od '* ' z praktycznymi sposobami uzycia.",
                "require_canonical_name": True,
            },
        ]
    return []


def render_partial_artifact_sections(section_values: Dict[str, str], artifact_type: ArtifactType) -> str:
    lines: List[str] = []
    for marker in artifact_required_markers(artifact_type):
        if marker not in section_values:
            continue
        if lines:
            lines.append("")
        lines.extend(render_artifact_section_block(marker, section_values[marker]))
    return "\n".join(lines).strip()


def strip_section_marker(text: str, marker: str) -> str:
    stripped = (text or "").strip()
    if stripped.lower().startswith(marker.lower()):
        stripped = stripped[len(marker):].lstrip()
    return stripped.strip()


def generate_section_candidate(
    *,
    artifact_type: ArtifactType,
    marker: str,
    instruction: str,
    message: str,
    world_context: str,
    structured_context: str,
    recent_sessions_context: str,
    canonical_names: List[str],
    prior_sections_text: str,
    require_canonical_name: bool,
) -> str:
    canonical_context = build_canonical_names_context(canonical_names)
    markers = artifact_required_markers(artifact_type)
    is_last_marker = marker == markers[-1]
    attempts: List[Dict[str, Any]] = []
    prompt = f"""
Jestes wspolautorem kampanii RPG "Krew Na Gwiazdach". Pisz po polsku.

Masz wygenerowac tylko tresc jednej sekcji artefaktu `{artifact_type}`.
Nie zwracaj JSON. Nie zwracaj code fence. Nie zwracaj nazwy sekcji.

SEKCJA:
{marker}

INSTRUKCJA DLA SEKCJI:
{instruction}

ZASADY:
- Zachowaj ton kampanii: smutne heroic fantasy, polityka, emocje, trudne wybory.
- Pisz zwiezle i konkretnie.
- Nie przeczyc twardym faktom z kontekstu.
- Nie tlumacz nazw kanonicznych.

{canonical_context}

AKTUALNY MODEL SWIATA:
{structured_context}

OSTATNIE SESJE:
{recent_sessions_context}

RELEWANTNY KONTEKST KAMPANII:
{world_context}

PROSBA UZYTKOWNIKA:
{message}

JUZ WYGNEROWANE SEKCJE:
{prior_sections_text or '[brak]'}

ZWROC TYLKO TRESC SEKCJI:
""".strip()

    def run_prompt(extra_rule: Optional[str] = None, *, temperature: float = 0.45) -> str:
        effective_prompt = prompt
        if extra_rule:
            effective_prompt = f"{prompt}\n\nDODATKOWA REGULA:\n{extra_rule}"
        return sanitize_generated_section(
            marker,
            gemini_generate(
                effective_prompt,
                response_mime_type="text/plain",
                temperature=temperature,
                max_output_tokens=900,
                telemetry_label=f"section:{artifact_type}:{marker}",
            ).strip(),
        )

    try:
        candidate = run_prompt()
        attempts.append(
            {
                "stage": "initial",
                "chars": len(candidate),
                "sentences": sentence_count(candidate),
                "complete_bullets": complete_bullet_count(candidate),
            }
        )
    except Exception:
        candidate = ""
        attempts.append({"stage": "initial", "error": True})

    needs_retry = section_needs_fill(
        artifact_type=artifact_type,
        marker=marker,
        content=candidate,
        is_last_marker=is_last_marker,
    )
    missing_name = require_canonical_name and canonical_names and not any(name in candidate for name in canonical_names)
    if needs_retry or missing_name:
        retry_rule = section_retry_rule(artifact_type, marker)
        if require_canonical_name and canonical_names:
            retry_rule += " Uzyj co najmniej jednej z tych nazw kanonicznych dokladnie: " + ", ".join(canonical_names) + "."
        try:
            retry_candidate = run_prompt(retry_rule, temperature=0.35)
            if retry_candidate:
                candidate = retry_candidate
            attempts.append(
                {
                    "stage": "retry",
                    "chars": len(candidate),
                    "sentences": sentence_count(candidate),
                    "complete_bullets": complete_bullet_count(candidate),
                }
            )
        except Exception:
            attempts.append({"stage": "retry", "error": True})
    if section_needs_fill(
        artifact_type=artifact_type,
        marker=marker,
        content=candidate,
        is_last_marker=is_last_marker,
    ) or (require_canonical_name and canonical_names and not any(name in candidate for name in canonical_names)):
        candidate = repair_creative_section(
            artifact_type=artifact_type,
            marker=marker,
            instruction=instruction,
            message=message,
            world_context=world_context,
            structured_context=structured_context,
            recent_sessions_context=recent_sessions_context,
            canonical_names=canonical_names,
            prior_sections_text=prior_sections_text,
            broken_content=candidate,
            require_canonical_name=require_canonical_name,
        )
        attempts.append(
            {
                "stage": "repair",
                "chars": len(candidate),
                "sentences": sentence_count(candidate),
                "complete_bullets": complete_bullet_count(candidate),
            }
        )
    if section_needs_fill(
        artifact_type=artifact_type,
        marker=marker,
        content=candidate,
        is_last_marker=is_last_marker,
    ):
        compact_rule = compact_retry_rule(artifact_type, marker)
        if require_canonical_name and canonical_names:
            compact_rule += " Uzyj co najmniej jednej z tych nazw kanonicznych dokladnie: " + ", ".join(canonical_names) + "."
        try:
            compact_candidate = run_prompt(compact_rule, temperature=0.25)
            if compact_candidate:
                candidate = compact_candidate
            attempts.append(
                {
                    "stage": "compact",
                    "chars": len(candidate),
                    "sentences": sentence_count(candidate),
                    "complete_bullets": complete_bullet_count(candidate),
                }
            )
        except Exception:
            attempts.append({"stage": "compact", "error": True})
    final_candidate = sanitize_generated_section(marker, candidate).strip()
    record_telemetry(
        "sections",
        {
            "artifact_type": artifact_type,
            "marker": marker,
            "mode": "section_candidate",
            "attempts": attempts,
            "final_chars": len(final_candidate),
            "final_sentences": sentence_count(final_candidate),
            "final_complete_bullets": complete_bullet_count(final_candidate),
            "needs_fill_after_finalize": section_needs_fill(
                artifact_type=artifact_type,
                marker=marker,
                content=final_candidate,
                is_last_marker=is_last_marker,
            ),
            "contains_canonical_name": any(name in final_candidate for name in canonical_names) if canonical_names else False,
            "final_preview": final_candidate[:180],
        },
    )
    return final_candidate


def generate_bullet_item(
    *,
    artifact_type: ArtifactType,
    marker: str,
    instruction: str,
    message: str,
    world_context: str,
    structured_context: str,
    recent_sessions_context: str,
    canonical_names: List[str],
    prior_sections_text: str,
    existing_items: List[str],
    require_canonical_name: bool,
    target_index: int,
) -> str:
    canonical_context = build_canonical_names_context(canonical_names)
    existing_items_text = "\n".join(f"* {item}" for item in existing_items) or "[brak]"
    prompt = f"""
Jestes wspolautorem kampanii RPG "Krew Na Gwiazdach". Pisz po polsku.

Masz wygenerowac dokladnie jeden nowy bullet do sekcji `{marker}` artefaktu `{artifact_type}`.
Nie zwracaj JSON. Nie zwracaj code fence. Nie zwracaj nazwy sekcji. Nie zwracaj numeracji.

SEKCJA:
{marker}

INSTRUKCJA DLA SEKCJI:
{instruction}

CEL:
- To ma byc bullet numer {target_index}.
- Ma byc inny od juz istniejacych bulletow.
- Zwroc tylko tresc jednego bulleta bez prefixu '* '.

ZASADY:
- Daj jedno pelne, domkniete zdanie.
- Pisz konkretnie i praktycznie.
- Nie tlumacz nazw kanonicznych.
- Nie powtarzaj juz istniejacych bulletow.

{canonical_context}

AKTUALNY MODEL SWIATA:
{structured_context}

OSTATNIE SESJE:
{recent_sessions_context}

RELEWANTNY KONTEKST KAMPANII:
{world_context}

PROSBA UZYTKOWNIKA:
{message}

JUZ WYGNEROWANE SEKCJE:
{prior_sections_text or '[brak]'}

ISTNIEJACE BULLETY W TEJ SEKCJI:
{existing_items_text}

ZWROC TYLKO NOWY BULLET:
""".strip()

    def clean_item(text: str) -> str:
        item = strip_section_marker(text, marker)
        item = re.sub(r"^\s*[\*\-]\s+", "", item).strip()
        item = trim_to_complete_sentences(item)
        item = re.sub(r"\s+", " ", item).strip()
        return item

    for attempt in range(3):
        extra_rule = ""
        if attempt == 1:
            extra_rule = "Zwroc krotsze zdanie, zakonczone kropka."
        elif attempt == 2:
            extra_rule = "Zwroc bardzo konkretne jedno zdanie, max 18 slow, zakonczone kropka."
        if require_canonical_name and canonical_names and not any(name in " ".join(existing_items) for name in canonical_names):
            extra_rule += " Uzyj co najmniej jednej z tych nazw kanonicznych dokladnie: " + ", ".join(canonical_names) + "."
        effective_prompt = prompt if not extra_rule else f"{prompt}\n\nDODATKOWA REGULA:\n{extra_rule.strip()}"
        try:
            candidate = clean_item(
                gemini_generate(
                    effective_prompt,
                    response_mime_type="text/plain",
                    temperature=0.3,
                    max_output_tokens=220,
                    telemetry_label=f"bullet_item:{artifact_type}:{marker}:{target_index}",
                ).strip()
            )
        except Exception:
            candidate = ""
        if (
            candidate
            and len(normalize_section_body(candidate)) >= 12
            and ends_with_sentence_punctuation(candidate)
            and candidate not in existing_items
        ):
            return candidate
    return ""


def repair_creative_section(
    *,
    artifact_type: ArtifactType,
    marker: str,
    instruction: str,
    message: str,
    world_context: str,
    structured_context: str,
    recent_sessions_context: str,
    canonical_names: List[str],
    prior_sections_text: str,
    broken_content: str,
    require_canonical_name: bool,
) -> str:
    canonical_context = build_canonical_names_context(canonical_names)
    prompt = f"""
Jestes wspolautorem kampanii RPG "Krew Na Gwiazdach". Pisz po polsku.

Napraw tylko jedna sekcje artefaktu `{artifact_type}`. Obecna wersja jest urwana, za slaba albo ma zly format.
Nie zwracaj JSON. Nie zwracaj code fence. Nie zwracaj nazwy sekcji.

SEKCJA:
{marker}

INSTRUKCJA DLA SEKCJI:
{instruction}

ZASADY:
- Zachowaj ton kampanii: smutne heroic fantasy, polityka, emocje, trudne wybory.
- Pisz zwiezle i konkretnie.
- Nie przeczyc twardym faktom z kontekstu.
- Nie tlumacz nazw kanonicznych.
- Zakoncz sekcje w pelnym, domknietym formacie.

{canonical_context}

AKTUALNY MODEL SWIATA:
{structured_context}

OSTATNIE SESJE:
{recent_sessions_context}

RELEWANTNY KONTEKST KAMPANII:
{world_context}

PROSBA UZYTKOWNIKA:
{message}

JUZ WYGNEROWANE SEKCJE:
{prior_sections_text or '[brak]'}

WADLIWA WERSJA TEJ SEKCJI:
{broken_content or '[empty]'}

DODATKOWA REGULA:
{section_retry_rule(artifact_type, marker)}
{" Uzyj co najmniej jednej z tych nazw kanonicznych dokladnie: " + ", ".join(canonical_names) + "." if require_canonical_name and canonical_names else ""}

ZWROC TYLKO POPRAWIONA TRESC SEKCJI:
""".strip()

    try:
        return sanitize_generated_section(
            marker,
            gemini_generate(
                prompt,
                response_mime_type="text/plain",
                temperature=0.35,
                max_output_tokens=900,
                telemetry_label=f"repair:{artifact_type}:{marker}",
            ).strip(),
        )
    except Exception:
        return sanitize_generated_section(marker, broken_content)


def generate_creative_section(
    *,
    artifact_type: ArtifactType,
    marker: str,
    instruction: str,
    message: str,
    world_context: str,
    structured_context: str,
    recent_sessions_context: str,
    canonical_names: List[str],
    prior_sections_text: str,
    require_canonical_name: bool,
) -> str:
    if is_bullet_section_marker(artifact_type, marker):
        candidate = generate_section_candidate(
            artifact_type=artifact_type,
            marker=marker,
            instruction=instruction,
            message=message,
            world_context=world_context,
            structured_context=structured_context,
            recent_sessions_context=recent_sessions_context,
            canonical_names=canonical_names,
            prior_sections_text=prior_sections_text,
            require_canonical_name=require_canonical_name,
        )
        items = complete_bullet_items(candidate)
        initial_item_count = len(items)
        target_count = section_target_bullet_count(artifact_type, marker)

        while len(items) < target_count:
            next_item = generate_bullet_item(
                artifact_type=artifact_type,
                marker=marker,
                instruction=instruction,
                message=message,
                world_context=world_context,
                structured_context=structured_context,
                recent_sessions_context=recent_sessions_context,
                canonical_names=canonical_names,
                prior_sections_text=prior_sections_text,
                existing_items=items,
                require_canonical_name=require_canonical_name,
                target_index=len(items) + 1,
            )
            if not next_item:
                break
            items.append(next_item)

        if require_canonical_name and canonical_names and not any(name in " ".join(items) for name in canonical_names):
            replacement = generate_bullet_item(
                artifact_type=artifact_type,
                marker=marker,
                instruction=instruction,
                message=message,
                world_context=world_context,
                structured_context=structured_context,
                recent_sessions_context=recent_sessions_context,
                canonical_names=canonical_names,
                prior_sections_text=prior_sections_text,
                existing_items=items[1:] if len(items) > 1 else [],
                require_canonical_name=True,
                target_index=1,
            )
            if replacement:
                if items:
                    items[0] = replacement
                else:
                    items.append(replacement)

        final_section = "\n".join(f"* {item}" for item in items)
        record_telemetry(
            "sections",
            {
                "artifact_type": artifact_type,
                "marker": marker,
                "mode": "bullet_fill",
                "initial_complete_items": initial_item_count,
                "target_items": target_count,
                "final_items": len(items),
                "used_fill_items": max(0, len(items) - initial_item_count),
                "needs_fill_after_finalize": section_needs_fill(
                    artifact_type=artifact_type,
                    marker=marker,
                    content=final_section,
                    is_last_marker=marker == artifact_required_markers(artifact_type)[-1],
                ),
                "final_preview": final_section[:180],
            },
        )
        return final_section

    return generate_section_candidate(
        artifact_type=artifact_type,
        marker=marker,
        instruction=instruction,
        message=message,
        world_context=world_context,
        structured_context=structured_context,
        recent_sessions_context=recent_sessions_context,
        canonical_names=canonical_names,
        prior_sections_text=prior_sections_text,
        require_canonical_name=require_canonical_name,
    )


def generate_structured_creative_artifact(
    *,
    artifact_type: ArtifactType,
    message: str,
    world_context: str,
    structured_context: str,
    recent_sessions_context: str,
    canonical_names: List[str],
) -> str:
    section_values: Dict[str, str] = {}
    if artifact_type == "pre_session_brief":
        section_values["# Pre-Session Brief"] = ""
    specs = creative_section_specs(artifact_type)
    for spec in specs:
        body = generate_creative_section(
            artifact_type=artifact_type,
            marker=spec["marker"],
            instruction=spec["instruction"],
            message=message,
            world_context=world_context,
            structured_context=structured_context,
            recent_sessions_context=recent_sessions_context,
            canonical_names=canonical_names,
            prior_sections_text=render_partial_artifact_sections(section_values, artifact_type),
            require_canonical_name=bool(spec.get("require_canonical_name")),
        )
        section_values[spec["marker"]] = body
    artifact_text = render_partial_artifact_sections(section_values, artifact_type)
    record_telemetry(
        "artifacts",
        {
            "artifact_type": artifact_type,
            "mode": "structured",
            "chars": len(artifact_text),
            "markers_requiring_fill": markers_requiring_fill(artifact_text, artifact_type),
            "preview": artifact_text[:200],
        },
    )
    return artifact_text


def append_missing_artifact_sections(text: str, artifact_type: ArtifactType) -> str:
    missing = missing_artifact_markers(text, artifact_type)
    if not missing:
        return text.strip()

    lines = [text.rstrip()]
    for marker in missing:
        lines.extend(["", marker])
        if marker.startswith("## "):
            lines.append("- Do doprecyzowania.")
        else:
            lines.append("Do doprecyzowania.")
    return "\n".join(lines).strip()


def ensure_artifact_shape(
    *,
    artifact_type: ArtifactType,
    text: str,
    repair_context: str,
) -> str:
    cleaned = (text or "").strip()
    if cleaned and not markers_requiring_fill(cleaned, artifact_type):
        return cleaned

    repair_prompt = f"""
Napraw artefakt tekstowy. Zwracaj tylko finalny artefakt tekstowy.
Nie zwracaj JSON. Nie zwracaj code fence.
Wszystkie tresci maja byc po polsku.
Artefakt musi zawierac wszystkie wymagane znaczniki:
{chr(10).join(f"- {marker}" for marker in artifact_required_markers(artifact_type))}

DOCZELOWY FORMAT:
{build_creative_artifact_sections(artifact_type)}

KONTEKST:
{repair_context}

ZLY ARTEFAKT:
{cleaned or '[empty]'}

POPRAWIONY ARTEFAKT:
""".strip()

    try:
        repaired = gemini_generate(
            repair_prompt,
            response_mime_type="text/plain",
            temperature=0.2,
            max_output_tokens=2500,
        ).strip()
    except Exception:
        repaired = ""

    candidate = repaired or cleaned
    markers_to_fill = markers_requiring_fill(candidate, artifact_type)
    if markers_to_fill:
        filled_sections = fill_missing_artifact_sections(
            artifact_type=artifact_type,
            partial_text=candidate,
            repair_context=repair_context,
            missing_markers=markers_to_fill,
        )
        if filled_sections:
            candidate = merge_artifact_sections(candidate, filled_sections, artifact_type)
    remaining_missing = missing_artifact_markers(candidate, artifact_type)
    if remaining_missing:
        candidate = merge_artifact_sections(
            candidate,
            build_placeholder_sections(artifact_type, remaining_missing),
            artifact_type,
        )
    return append_missing_artifact_sections(candidate, artifact_type)


def build_creative_prompt(
    *,
    message: str,
    artifact_type: ArtifactType,
    world_context: str,
    structured_context: str,
    canonical_names_context: str,
) -> str:
    format_instructions = build_creative_artifact_sections(artifact_type)
    return f"""
Jestes wspolautorem kampanii RPG "Krew Na Gwiazdach". Pisz po polsku.

CEL:
- Mozesz wymyslac nowe rzeczy.
- Musza byc spojne z istniejacym swiatem, tonem kampanii i znanymi faktami.
- Nie wolno przeczyc twardym faktom z kontekstu.
- Jesli czegos nie ma w kontekscie, wolno Ci dopowiedziec tylko tyle, ile jest potrzebne do stworzenia uzytecznego materialu MG.

ZASADY:
1) Nie zwracaj JSON.
2) Nie zwracaj code fence.
3) Zwracaj tylko finalny artefakt tekstowy.
4) Korzystaj z dokladnych nazw encji, watkow i miejsc, jesli sa znane.
5) Zachowaj ton kampanii: smutne heroic fantasy, polityka, emocje, trudne wybory.
6) Uzyj wszystkich wymaganych sekcji z formatu i nie pomijaj zadnej.
7) Jesli format zawiera Hook 1/2/3 albo Twist 1/2/3, wypelnij wszystkie te sekcje.
8) Pisz zwiezle i praktycznie. Unikaj jednego bardzo dlugiego akapitu kosztem pozostalych sekcji.
9) Nie tlumacz nazw kanonicznych i nie zamieniaj ich na synonimy.

WYTYCZNE STYLU:
{artifact_style_guidance(artifact_type)}

{canonical_names_context}

AKTUALNY MODEL SWIATA:
{structured_context}

RELEWANTNY KONTEKST KAMPANII:
{world_context}

PROSBA UZYTKOWNIKA:
{message}

ZWROC DOKLADNIE TEN FORMAT:
{format_instructions}
""".strip()


def build_pre_session_brief_prompt(
    *,
    message: str,
    world_context: str,
    structured_context: str,
    recent_sessions_context: str,
    canonical_names_context: str,
) -> str:
    format_instructions = build_creative_artifact_sections("pre_session_brief")
    return f"""
Jestes asystentem MG kampanii "Krew Na Gwiazdach". Pisz po polsku.

CEL:
- Przygotuj praktyczny briefing przed kolejna sesja.
- Nie wymyslaj twardych faktow sprzecznych z kontekstem.
- Mozesz proponowac sceny, ryzyka i checklisty MG, jesli wynikaja logicznie z kontekstu.

ZASADY:
1) Zwracaj tylko finalny artefakt tekstowy.
2) Nie zwracaj JSON.
3) Nie zwracaj code fence.
4) Uzywaj dokladnych nazw encji, watkow i dokumentow, jesli sa znane.
5) W sekcjach "Scene Opportunities" i "Prep Checklist" dawaj konkretne, praktyczne propozycje MG.
6) Wypelnij wszystkie sekcje z wymaganego formatu.
7) Pisz zwiezle i praktycznie. Lepiej dac krotsze sekcje niz urwac artefakt po pierwszej.
8) Nie tlumacz nazw kanonicznych i nie zamieniaj ich na synonimy.

WYTYCZNE STYLU:
{artifact_style_guidance("pre_session_brief")}

{canonical_names_context}

AKTUALNY MODEL SWIATA:
{structured_context}

OSTATNIE SESJE:
{recent_sessions_context}

RELEWANTNY KONTEKST KAMPANII:
{world_context}

PROSBA UZYTKOWNIKA:
{message}

ZWROC DOKLADNIE TEN FORMAT:
{format_instructions}
""".strip()


def generate_creative_artifact(
    *,
    message: str,
    artifact_type: ArtifactType,
) -> tuple[str, List[str]]:
    structured_context = build_world_model_context(limit=30)
    recent_sessions_context = build_recent_sessions_context(limit=5)
    try:
        hits = vector_search(message, 6)
    except Exception:
        hits = []
    if hits:
        world_context = build_campaign_context(hits)
    else:
        try:
            world_context = build_context_for_planner(drive_store_v2)
        except Exception:
            world_context = "Brak dodatkowego kontekstu kampanii."
    canonical_names = collect_canonical_names(message, hits)
    canonical_names_context = build_canonical_names_context(canonical_names)

    if artifact_type in {"session_hooks", "npc_brief"}:
        artifact_text = generate_structured_creative_artifact(
            artifact_type=artifact_type,
            message=message,
            world_context=world_context,
            structured_context=structured_context,
            recent_sessions_context=recent_sessions_context,
            canonical_names=canonical_names,
        )
        return artifact_text, render_source_labels(hits)

    prompt = build_creative_prompt(
        message=message,
        artifact_type=artifact_type,
        world_context=world_context,
        structured_context=structured_context,
        canonical_names_context=canonical_names_context,
    )
    artifact_text = gemini_generate(
        prompt,
        response_mime_type="text/plain",
        temperature=0.8,
        max_output_tokens=2500,
    ).strip()
    repair_context = "\n\n".join(
        [
            "AKTUALNY MODEL SWIATA:",
            structured_context,
            "",
            "RELEWANTNY KONTEKST KAMPANII:",
            world_context,
            "",
            "NAZWY KANONICZNE:",
            canonical_names_context,
            "",
            "PROSBA UZYTKOWNIKA:",
            message,
        ]
    ).strip()
    artifact_text = ensure_artifact_shape(
        artifact_type=artifact_type,
        text=artifact_text,
        repair_context=repair_context,
    )
    return artifact_text, render_source_labels(hits)


def generate_pre_session_brief(message: str) -> tuple[str, List[str]]:
    structured_context = build_world_model_context(limit=40)
    recent_sessions_context = build_recent_sessions_context(limit=6)
    try:
        hits = vector_search(message, 8)
    except Exception:
        hits = []
    if hits:
        world_context = build_campaign_context(hits)
    else:
        try:
            world_context = build_context_for_planner(drive_store_v2)
        except Exception:
            world_context = "Brak dodatkowego kontekstu kampanii."
    canonical_names = collect_canonical_names(message, hits)

    artifact_text = generate_structured_creative_artifact(
        artifact_type="pre_session_brief",
        message=message,
        world_context=world_context,
        structured_context=structured_context,
        recent_sessions_context=recent_sessions_context,
        canonical_names=canonical_names,
    )
    return artifact_text, render_source_labels(hits)


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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    telemetry_token = start_request_telemetry(req.include_telemetry)
    try:
        def build_chat_response(**kwargs) -> ChatResponse:
            return ChatResponse(
                **kwargs,
                telemetry=current_request_telemetry() if req.include_telemetry else None,
            )

        resolved_artifact_type = infer_artifact_type(req.message, req.artifact_type)
        resolved_intent = req.intent if req.intent != "auto" else detect_chat_intent(req.message)
        if is_creative_artifact_type(resolved_artifact_type):
            resolved_intent = "creative"
        warnings: List[str] = []

        if resolved_artifact_type == "pre_session_brief":
            artifact_text, references = generate_pre_session_brief(req.message)
            output_doc = None
            if req.save_output:
                output_doc, warning = try_save_chat_output(
                    kind="answer",
                    content=artifact_text,
                    requested_title=req.output_title,
                    artifact_type=resolved_artifact_type,
                )
                if warning:
                    warnings.append(warning)
            return build_chat_response(
                kind="answer",
                reply=artifact_text,
                artifact_type=resolved_artifact_type,
                artifact_text=artifact_text,
                references=references,
                warnings=warnings,
                output_doc_id=output_doc.doc_id if output_doc else None,
                output_title=output_doc.title if output_doc else None,
                output_path=output_doc.path_hint if output_doc else None,
            )

        if resolved_intent == "creative":
            creative_artifact_type = resolved_creative_artifact_type(resolved_artifact_type)
            artifact_text, references = generate_creative_artifact(
                message=req.message,
                artifact_type=creative_artifact_type,
            )
            output_doc = None
            if req.save_output:
                output_doc, warning = try_save_chat_output(
                    kind="answer",
                    content=artifact_text,
                    requested_title=req.output_title,
                    artifact_type=creative_artifact_type,
                )
                if warning:
                    warnings.append(warning)
            return build_chat_response(
                kind="creative",
                reply=artifact_text,
                artifact_type=creative_artifact_type,
                artifact_text=artifact_text,
                references=references,
                warnings=warnings,
                output_doc_id=output_doc.doc_id if output_doc else None,
                output_title=output_doc.title if output_doc else None,
                output_path=output_doc.path_hint if output_doc else None,
            )

        if resolved_intent == "answer":
            ask_response = ask(
                AskRequest(
                    question=req.message,
                    include_sources=req.include_sources,
                    mode="auto",
                )
            )
            references = render_source_labels(ask_response.sources)
            reply = ask_response.answer.strip()
            if references:
                reply = reply + "\n\nZrodla:\n" + "\n".join(f"- {label}" for label in references)
            artifact_text = (
                build_chat_artifact(
                    artifact_type=resolved_artifact_type,
                    kind="answer",
                    message=req.message,
                    reply=reply,
                    references=references,
                )
                if resolved_artifact_type
                else None
            )
            output_doc = None
            if req.save_output:
                output_doc, warning = try_save_chat_output(
                    kind="answer",
                    content=artifact_text or reply,
                    requested_title=req.output_title,
                    artifact_type=resolved_artifact_type,
                )
                if warning:
                    warnings.append(warning)
            return build_chat_response(
                kind="answer",
                reply=reply,
                artifact_type=resolved_artifact_type,
                artifact_text=artifact_text,
                references=references,
                warnings=warnings,
                output_doc_id=output_doc.doc_id if output_doc else None,
                output_title=output_doc.title if output_doc else None,
                output_path=output_doc.path_hint if output_doc else None,
            )

        if resolved_intent == "session_sync":
            sync_response = ingest_session_and_sync(
                IngestAndSyncSessionRequest(
                    raw_notes=req.message,
                    source_title=req.source_title,
                )
            )
            reply = render_session_sync_reply(
                sync_response.patch,
                sync_response.sync,
                source_title=req.source_title,
            )
            artifact_text = (
                build_chat_artifact(
                    artifact_type=resolved_artifact_type,
                    kind="session_sync",
                    message=req.message,
                    reply=reply,
                    patch=sync_response.patch,
                    source_title=req.source_title,
                    session_id=sync_response.sync.session_id,
                )
                if resolved_artifact_type
                else None
            )
            output_doc = None
            if req.save_output:
                output_doc, warning = try_save_chat_output(
                    kind="session_sync",
                    content=artifact_text or reply,
                    requested_title=req.output_title,
                    artifact_type=resolved_artifact_type,
                )
                if warning:
                    warnings.append(warning)
            return build_chat_response(
                kind="session_sync",
                reply=reply,
                artifact_type=resolved_artifact_type,
                artifact_text=artifact_text,
                session_id=sync_response.sync.session_id,
                warnings=warnings,
                output_doc_id=output_doc.doc_id if output_doc else None,
                output_title=output_doc.title if output_doc else None,
                output_path=output_doc.path_hint if output_doc else None,
            )

        docs = drive_store_v2.list_world_docs()
        context = build_context_for_planner(drive_store_v2)
        proposal_request = ProposeChangesRequest(instruction=req.message, mode="auto", dry_run=True)
        proposal = planner_v2.propose(request=proposal_request, world_docs=docs, world_context=context)
        proposal_id = workflow_store_v2.save_proposal(proposal_request, proposal)
        if proposal_id is not None:
            proposal = ChangeProposal.model_validate(
                {
                    **proposal.model_dump(mode="json"),
                    "proposal_id": proposal_id,
                }
            )

        reply = render_proposal_reply(proposal)
        artifact_text = (
            build_chat_artifact(
                artifact_type=resolved_artifact_type,
                kind="proposal",
                message=req.message,
                reply=reply,
                proposal=proposal,
                proposal_id=proposal.proposal_id,
            )
            if resolved_artifact_type
            else None
        )
        output_doc = None
        if req.save_output:
            output_doc, warning = try_save_chat_output(
                kind="proposal",
                content=artifact_text or reply,
                requested_title=req.output_title,
                artifact_type=resolved_artifact_type,
            )
            if warning:
                warnings.append(warning)
        return build_chat_response(
            kind="proposal",
            reply=reply,
            artifact_type=resolved_artifact_type,
            artifact_text=artifact_text,
            proposal_id=proposal.proposal_id,
            warnings=warnings,
            output_doc_id=output_doc.doc_id if output_doc else None,
            output_title=output_doc.title if output_doc else None,
            output_path=output_doc.path_hint if output_doc else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=format_exception_message(e))
    finally:
        reset_request_telemetry(telemetry_token)


@app.post("/chat_text", response_class=PlainTextResponse)
def chat_text(req: ChatRequest):
    response = chat(req)
    text = response.artifact_text or response.reply
    if response.output_path:
        text = text + "\n\nZapisano do:\n- " + response.output_path
    if response.warnings:
        text = text + "\n\nUwagi:\n" + "\n".join(f"- {warning}" for warning in response.warnings)
    if response.telemetry:
        text = text + "\n\nTelemetry:\n" + json.dumps(response.telemetry, ensure_ascii=False, indent=2)
    return text + "\n"


def build_world_model_context(limit: int = 50) -> str:
    if not world_model_store_v2:
        return "No structured world model entries available yet."

    try:
        entities = world_model_store_v2.list_entities(limit=limit)
        threads = world_model_store_v2.list_threads(limit=limit)
    except Exception:
        return "World model context unavailable."

    lines: List[str] = []
    if entities:
        lines.append("KNOWN ENTITIES:")
        for entity in entities:
            lines.append(f"- {entity.entity_kind}: {entity.name}")

    if threads:
        lines.append("KNOWN THREADS:")
        for thread in threads:
            prefix = f"{thread.thread_id} | " if thread.thread_id else ""
            status = f" | status={thread.status}" if thread.status else ""
            lines.append(f"- {prefix}{thread.title}{status}")

    return "\n".join(lines) if lines else "No structured world model entries available yet."


def normalize_world_model_key(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def reconcile_session_patch_with_world_model(patch: SessionPatch) -> SessionPatch:
    if not world_model_store_v2:
        return patch

    try:
        known_entities = world_model_store_v2.list_entities(limit=100)
        known_threads = world_model_store_v2.list_threads(limit=100)
    except Exception:
        return patch

    entity_by_name = {normalize_world_model_key(entity.name): entity for entity in known_entities}
    reconciled_entities: List[EntityPatch] = []
    for entity in patch.entities_patch:
        matched = entity_by_name.get(normalize_world_model_key(entity.name))
        if matched:
            reconciled_entities.append(
                EntityPatch(
                    kind=matched.entity_kind,  # type: ignore[arg-type]
                    name=matched.name,
                    description=entity.description,
                    tags=entity.tags,
                )
            )
        else:
            reconciled_entities.append(entity)

    def match_thread(thread: ThreadPatch):
        if thread.thread_id:
            normalized_thread_id = normalize_world_model_key(thread.thread_id)
            for known in known_threads:
                if known.thread_id and normalize_world_model_key(known.thread_id) == normalized_thread_id:
                    return known

        haystack = normalize_world_model_key(" ".join(filter(None, [thread.title, thread.change, thread.status or ""])))
        ranked_matches = []

        if not thread.thread_id:
            for known in known_threads:
                if known.thread_id:
                    candidates = [
                        normalize_world_model_key(known.title),
                        normalize_world_model_key(known.thread_key),
                        normalize_world_model_key(known.thread_id),
                    ]
                    matched_lengths = [len(candidate) for candidate in candidates if candidate and candidate in haystack]
                    if matched_lengths:
                        ranked_matches.append((1000 + max(matched_lengths), known))

        normalized_title = normalize_world_model_key(thread.title)
        for known in known_threads:
            if normalize_world_model_key(known.title) == normalized_title:
                ranked_matches.append((800 if known.thread_id else 500, known))

        for known in known_threads:
            candidates = [
                normalize_world_model_key(known.title),
                normalize_world_model_key(known.thread_key),
                normalize_world_model_key(known.thread_id),
            ]
            matched_lengths = [len(candidate) for candidate in candidates if candidate and candidate in haystack]
            if matched_lengths:
                ranked_matches.append((max(matched_lengths), known))

        if ranked_matches:
            ranked_matches.sort(key=lambda item: item[0], reverse=True)
            return ranked_matches[0][1]
        return None

    reconciled_threads: List[ThreadPatch] = []
    seen_thread_keys = set()
    for thread in patch.thread_tracker_patch:
        matched = match_thread(thread)
        if matched:
            resolved = ThreadPatch(
                thread_id=matched.thread_id or thread.thread_id,
                title=matched.title,
                status=thread.status or matched.status,
                change=thread.change,
            )
        else:
            resolved = thread

        dedupe_key = normalize_world_model_key(resolved.thread_id or resolved.title)
        if dedupe_key in seen_thread_keys:
            continue
        seen_thread_keys.add(dedupe_key)
        reconciled_threads.append(resolved)

    final_threads: List[ThreadPatch] = []
    for thread in reconciled_threads:
        if thread.thread_id:
            final_threads.append(thread)
            continue

        haystack = normalize_world_model_key(" ".join(filter(None, [thread.title, thread.change, thread.status or ""])))
        overlaps_keyed_thread = any(
            existing.thread_id
            and normalize_world_model_key(existing.title) in haystack
            for existing in final_threads
        )
        if overlaps_keyed_thread:
            continue
        final_threads.append(thread)

    return SessionPatch(
        session_summary=patch.session_summary,
        thread_tracker_patch=final_threads,
        entities_patch=reconciled_entities,
        rag_additions=patch.rag_additions,
    )


def generate_session_patch(raw_notes: str) -> SessionPatch:
    cleaned_notes = raw_notes.strip()
    if not cleaned_notes:
        raise HTTPException(status_code=400, detail="raw_notes is empty")

    notes = sanitize_for_rag(cleaned_notes)
    world_model_context = build_world_model_context()

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

RULES:
- Wszystkie pola tekstowe w JSON pisz po polsku.
- If a thread matches an existing known thread, reuse its exact title and exact thread_id when available.
- If an entity matches an existing known entity, reuse its exact name and kind.
- Prefer updating existing threads and entities over inventing duplicates.

CURRENT WORLD MODEL:
{world_model_context}

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
    return reconcile_session_patch_with_world_model(patch)


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
