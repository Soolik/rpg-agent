from __future__ import annotations

import json
import os
import re
import uuid
import google.auth
import psycopg
import requests

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
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

# -------------------------
# Models (API)
# -------------------------

AskMode = Literal["auto", "campaign", "general"]


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

    # wiersze i komórki
    h = re.sub(r"(?is)</tr\s*>", "\n", h)
    h = re.sub(r"(?is)</t[dh]\s*>", " | ", h)

    # usuń tagi
    h = re.sub(r"(?is)<[^>]+>", " ", h)

    # normalizacja whitespace
    h = re.sub(r"[ \t]{2,}", " ", h)

    lines = [ln.strip() for ln in h.splitlines()]
    lines = [ln.strip(" |") for ln in lines if ln and len(ln) > 3]

    norm = []
    for ln in lines:
        ln = ln.replace("\t", " | ")
        ln = re.sub(r"\s*\|\s*", " | ", ln)
        ln = re.sub(r"[ \t]{2,}", " ", ln).strip()

        # wiersz tabeli = co najmniej 3 kolumny (2 kreski)
        if ln.count("|") >= 2:
            norm.append(ln)

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
    return data.decode("utf-8", errors="ignore")


def export_google_doc(drive, doc_id: str, mime_type: str) -> str:
    data = drive.files().export(fileId=doc_id, mimeType=mime_type).execute()
    return data.decode("utf-8", errors="ignore")


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


def upsert_chunks(doc_id: str, doc_type: str, chunks: List[str], embeddings: List[List[float]]):
    """
    Insert-only (tak jak wcześniej). Dla MVP OK.
    Przy reindex warto używać clean=true.
    """
    now = datetime.now(timezone.utc)
    with db_conn() as conn:
        with conn.cursor() as cur:
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
                        doc_id,
                        doc_type,
                        t,
                        emb,
                        json.dumps({"source": doc_type}),
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
                select id, doc_id, doc_type, chunk_text, (embedding <-> %s::vector) as distance
                from chunks
                where campaign_id = %s
                order by embedding <-> %s::vector
                limit %s
                """,
                (q_emb, CAMPAIGN_ID, q_emb, top_k),
            )
            rows = cur.fetchall()
    return [
        {"chunk_id": cid, "doc_id": d, "doc_type": t, "chunk_text": c, "distance": float(dist)}
        for (cid, d, t, c, dist) in rows
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
        require_env("BIBLE_DOC_ID", BIBLE_DOC_ID)
        require_env("RULES_DOC_ID", RULES_DOC_ID)
        require_env("GLOSSARY_DOC_ID", GLOSSARY_DOC_ID)
        require_env("THREADS_DOC_ID", THREADS_DOC_ID)

        drive = get_drive_service()
        docs: List[Tuple[str, str]] = [
            ("bible", BIBLE_DOC_ID),  # type: ignore[arg-type]
            ("rules", RULES_DOC_ID),  # type: ignore[arg-type]
            ("glossary", GLOSSARY_DOC_ID),  # type: ignore[arg-type]
            ("threads", THREADS_DOC_ID),  # type: ignore[arg-type]
        ]

        total_chunks = 0
        for doc_type, doc_id in docs:
            if body.clean:
                delete_chunks_for_doc(doc_id)

            if doc_type == "threads":
                # Prefer HTML export for tables. Fallback to plain text if HTML parse yields nothing.
                raw_html = export_google_doc(drive, doc_id, "text/html")
                raw = html_table_to_tsv(raw_html)
                if not raw.strip():
                    raw = export_google_doc(drive, doc_id, "text/plain")

                # IMPORTANT: do not over-sanitize threads, keep row structure for quoting
                cleaned = raw.strip()
                chunks = chunk_threads(cleaned)

                # Optional guard: avoid indexing empty/near-empty threads
                if len(cleaned) < 50 or not chunks:
                    continue
            else:
                raw = export_google_doc(drive, doc_id, "text/plain")
                cleaned = sanitize_for_rag(raw)
                chunks = chunk_text(cleaned)

                if not chunks:
                    continue

            embs = gemini_embed(chunks)
            upsert_chunks(doc_id=doc_id, doc_type=doc_type, chunks=chunks, embeddings=embs)
            total_chunks += len(chunks)

        return {"ok": True, "indexed_chunks": total_chunks, "clean": body.clean}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        q = req.question.strip()

        if req.mode == "campaign":
            campaign_mode = True
        elif req.mode == "general":
            campaign_mode = False
        else:
            campaign_mode = is_campaign_question(q)

        if not campaign_mode:
            answer = gemini_generate(build_general_prompt(q)).strip()
            return AskResponse(answer=answer, sources=[])

        hits = vector_search(q, req.top_k)
        context = "\n\n".join([f"[{i+1}] ({h['doc_type']}) {h['chunk_text'][:900]}" for i, h in enumerate(hits)])
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


@app.post("/ingest_session", response_model=SessionPatch)
def ingest_session(req: IngestSessionRequest):
    """
    Wklejasz surowe notatki z sesji -> agent generuje patch.
    Niczego nie zapisujemy do Google Docs automatem.
    """
    try:
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


