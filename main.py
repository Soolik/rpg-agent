import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import google.auth
import psycopg
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from googleapiclient.discovery import build
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

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


# -------------------------
# Helpers
# -------------------------
def require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise HTTPException(status_code=400, detail=f"Missing env: {name}")
    return value


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
    Lepsze jest chunkowanie po liniach, żeby nie ucinało wątku typu "T05 | ...".
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # grupujemy po ~40 linii
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
    """
    Embeddings przez embedContent (pojedyncze).
    """
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


def gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Brak GEMINI_API_KEY")

    url = f"https://generativelanguage.googleapis.com/v1beta/{GEN_MODEL}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 2500,
            "responseMimeType": "text/plain",
        },
    }
    r = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini generate error: {r.status_code} {r.text}")

    data = r.json()

    # normalny case: candidates[0].content.parts[].text
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
                select doc_id, doc_type, chunk_text, (embedding <-> %s::vector) as distance
                from chunks
                where campaign_id = %s
                order by embedding <-> %s::vector
                limit %s
                """,
                (q_emb, CAMPAIGN_ID, q_emb, top_k),
            )
            rows = cur.fetchall()

    return [
        {"doc_id": d, "doc_type": t, "chunk_text": c, "distance": float(dist)}
        for (d, t, c, dist) in rows
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


def looks_truncated(answer: str, question: str) -> bool:
    s = (answer or "").strip()
    if not s:
        return True
    if "brak w notatkach" in s.lower():
        return False
    # urwane w pół zdania + sensowna długość
    if s[-1] not in ".!?\n":
        return len(s) > 200 and len(question) > 40
    return False


# -------------------------
# Endpoints
# -------------------------
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
            ("bible", BIBLE_DOC_ID),       # type: ignore[arg-type]
            ("rules", RULES_DOC_ID),       # type: ignore[arg-type]
            ("glossary", GLOSSARY_DOC_ID), # type: ignore[arg-type]
            ("threads", THREADS_DOC_ID),   # type: ignore[arg-type]
        ]

        total_chunks = 0

        for doc_type, doc_id in docs:
            if body.clean:
                delete_chunks_for_doc(doc_id)

            text = export_google_doc_as_text(drive, doc_id)

            if doc_type == "threads":
                chunks = chunk_threads(text)
            else:
                chunks = chunk_text(text)

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

        # wybór trybu
        if req.mode == "campaign":
            campaign_mode = True
        elif req.mode == "general":
            campaign_mode = False
        else:
            campaign_mode = is_campaign_question(q)

        hits: List[Dict[str, Any]] = []
        context = ""

        if campaign_mode:
            hits = vector_search(q, req.top_k)
            # 900 znaków na chunk to sensowny kompromis
            context = "\n\n".join(
                [f"[{i+1}] ({h['doc_type']}) {h['chunk_text'][:900]}" for i, h in enumerate(hits)]
            )

            prompt = f"""
Jesteś asystentem MG kampanii "Krew Na Gwiazdach". Odpowiadasz po polsku.

ZASADY:
- Używaj tylko faktów z KONTEKSTU.
- Jeśli czegoś nie ma w kontekście, napisz dosłownie: "brak w notatkach".
- Odpowiedz dokładnie na PYTANIE użytkownika. Nie dodawaj sekcji, których nie wymaga pytanie.
- Zwracaj odpowiedź w markdown (listy numerowane "1.", "2.", tabele markdown).
- Jeśli odpowiedź się urywa, dokończ ją zamiast kończyć w pół zdania.

KONTEKST:
{context}

PYTANIE:
{q}

ODPOWIEDŹ:
""".strip()
        else:
            # tryb ogólny: zero RAG, zero "brak w notatkach"
            prompt = f"""
Odpowiedz po polsku dokładnie na pytanie użytkownika.
Nie dodawaj sekcji, których nie wymaga pytanie.

PYTANIE:
{q}

ODPOWIEDŹ:
""".strip()

        answer = gemini_generate(prompt).strip()

        # retry tylko w campaign i tylko gdy wygląda na urwane
        if campaign_mode and looks_truncated(answer, q):
            prompt2 = f"""
Dokończ odpowiedź od miejsca, w którym się urwała. Nie powtarzaj wcześniejszych fragmentów.
Nadal trzymaj się zasady: tylko fakty z KONTEKSTU, a jeśli czegoś brakuje - "brak w notatkach".
Zwracaj odpowiedź w markdown.

KONTEKST:
{context}

DOTYCHCZAS:
{answer}

KONTYNUACJA:
""".strip()
            more = gemini_generate(prompt2).strip()
            if more:
                answer = (answer + "\n" + more).strip()

        sources = (
            [{"doc_type": h["doc_type"], "distance": h["distance"]} for h in hits]
            if req.include_sources
            else []
        )

        return AskResponse(answer=answer, sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_text", response_class=PlainTextResponse)
def ask_text(req: AskRequest):
    resp = ask(req)
    return resp.answer
