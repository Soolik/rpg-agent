import os
import re
import uuid
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import psycopg
from pgvector.psycopg import register_vector

from googleapiclient.discovery import build
import google.auth

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


class AskRequest(BaseModel):
    question: str
    top_k: int = 6


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


def chunk_text(text: str, max_chars: int = 2400, overlap: int = 400) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + max_chars)
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = max(0, end - overlap)
    return chunks


def gemini_embed(texts: List[str]) -> List[List[float]]:
    """
    Embeddings przez endpoint embedContent (pojedyncze) - kompatybilne i stabilne.
    Wymiar embeddingów zależy od modelu, u ciebie to 3072 dla gemini-embedding-001.
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
            "maxOutputTokens": 800,
            "responseMimeType": "text/plain",
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


def db_conn():
    if not DB_URL:
        raise RuntimeError("Brak DB_URL")
    conn = psycopg.connect(DB_URL)
    register_vector(conn)
    return conn


def upsert_chunks(doc_id: str, doc_type: str, chunks: List[str], embeddings: List[List[float]]):
    now = datetime.now(timezone.utc)
    with db_conn() as conn:
        with conn.cursor() as cur:
            for t, emb in zip(chunks, embeddings):
                cid = uuid.uuid4()
                cur.execute(
                    """
                    insert into chunks (id, campaign_id, doc_id, doc_type, chunk_text, embedding, metadata, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(cid),
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
                select doc_id, doc_type, chunk_text,
                       (embedding <-> %s::vector) as distance
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


@app.get("/health")
def health():
    return {"ok": True, "campaign_id": CAMPAIGN_ID, "revision": REVISION}


@app.post("/reindex")
def reindex():
    try:
        for name, val in [
            ("BIBLE_DOC_ID", BIBLE_DOC_ID),
            ("RULES_DOC_ID", RULES_DOC_ID),
            ("GLOSSARY_DOC_ID", GLOSSARY_DOC_ID),
            ("THREADS_DOC_ID", THREADS_DOC_ID),
        ]:
            if not val:
                raise HTTPException(status_code=400, detail=f"Missing env: {name}")

        drive = get_drive_service()

        docs = [
            ("bible", BIBLE_DOC_ID),
            ("rules", RULES_DOC_ID),
            ("glossary", GLOSSARY_DOC_ID),
            ("threads", THREADS_DOC_ID),
        ]

        # MVP: nie kasujemy starych chunków automatycznie, bo może chcesz porównać.
        # Jak chcesz, dodamy w następnym kroku "czysty reindex" (delete by campaign/doc_id).

        total_chunks = 0
        for doc_type, doc_id in docs:
            text = export_google_doc_as_text(drive, doc_id)
            chunks = chunk_text(text)
            if not chunks:
                continue
            embs = gemini_embed(chunks)
            upsert_chunks(doc_id=doc_id, doc_type=doc_type, chunks=chunks, embeddings=embs)
            total_chunks += len(chunks)

        return {"ok": True, "indexed_chunks": total_chunks}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        hits = vector_search(req.question, req.top_k)

        # Ucinamy kontekst, żeby nie dobijać tokenów
        context = "\n\n".join(
            [f"[{i+1}] ({h['doc_type']}) {h['chunk_text'][:1200]}" for i, h in enumerate(hits)]
        )

        prompt = f"""
Jesteś asystentem MG kampanii "Krew Na Gwiazdach".
Używaj tylko faktów z KONTEKSTU. Jeśli brak danych, napisz "brak w notatkach".

Odpowiedz po polsku:
1) Premisa: 6 punktów
2) Wątki: Thread ID | nazwa | stawka
3) Luki: 3 rzeczy do dopisania

KONTEKST:
{context}

PYTANIE:
{req.question}
""".strip()

        answer = gemini_generate(prompt)

        return AskResponse(
            answer=answer,
            sources=[{"doc_type": h["doc_type"], "distance": h["distance"]} for h in hits],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

