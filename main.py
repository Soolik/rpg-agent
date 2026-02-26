import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build

import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

APP_NAME = "rpg-agent"
app = FastAPI(title=APP_NAME)

# ---------- Config ----------
CAMPAIGN_ID = os.getenv("CAMPAIGN_ID", "kng")

BIBLE_DOC_ID = os.getenv("BIBLE_DOC_ID")
RULES_DOC_ID = os.getenv("RULES_DOC_ID")
GLOSSARY_DOC_ID = os.getenv("GLOSSARY_DOC_ID")
THREADS_DOC_ID = os.getenv("THREADS_DOC_ID")

DB_URL = os.getenv("DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# For local dev only: path to a service account json key file.
# On Cloud Run, we'll use the runtime service account instead.
GOOGLE_SA_JSON = os.getenv("GOOGLE_SA_JSON")  # optional


# ---------- Models ----------
class AskRequest(BaseModel):
    question: str
    top_k: int = 12


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


# ---------- Helpers ----------
def chunk_text(text: str, max_chars: int = 2400, overlap: int = 400) -> List[str]:
    """
    Cheap chunker by characters. Good enough for MVP.
    """
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
    Calls Gemini embeddings endpoint. Uses REST to avoid extra SDK.
    Note: model name can change; we'll tune later.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY")

    url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents"
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": t}]}} for t in texts]
    }
    r = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini embeddings error: {r.status_code} {r.text}")
    data = r.json()
    return [item["embedding"]["values"] for item in data["embeddings"]]


def gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 800},
    }
    r = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini generate error: {r.status_code} {r.text}")
    data = r.json()
    # defensive parse
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return str(data)


def get_drive_service():
    if GOOGLE_SA_JSON:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SA_JSON,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
    else:
        # On Cloud Run, ADC will pick runtime service account
        import google.auth
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def export_google_doc_as_text(drive, doc_id: str) -> str:
    # export Google Doc as plain text
    data = drive.files().export(fileId=doc_id, mimeType="text/plain").execute()
    return data.decode("utf-8", errors="ignore")


def db_conn():
    if not DB_URL:
        raise RuntimeError("Missing DB_URL")
    conn = psycopg.connect(DB_URL)
    register_vector(conn)
    return conn


def upsert_chunks(doc_id: str, doc_type: str, chunks: List[str], embeddings: List[List[float]]):
    now = datetime.now(timezone.utc)
    with db_conn() as conn:
        with conn.cursor() as cur:
            for chunk_text_val, emb in zip(chunks, embeddings):
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
                        chunk_text_val,
                        emb,
                        {"source": doc_type},
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
    results = []
    for doc_id, doc_type, chunk_text_val, dist in rows:
        results.append(
            {"doc_id": doc_id, "doc_type": doc_type, "chunk_text": chunk_text_val, "distance": float(dist)}
        )
    return results


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "campaign_id": CAMPAIGN_ID}


@app.post("/reindex")
def reindex():
    # Basic validation
    for name, val in [("BIBLE_DOC_ID", BIBLE_DOC_ID), ("RULES_DOC_ID", RULES_DOC_ID),
                      ("GLOSSARY_DOC_ID", GLOSSARY_DOC_ID), ("THREADS_DOC_ID", THREADS_DOC_ID)]:
        if not val:
            raise HTTPException(status_code=400, detail=f"Missing env: {name}")

    drive = get_drive_service()

    docs = [
        ("bible", BIBLE_DOC_ID),
        ("rules", RULES_DOC_ID),
        ("glossary", GLOSSARY_DOC_ID),
        ("threads", THREADS_DOC_ID),
    ]

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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = vector_search(req.question, req.top_k)
    context = "\n\n".join(
        [f"[{i+1}] ({h['doc_type']}) {h['chunk_text']}" for i, h in enumerate(hits)]
    )

    prompt = f"""
Jesteś asystentem MG do kampanii "{CAMPAIGN_ID}".
ZASADA: Fakty bierz wyłącznie z KONTEKSTU. Jeśli czegoś nie ma, powiedz 'brak w notatkach' i zaproponuj pytania.
Zwróć odpowiedź po polsku w 4 sekcjach:
1) Fakty z notatek
2) Luki (czego brakuje)
3) Sugestie (pomysły zgodne z faktami)
4) Następne kroki (co dopisać / co przygotować)

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