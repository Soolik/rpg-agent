from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .canon_guard import (
    IGNORED_SINGLE_WORDS,
    PROPER_NOUN_IGNORE_KEYS,
    extract_titlecase_phrases,
    looks_like_proper_noun_label,
    normalize_key,
)
from .models_v2 import WorldDocInfo, WorldEntityType
from .text_normalization import normalize_text_artifacts

DOC_CANON_IGNORE_KEYS = PROPER_NOUN_IGNORE_KEYS | {
    "black",
    "campaign",
    "cienie",
    "dossier",
    "docs ids",
    "events",
    "factions",
    "glossary",
    "index",
    "index - docs ids",
    "npcs",
    "outputs",
    "places",
    "port",
    "przewodnik",
    "rozdzial",
    "sessions",
    "secrets",
    "shackles canonic",
    "sprawa",
    "thread tracker",
    "world",
    "zrodla",
}

DOC_CANON_EDGE_IGNORE_KEYS = {
    "co",
    "czy",
    "dla",
    "do",
    "gdzie",
    "jak",
    "kiedy",
    "kto",
    "na",
    "o",
    "od",
    "operacyjnie",
    "oraz",
    "po",
    "politycznie",
    "przed",
    "przez",
    "to",
    "w",
    "za",
    "ze",
}


@dataclass(frozen=True)
class DocBackedEntity:
    doc_id: str
    entity_kind: str
    name: str
    description: str
    tags: list[str]
    metadata: dict[str, str]


def _world_entity_kind(doc: WorldDocInfo) -> str:
    entity_type = getattr(doc.entity_type, "value", str(doc.entity_type or WorldEntityType.other))
    mapping = {
        WorldEntityType.npc.value: "npc",
        WorldEntityType.location.value: "location",
        WorldEntityType.faction.value: "faction",
        WorldEntityType.secret.value: "secret",
        WorldEntityType.glossary.value: "glossary",
        WorldEntityType.bible.value: "bible",
        WorldEntityType.rules.value: "rules",
    }
    return mapping.get(entity_type, "canon")


def _clean_candidate(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_text_artifacts(value or "").strip()).strip(" -*:;,.!?()[]{}\"'")


def _looks_like_doc_canon_name(value: str) -> bool:
    cleaned = _clean_candidate(value)
    if not cleaned:
        return False
    key = normalize_key(cleaned)
    if not key or key in DOC_CANON_IGNORE_KEYS:
        return False
    if " " not in cleaned:
        return len(cleaned) >= 5 and key not in IGNORED_SINGLE_WORDS and cleaned[:1].isupper()
    return True


def _looks_like_meaningful_multiword_doc_name(value: str) -> bool:
    cleaned = _clean_candidate(value)
    if not cleaned or " " not in cleaned:
        return False
    words = [part for part in cleaned.split() if part]
    if len(words) < 2:
        return False
    first_key = normalize_key(words[0])
    last_key = normalize_key(words[-1])
    if first_key in DOC_CANON_EDGE_IGNORE_KEYS or last_key in DOC_CANON_EDGE_IGNORE_KEYS:
        return False
    return all(word[:1].isupper() for word in words)


def _extract_structured_name_candidates(text: str) -> list[str]:
    normalized = normalize_text_artifacts(text or "")
    candidates: list[str] = []
    seen = set()

    def add_candidate(value: str) -> None:
        cleaned = _clean_candidate(value)
        key = normalize_key(cleaned)
        if not cleaned or not key or key in seen:
            return
        if not looks_like_proper_noun_label(cleaned):
            return
        seen.add(key)
        candidates.append(cleaned)

    for match in re.findall(r"(?m)^\s*#{1,6}\s+([^\n]{2,80})$", normalized):
        add_candidate(match)
    for match in re.findall(r"\*\*([^*\n]{2,80})\*\*", normalized):
        add_candidate(match)
    for match in re.findall(
        r"(?m)^\s*[\*\-]?\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9'_\- ]{1,80})\s*:",
        normalized,
    ):
        add_candidate(match)
    return candidates


def _supporting_line(text: str, name: str, doc: WorldDocInfo) -> str:
    normalized_name = normalize_key(name)
    for raw_line in normalize_text_artifacts(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if normalized_name in normalize_key(line):
            return line[:240]
    return f"Zrodlo: {doc.folder} / {doc.title}"


def extract_doc_backed_entities(doc: WorldDocInfo, text: str, *, limit: int = 80) -> list[DocBackedEntity]:
    if not doc.doc_id or not text.strip():
        return []

    candidates: list[str] = []
    seen = set()

    def add_candidate(value: str) -> None:
        cleaned = _clean_candidate(value)
        key = normalize_key(cleaned)
        if not cleaned or not key or key in seen or not _looks_like_doc_canon_name(cleaned):
            return
        seen.add(key)
        candidates.append(cleaned)

    for phrase in extract_titlecase_phrases(doc.title):
        add_candidate(phrase)

    for phrase in _extract_structured_name_candidates(text):
        add_candidate(phrase)

    for phrase in extract_titlecase_phrases(text):
        if not _looks_like_meaningful_multiword_doc_name(phrase):
            continue
        add_candidate(phrase)

    entity_kind = _world_entity_kind(doc)
    entities: list[DocBackedEntity] = []
    for name in candidates[:limit]:
        entities.append(
            DocBackedEntity(
                doc_id=doc.doc_id,
                entity_kind=entity_kind,
                name=name,
                description=_supporting_line(text, name, doc),
                tags=[entity_kind, doc.folder, doc.title],
                metadata={
                    "source": "doc_index",
                    "doc_id": doc.doc_id,
                    "doc_title": doc.title,
                    "folder": doc.folder,
                },
            )
        )
    return entities


def sync_doc_backed_entities(
    *,
    campaign_id: str,
    connection_factory: Callable[[], object],
    docs_with_content: Sequence[tuple[WorldDocInfo, str]],
) -> int:
    doc_ids = [doc.doc_id for doc, _ in docs_with_content if doc.doc_id]
    if not doc_ids:
        return 0

    entities: list[DocBackedEntity] = []
    for doc, text in docs_with_content:
        entities.extend(extract_doc_backed_entities(doc, text))

    with connection_factory() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                delete from world_entities
                where campaign_id = %s
                  and metadata->>'source' = 'doc_index'
                  and metadata->>'doc_id' = any(%s)
                """,
                (campaign_id, doc_ids),
            )
            for entity in entities:
                cur.execute(
                    """
                    insert into world_entities (
                        campaign_id,
                        entity_kind,
                        name,
                        normalized_name,
                        description,
                        tags,
                        metadata,
                        last_session_id
                    )
                    values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                    on conflict (campaign_id, entity_kind, normalized_name)
                    do update set
                        name = excluded.name,
                        description = excluded.description,
                        tags = excluded.tags,
                        metadata = excluded.metadata,
                        last_session_id = excluded.last_session_id,
                        updated_at = now()
                    """,
                    (
                        campaign_id,
                        entity.entity_kind,
                        entity.name,
                        normalize_key(entity.name),
                        entity.description,
                        json.dumps(entity.tags),
                        json.dumps(entity.metadata),
                        None,
                    ),
                )
        conn.commit()

    return len(entities)


def strip_reference_block(text: str) -> str:
    lines = normalize_text_artifacts(text or "").splitlines()
    if not lines:
        return ""

    cut_index = len(lines)
    for idx, raw_line in enumerate(lines):
        key = normalize_key(raw_line)
        if key in {"zrodla", "zrodla:", "sources", "sources:"}:
            cut_index = idx
            break

    return "\n".join(lines[:cut_index]).strip()
