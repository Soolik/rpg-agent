from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

from .models_v2 import EntityRelationRecord, WorldFactRecord
from .world_fact_store import (
    NullWorldFactStore,
    ProjectionFact,
    ProjectionRelation,
    WorldFactStore,
    normalize_key,
)
from .world_model_store import NullWorldModelStore, WorldModelStore


def _match_known_names(text: str, names: Iterable[str], *, exclude: str = "") -> list[str]:
    haystack = normalize_key(text)
    exclude_key = normalize_key(exclude)
    matches: list[str] = []
    seen = set()
    for candidate in names:
        key = normalize_key(candidate)
        if not key or key == exclude_key or key in seen:
            continue
        if key in haystack:
            matches.append(candidate)
            seen.add(key)
    return matches


def _supporting_evidence(text: str, match: str) -> str:
    normalized_match = normalize_key(match)
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if normalized_match in normalize_key(line):
            return line[:240]
    compact = " ".join((text or "").split())
    return compact[:240]


@dataclass
class WorldFactService:
    world_model_store: WorldModelStore | NullWorldModelStore
    world_fact_store: WorldFactStore | NullWorldFactStore

    def rebuild_projection(self) -> dict[str, int]:
        entities = self.world_model_store.list_entities(limit=1000)
        threads = self.world_model_store.list_threads(limit=1000)
        sessions = self.world_model_store.list_sessions(limit=300)

        known_names = [entity.name for entity in entities] + [thread.title for thread in threads if thread.title]
        facts: list[ProjectionFact] = []
        relations: list[ProjectionRelation] = []

        for entity in entities:
            if entity.description:
                facts.append(
                    ProjectionFact(
                        subject_type=entity.entity_kind,
                        subject_name=entity.name,
                        predicate="description",
                        object_value=entity.description,
                        source_type="world_model.entity",
                        source_ref=str(entity.id),
                    )
                )
            for tag in entity.tags[:12]:
                facts.append(
                    ProjectionFact(
                        subject_type=entity.entity_kind,
                        subject_name=entity.name,
                        predicate="tag",
                        object_value=tag,
                        source_type="world_model.entity",
                        source_ref=str(entity.id),
                        confidence=0.8,
                    )
                )
            for target_name in _match_known_names(entity.description, known_names, exclude=entity.name):
                relations.append(
                    ProjectionRelation(
                        source_name=entity.name,
                        relation_type="mentions",
                        target_name=target_name,
                        evidence=_supporting_evidence(entity.description, target_name),
                        source_type="world_model.entity",
                        source_ref=str(entity.id),
                        confidence=0.75,
                    )
                )

        for thread in threads:
            if thread.thread_id:
                facts.append(
                    ProjectionFact(
                        subject_type="thread",
                        subject_name=thread.title,
                        predicate="thread_id",
                        object_value=thread.thread_id,
                        source_type="world_model.thread",
                        source_ref=str(thread.id),
                    )
                )
            if thread.status:
                facts.append(
                    ProjectionFact(
                        subject_type="thread",
                        subject_name=thread.title,
                        predicate="status",
                        object_value=thread.status,
                        source_type="world_model.thread",
                        source_ref=str(thread.id),
                    )
                )
            if thread.last_change:
                facts.append(
                    ProjectionFact(
                        subject_type="thread",
                        subject_name=thread.title,
                        predicate="last_change",
                        object_value=thread.last_change,
                        source_type="world_model.thread",
                        source_ref=str(thread.id),
                        confidence=0.8,
                    )
                )
                for target_name in _match_known_names(thread.last_change, known_names, exclude=thread.title):
                    relations.append(
                        ProjectionRelation(
                            source_name=thread.title,
                            relation_type="mentions",
                            target_name=target_name,
                            evidence=_supporting_evidence(thread.last_change, target_name),
                            source_type="world_model.thread",
                            source_ref=str(thread.id),
                            confidence=0.7,
                        )
                    )

        for session in sessions:
            subject_name = session.source_title or f"Session {session.id}"
            facts.append(
                ProjectionFact(
                    subject_type="session",
                    subject_name=subject_name,
                    predicate="summary",
                    object_value=session.session_summary,
                    source_type="world_model.session",
                    source_ref=str(session.id),
                    confidence=0.7,
                )
            )
            for target_name in _match_known_names(session.session_summary, known_names):
                relations.append(
                    ProjectionRelation(
                        source_name=subject_name,
                        relation_type="mentions",
                        target_name=target_name,
                        evidence=_supporting_evidence(session.session_summary, target_name),
                        source_type="world_model.session",
                        source_ref=str(session.id),
                        confidence=0.65,
                    )
                )

        self.world_fact_store.replace_projection(facts=facts, relations=relations)
        return {"facts": len(facts), "relations": len(relations)}

    def list_facts(self, *, limit: int = 50, subject_name: str | None = None, predicate: str | None = None) -> list[WorldFactRecord]:
        return self.world_fact_store.list_facts(limit=limit, subject_name=subject_name, predicate=predicate)

    def list_relations(
        self,
        *,
        limit: int = 50,
        source_name: str | None = None,
        target_name: str | None = None,
        relation_type: str | None = None,
    ) -> list[EntityRelationRecord]:
        return self.world_fact_store.list_relations(
            limit=limit,
            source_name=source_name,
            target_name=target_name,
            relation_type=relation_type,
        )

    def search_facts(self, query: str, *, limit: int = 20) -> list[WorldFactRecord]:
        qkey = normalize_key(query)
        if not qkey:
            return []
        matches: list[tuple[int, WorldFactRecord]] = []
        for fact in self.world_fact_store.list_facts(limit=max(limit * 20, 500)):
            haystack = normalize_key(" ".join([fact.subject_name, fact.predicate, fact.object_value]))
            if qkey not in haystack:
                continue
            score = 120 if qkey == normalize_key(fact.subject_name) else 80
            if qkey in normalize_key(fact.object_value):
                score += 20
            matches.append((score, fact))
        matches.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [fact for _, fact in matches[:limit]]

    def related_subject_names(self, text: str, *, limit: int = 12) -> list[str]:
        names = []
        seen = set()
        for fact in self.search_facts(text, limit=limit):
            key = normalize_key(fact.subject_name)
            if key in seen:
                continue
            seen.add(key)
            names.append(fact.subject_name)
        return names[:limit]
