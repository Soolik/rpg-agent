from __future__ import annotations

from dataclasses import dataclass

from .api_models import WorldModelSearchItem
from .canon_guard import normalize_key
from .world_model_store import NullWorldModelStore, WorldModelStore


@dataclass
class WorldModelService:
    world_model_store: WorldModelStore | NullWorldModelStore

    def list_entities(self, *, limit: int = 20, kind: str | None = None):
        return self.world_model_store.list_entities(limit=limit, kind=kind)

    def list_threads(self, *, limit: int = 20, status: str | None = None):
        return self.world_model_store.list_threads(limit=limit, status=status)

    def list_sessions(self, *, limit: int = 20):
        return self.world_model_store.list_sessions(limit=limit)

    def search(self, query: str, *, limit: int = 20) -> list[WorldModelSearchItem]:
        qkey = normalize_key(query)
        if not qkey:
            return []

        items: list[WorldModelSearchItem] = []
        scan_limit = max(limit * 10, 1000)

        for entity in self.world_model_store.list_entities(limit=scan_limit):
            haystack = normalize_key(" ".join(filter(None, [entity.name, entity.description, " ".join(entity.tags)])))
            if qkey not in haystack:
                continue
            score = 120 if qkey == normalize_key(entity.name) else 90 if qkey in normalize_key(entity.name) else 60
            items.append(
                WorldModelSearchItem(
                    record_type="entity",
                    record_id=entity.id,
                    title=entity.name,
                    snippet=entity.description[:220],
                    entity_kind=entity.entity_kind,
                    score=score,
                )
            )

        for thread in self.world_model_store.list_threads(limit=scan_limit):
            haystack = normalize_key(" ".join(filter(None, [thread.thread_id or "", thread.title, thread.last_change, thread.status or ""])))
            if qkey not in haystack:
                continue
            score = 115 if qkey == normalize_key(thread.title) else 85 if qkey in normalize_key(thread.title) else 55
            items.append(
                WorldModelSearchItem(
                    record_type="thread",
                    record_id=thread.id,
                    title=thread.title,
                    snippet=thread.last_change[:220],
                    status=thread.status,
                    score=score,
                )
            )

        for session in self.world_model_store.list_sessions(limit=scan_limit):
            haystack = normalize_key(" ".join(filter(None, [session.session_summary, session.source_title or ""])))
            if qkey not in haystack:
                continue
            score = 70 if qkey in normalize_key(session.source_title or "") else 50
            items.append(
                WorldModelSearchItem(
                    record_type="session",
                    record_id=session.id,
                    title=session.source_title or f"Session {session.id}",
                    snippet=session.session_summary[:220],
                    source_title=session.source_title,
                    score=score,
                )
            )

        items.sort(key=lambda item: (item.score, item.record_id), reverse=True)
        return items[:limit]
