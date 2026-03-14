from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .models_v2 import (
    SessionPatchPayload,
    SyncSessionPatchRequest,
    SyncSessionPatchResponse,
    WorldModelCleanupResponse,
    WorldEntityRecord,
    WorldModelStatusResponse,
    WorldSessionRecord,
    WorldThreadRecord,
)


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _best_canonical_thread_match(
    duplicate_row: tuple[int, str, Optional[str], str, Optional[str], str],
    canonical_rows: list[tuple[int, str, Optional[str], str, Optional[str], str]],
) -> Optional[tuple[int, str, Optional[str], str, Optional[str], str]]:
    _, _, _, duplicate_title, duplicate_status, duplicate_change = duplicate_row
    duplicate_title_key = normalize_key(duplicate_title)
    duplicate_haystack = normalize_key(" ".join(filter(None, [duplicate_title, duplicate_status, duplicate_change])))
    best_match = None
    best_score = 0

    for candidate in canonical_rows:
        _, _, candidate_thread_id, candidate_title, _, _ = candidate
        candidate_title_key = normalize_key(candidate_title)
        candidate_thread_id_key = normalize_key(candidate_thread_id or "")
        score = 0

        if duplicate_title_key and duplicate_title_key == candidate_title_key:
            score += 3
        if candidate_title_key and candidate_title_key in duplicate_haystack:
            score += 2
        if candidate_thread_id_key and candidate_thread_id_key in duplicate_haystack:
            score += 2

        if score > best_score:
            best_match = candidate
            best_score = score
        elif score == best_score:
            best_match = None

    if best_score < 2:
        return None
    return best_match


@dataclass
class WorldModelStore:
    campaign_id: str
    connection_factory: Callable[[], object]

    def sync_session_patch(self, request: SyncSessionPatchRequest) -> SyncSessionPatchResponse:
        patch = request.patch
        session_payload = patch.model_dump(mode="json")

        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into world_sessions (
                        campaign_id,
                        session_summary,
                        raw_notes,
                        patch_json,
                        rag_additions,
                        entity_count,
                        thread_count,
                        source_doc_id,
                        source_title
                    )
                    values (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
                    returning id
                    """,
                    (
                        self.campaign_id,
                        patch.session_summary,
                        request.raw_notes,
                        json.dumps(session_payload),
                        json.dumps(patch.rag_additions),
                        len(patch.entities_patch),
                        len(patch.thread_tracker_patch),
                        request.source_doc_id,
                        request.source_title,
                    ),
                )
                row = cur.fetchone()
                session_id = int(row[0])

                for entity in patch.entities_patch:
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
                            self.campaign_id,
                            entity.kind,
                            entity.name,
                            normalize_key(entity.name),
                            entity.description,
                            json.dumps(entity.tags),
                            json.dumps({"source": "sync_session_patch"}),
                            session_id,
                        ),
                    )

                for thread in patch.thread_tracker_patch:
                    thread_key = thread.thread_id.strip() if thread.thread_id else normalize_key(thread.title)
                    cur.execute(
                        """
                        insert into world_threads (
                            campaign_id,
                            thread_key,
                            thread_id,
                            title,
                            normalized_title,
                            status,
                            last_change,
                            metadata,
                            last_session_id
                        )
                        values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        on conflict (campaign_id, thread_key)
                        do update set
                            thread_id = coalesce(excluded.thread_id, world_threads.thread_id),
                            title = excluded.title,
                            normalized_title = excluded.normalized_title,
                            status = coalesce(excluded.status, world_threads.status),
                            last_change = excluded.last_change,
                            metadata = excluded.metadata,
                            last_session_id = excluded.last_session_id,
                            updated_at = now()
                        """,
                        (
                            self.campaign_id,
                            thread_key,
                            thread.thread_id,
                            thread.title,
                            normalize_key(thread.title),
                            thread.status,
                            thread.change,
                            json.dumps({"source": "sync_session_patch"}),
                            session_id,
                        ),
                    )
            conn.commit()

        return SyncSessionPatchResponse(
            session_id=session_id,
            campaign_id=self.campaign_id,
            summary="Session patch synced into world model",
            entity_count=len(patch.entities_patch),
            thread_count=len(patch.thread_tracker_patch),
        )

    def list_entities(self, limit: int = 20, kind: Optional[str] = None) -> list[WorldEntityRecord]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                if kind:
                    cur.execute(
                        """
                        select id, campaign_id, entity_kind, name, description, tags, last_session_id, updated_at
                        from world_entities
                        where campaign_id = %s and entity_kind = %s
                        order by updated_at desc, name asc
                        limit %s
                        """,
                        (self.campaign_id, kind, limit),
                    )
                else:
                    cur.execute(
                        """
                        select id, campaign_id, entity_kind, name, description, tags, last_session_id, updated_at
                        from world_entities
                        where campaign_id = %s
                        order by updated_at desc, name asc
                        limit %s
                        """,
                        (self.campaign_id, limit),
                    )
                rows = cur.fetchall()
        return [
            WorldEntityRecord(
                id=row[0],
                campaign_id=row[1],
                entity_kind=row[2],
                name=row[3],
                description=row[4],
                tags=row[5] or [],
                last_session_id=row[6],
                updated_at=row[7].isoformat(),
            )
            for row in rows
        ]

    def list_threads(self, limit: int = 20, status: Optional[str] = None) -> list[WorldThreadRecord]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                if status:
                    cur.execute(
                        """
                        select id, campaign_id, thread_key, thread_id, title, status, last_change, last_session_id, updated_at
                        from world_threads
                        where campaign_id = %s and status = %s
                        order by updated_at desc, title asc
                        limit %s
                        """,
                        (self.campaign_id, status, limit),
                    )
                else:
                    cur.execute(
                        """
                        select id, campaign_id, thread_key, thread_id, title, status, last_change, last_session_id, updated_at
                        from world_threads
                        where campaign_id = %s
                        order by updated_at desc, title asc
                        limit %s
                        """,
                        (self.campaign_id, limit),
                    )
                rows = cur.fetchall()
        return [
            WorldThreadRecord(
                id=row[0],
                campaign_id=row[1],
                thread_key=row[2],
                thread_id=row[3],
                title=row[4],
                status=row[5],
                last_change=row[6],
                last_session_id=row[7],
                updated_at=row[8].isoformat(),
            )
            for row in rows
        ]

    def list_sessions(self, limit: int = 20) -> list[WorldSessionRecord]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, campaign_id, session_summary, entity_count, thread_count, source_title, created_at
                    from world_sessions
                    where campaign_id = %s
                    order by created_at desc
                    limit %s
                    """,
                    (self.campaign_id, limit),
                )
                rows = cur.fetchall()
        return [
            WorldSessionRecord(
                id=row[0],
                campaign_id=row[1],
                session_summary=row[2],
                entity_count=row[3],
                thread_count=row[4],
                source_title=row[5],
                created_at=row[6].isoformat(),
            )
            for row in rows
        ]

    def cleanup_duplicate_threads(self, dry_run: bool = True) -> WorldModelCleanupResponse:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, thread_key, thread_id, title, status, last_change
                    from world_threads
                    where campaign_id = %s
                    order by updated_at desc, id desc
                    """,
                    (self.campaign_id,),
                )
                rows = cur.fetchall()

                canonical_rows = [row for row in rows if row[2]]
                duplicate_ids: list[int] = []

                for row in rows:
                    if row[2]:
                        continue
                    if _best_canonical_thread_match(row, canonical_rows):
                        duplicate_ids.append(int(row[0]))

                if duplicate_ids and not dry_run:
                    cur.execute(
                        """
                        delete from world_threads
                        where campaign_id = %s and id = any(%s)
                        """,
                        (self.campaign_id, duplicate_ids),
                    )
            if duplicate_ids and not dry_run:
                conn.commit()

        summary = "World model cleanup completed"
        if duplicate_ids:
            summary = "Duplicate world threads identified"
            if not dry_run:
                summary = "Duplicate world threads deleted"

        return WorldModelCleanupResponse(
            campaign_id=self.campaign_id,
            summary=summary,
            dry_run=dry_run,
            duplicate_thread_count=len(duplicate_ids),
            deleted_thread_ids=duplicate_ids,
        )

    def status(self) -> WorldModelStatusResponse:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from world_entities where campaign_id = %s", (self.campaign_id,))
                entity_count = int(cur.fetchone()[0])
                cur.execute("select count(*) from world_threads where campaign_id = %s", (self.campaign_id,))
                thread_count = int(cur.fetchone()[0])
                cur.execute("select count(*) from world_sessions where campaign_id = %s", (self.campaign_id,))
                session_count = int(cur.fetchone()[0])
        return WorldModelStatusResponse(
            campaign_id=self.campaign_id,
            entity_count=entity_count,
            thread_count=thread_count,
            session_count=session_count,
        )


class NullWorldModelStore:
    def sync_session_patch(self, request: SyncSessionPatchRequest) -> Optional[SyncSessionPatchResponse]:
        return None

    def list_entities(self, limit: int = 20, kind: Optional[str] = None) -> list[WorldEntityRecord]:
        return []

    def list_threads(self, limit: int = 20, status: Optional[str] = None) -> list[WorldThreadRecord]:
        return []

    def list_sessions(self, limit: int = 20) -> list[WorldSessionRecord]:
        return []

    def cleanup_duplicate_threads(self, dry_run: bool = True) -> Optional[WorldModelCleanupResponse]:
        return None

    def status(self) -> WorldModelStatusResponse:
        return WorldModelStatusResponse(campaign_id="", entity_count=0, thread_count=0, session_count=0)
