from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "004_conversations.sql"


@lru_cache(maxsize=1)
def _schema_statements() -> list[str]:
    return [statement.strip() for statement in SCHEMA_PATH.read_text(encoding="utf-8").split(";") if statement.strip()]


class ConversationRecord(BaseModel):
    conversation_id: str
    campaign_id: str
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str
    last_message_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessageRecord(BaseModel):
    message_id: int
    conversation_id: str
    campaign_id: str
    role: str
    content: str
    kind: Optional[str] = None
    artifact_type: Optional[str] = None
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ConversationStore:
    campaign_id: str
    connection_factory: Callable[[], object]
    _schema_ready: bool = field(default=False, init=False, repr=False)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                for statement in _schema_statements():
                    cur.execute(statement)
            conn.commit()
        self._schema_ready = True

    def create_conversation(
        self,
        *,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationRecord:
        self.ensure_schema()
        conversation_id = uuid4().hex
        payload = metadata or {}

        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into conversations (id, campaign_id, title, metadata)
                    values (%s, %s, %s, %s::jsonb)
                    returning id, campaign_id, title, metadata, created_at, updated_at, last_message_at
                    """,
                    (
                        conversation_id,
                        self.campaign_id,
                        title,
                        json.dumps(payload),
                    ),
                )
                row = cur.fetchone()
            conn.commit()

        return ConversationRecord(
            conversation_id=row[0],
            campaign_id=row[1],
            title=row[2],
            metadata=row[3] or {},
            created_at=row[4].isoformat(),
            updated_at=row[5].isoformat(),
            last_message_at=row[6].isoformat() if row[6] else None,
            message_count=0,
        )

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        self.ensure_schema()
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select
                        c.id,
                        c.campaign_id,
                        c.title,
                        c.metadata,
                        c.created_at,
                        c.updated_at,
                        c.last_message_at,
                        (
                            select count(*)
                            from conversation_messages m
                            where m.conversation_id = c.id
                        ) as message_count
                    from conversations c
                    where c.campaign_id = %s and c.id = %s
                    limit 1
                    """,
                    (self.campaign_id, conversation_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return ConversationRecord(
            conversation_id=row[0],
            campaign_id=row[1],
            title=row[2],
            metadata=row[3] or {},
            created_at=row[4].isoformat(),
            updated_at=row[5].isoformat(),
            last_message_at=row[6].isoformat() if row[6] else None,
            message_count=int(row[7] or 0),
        )

    def list_conversations(self, limit: int = 20) -> List[ConversationRecord]:
        self.ensure_schema()
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select
                        c.id,
                        c.campaign_id,
                        c.title,
                        c.metadata,
                        c.created_at,
                        c.updated_at,
                        c.last_message_at,
                        (
                            select count(*)
                            from conversation_messages m
                            where m.conversation_id = c.id
                        ) as message_count
                    from conversations c
                    where c.campaign_id = %s
                    order by coalesce(c.last_message_at, c.updated_at) desc, c.created_at desc
                    limit %s
                    """,
                    (self.campaign_id, limit),
                )
                rows = cur.fetchall()
        return [
            ConversationRecord(
                conversation_id=row[0],
                campaign_id=row[1],
                title=row[2],
                metadata=row[3] or {},
                created_at=row[4].isoformat(),
                updated_at=row[5].isoformat(),
                last_message_at=row[6].isoformat() if row[6] else None,
                message_count=int(row[7] or 0),
            )
            for row in rows
        ]

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        kind: Optional[str] = None,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessageRecord:
        self.ensure_schema()
        payload = metadata or {}

        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into conversation_messages (
                        conversation_id,
                        campaign_id,
                        role,
                        kind,
                        artifact_type,
                        content,
                        metadata
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    returning id, conversation_id, campaign_id, role, kind, artifact_type, content, metadata, created_at
                    """,
                    (
                        conversation_id,
                        self.campaign_id,
                        role,
                        kind,
                        artifact_type,
                        content,
                        json.dumps(payload),
                    ),
                )
                row = cur.fetchone()
                cur.execute(
                    """
                    update conversations
                    set updated_at = now(),
                        last_message_at = now()
                    where campaign_id = %s and id = %s
                    """,
                    (self.campaign_id, conversation_id),
                )
            conn.commit()

        return ConversationMessageRecord(
            message_id=int(row[0]),
            conversation_id=row[1],
            campaign_id=row[2],
            role=row[3],
            kind=row[4],
            artifact_type=row[5],
            content=row[6],
            metadata=row[7] or {},
            created_at=row[8].isoformat(),
        )

    def list_messages(self, conversation_id: str, limit: int = 100) -> List[ConversationMessageRecord]:
        self.ensure_schema()
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, conversation_id, campaign_id, role, kind, artifact_type, content, metadata, created_at
                    from conversation_messages
                    where campaign_id = %s and conversation_id = %s
                    order by created_at asc, id asc
                    limit %s
                    """,
                    (self.campaign_id, conversation_id, limit),
                )
                rows = cur.fetchall()
        return [
            ConversationMessageRecord(
                message_id=int(row[0]),
                conversation_id=row[1],
                campaign_id=row[2],
                role=row[3],
                kind=row[4],
                artifact_type=row[5],
                content=row[6],
                metadata=row[7] or {},
                created_at=row[8].isoformat(),
            )
            for row in rows
        ]


class NullConversationStore:
    def ensure_schema(self) -> None:
        return None

    def create_conversation(
        self,
        *,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationRecord]:
        return None

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        return None

    def list_conversations(self, limit: int = 20) -> List[ConversationRecord]:
        return []

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        kind: Optional[str] = None,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationMessageRecord]:
        return None

    def list_messages(self, conversation_id: str, limit: int = 100) -> List[ConversationMessageRecord]:
        return []
