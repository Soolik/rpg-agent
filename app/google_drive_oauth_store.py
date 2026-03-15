from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, Field


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "005_google_drive_oauth.sql"
PROVIDER_NAME = "google_drive"


@lru_cache(maxsize=1)
def _schema_statements() -> list[str]:
    return [statement.strip() for statement in SCHEMA_PATH.read_text(encoding="utf-8").split(";") if statement.strip()]


class GoogleDriveOAuthConnection(BaseModel):
    campaign_id: str
    provider: str = PROVIDER_NAME
    subject_email: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    token_uri: str = "https://oauth2.googleapis.com/token"
    created_at: str
    updated_at: str


@dataclass
class GoogleDriveOAuthSecretRecord:
    campaign_id: str
    provider: str
    subject_email: Optional[str]
    refresh_token_encrypted: str
    scopes: List[str]
    token_uri: str
    created_at: str
    updated_at: str


@dataclass
class GoogleDriveOAuthStore:
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

    def get_connection(self) -> Optional[GoogleDriveOAuthConnection]:
        secret = self.get_secret_record()
        if not secret:
            return None
        return GoogleDriveOAuthConnection(
            campaign_id=secret.campaign_id,
            provider=secret.provider,
            subject_email=secret.subject_email,
            scopes=secret.scopes,
            token_uri=secret.token_uri,
            created_at=secret.created_at,
            updated_at=secret.updated_at,
        )

    def get_secret_record(self) -> Optional[GoogleDriveOAuthSecretRecord]:
        self.ensure_schema()
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select
                        campaign_id,
                        provider,
                        subject_email,
                        refresh_token_encrypted,
                        scopes,
                        token_uri,
                        created_at,
                        updated_at
                    from google_drive_oauth_connections
                    where campaign_id = %s and provider = %s
                    limit 1
                    """,
                    (self.campaign_id, PROVIDER_NAME),
                )
                row = cur.fetchone()
        if not row:
            return None
        return GoogleDriveOAuthSecretRecord(
            campaign_id=row[0],
            provider=row[1],
            subject_email=row[2],
            refresh_token_encrypted=row[3],
            scopes=row[4] or [],
            token_uri=row[5],
            created_at=row[6].isoformat(),
            updated_at=row[7].isoformat(),
        )

    def upsert_connection(
        self,
        *,
        subject_email: Optional[str],
        refresh_token_encrypted: str,
        scopes: List[str],
        token_uri: str = "https://oauth2.googleapis.com/token",
    ) -> GoogleDriveOAuthConnection:
        self.ensure_schema()
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into google_drive_oauth_connections (
                        campaign_id,
                        provider,
                        subject_email,
                        refresh_token_encrypted,
                        scopes,
                        token_uri
                    )
                    values (%s, %s, %s, %s, %s::jsonb, %s)
                    on conflict (campaign_id, provider)
                    do update set
                        subject_email = excluded.subject_email,
                        refresh_token_encrypted = excluded.refresh_token_encrypted,
                        scopes = excluded.scopes,
                        token_uri = excluded.token_uri,
                        updated_at = now()
                    returning campaign_id, provider, subject_email, scopes, token_uri, created_at, updated_at
                    """,
                    (
                        self.campaign_id,
                        PROVIDER_NAME,
                        subject_email,
                        refresh_token_encrypted,
                        json.dumps(scopes),
                        token_uri,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return GoogleDriveOAuthConnection(
            campaign_id=row[0],
            provider=row[1],
            subject_email=row[2],
            scopes=row[3] or [],
            token_uri=row[4],
            created_at=row[5].isoformat(),
            updated_at=row[6].isoformat(),
        )

    def clear_connection(self) -> bool:
        self.ensure_schema()
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    delete from google_drive_oauth_connections
                    where campaign_id = %s and provider = %s
                    """,
                    (self.campaign_id, PROVIDER_NAME),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
