from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence
from urllib.parse import urlencode

import jwt
import requests
from cryptography.fernet import Fernet
from google.oauth2.credentials import Credentials

from .drive_store import DriveStore
from .google_drive_oauth_store import GoogleDriveOAuthConnection, GoogleDriveOAuthStore


AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
DEFAULT_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]


@dataclass(frozen=True)
class GoogleDriveOAuthConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    state_secret: str
    token_encryption_key: str
    scopes: List[str] = field(default_factory=lambda: list(DEFAULT_SCOPES))


@dataclass
class GoogleDriveOAuthStatus:
    configured: bool
    connected: bool
    subject_email: Optional[str]
    scopes: List[str]
    redirect_uri: Optional[str]
    write_mode: str


@dataclass
class GoogleDriveOAuthStart:
    authorization_url: str
    redirect_uri: str
    scopes: List[str]


@dataclass
class GoogleDriveOAuthCallbackResult:
    status: GoogleDriveOAuthStatus
    html_body: str


class GoogleDriveOAuthError(RuntimeError):
    pass


@dataclass
class GoogleDriveOAuthService:
    store: GoogleDriveOAuthStore
    config: Optional[GoogleDriveOAuthConfig] = None
    http_post: Callable[..., requests.Response] = requests.post
    http_get: Callable[..., requests.Response] = requests.get

    def is_configured(self) -> bool:
        return self.config is not None

    def get_status(self) -> GoogleDriveOAuthStatus:
        connection = self.store.get_connection() if self.is_configured() else None
        return GoogleDriveOAuthStatus(
            configured=self.is_configured(),
            connected=connection is not None,
            subject_email=connection.subject_email if connection else None,
            scopes=connection.scopes if connection else (list(self.config.scopes) if self.config else []),
            redirect_uri=self.config.redirect_uri if self.config else None,
            write_mode="user_oauth" if connection else "service_account",
        )

    def start_authorization(self) -> GoogleDriveOAuthStart:
        config = self._require_config()
        state = self._encode_state()
        params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "access_type": "offline",
            "include_granted_scopes": "true",
            "prompt": "consent",
            "state": state,
        }
        return GoogleDriveOAuthStart(
            authorization_url=f"{AUTH_URL}?{urlencode(params)}",
            redirect_uri=config.redirect_uri,
            scopes=list(config.scopes),
        )

    def handle_callback(self, *, code: str, state: str) -> GoogleDriveOAuthCallbackResult:
        config = self._require_config()
        self._decode_state(state)
        token_payload = self._exchange_code_for_tokens(code)
        access_token = token_payload.get("access_token")
        refresh_token = token_payload.get("refresh_token")

        existing = self.store.get_secret_record()
        if not refresh_token and existing:
            refresh_token = self._decrypt_refresh_token(existing.refresh_token_encrypted)
        if not refresh_token:
            raise GoogleDriveOAuthError("Google did not return a refresh token. Reconnect with prompt=consent.")
        if not access_token:
            raise GoogleDriveOAuthError("Google did not return an access token.")

        userinfo = self._fetch_userinfo(access_token)
        subject_email = userinfo.get("email")
        scopes = str(token_payload.get("scope") or " ".join(config.scopes)).split()

        encrypted_refresh = self._encrypt_refresh_token(refresh_token)
        self.store.upsert_connection(
            subject_email=subject_email,
            refresh_token_encrypted=encrypted_refresh,
            scopes=scopes,
            token_uri=TOKEN_URL,
        )
        status = self.get_status()
        subject = html.escape(subject_email or "unknown user")
        body = (
            "<html><body>"
            "<h1>Google Drive connected</h1>"
            f"<p>Writes will now use <strong>{subject}</strong>.</p>"
            "<p>You can close this tab.</p>"
            "</body></html>"
        )
        return GoogleDriveOAuthCallbackResult(status=status, html_body=body)

    def disconnect(self) -> GoogleDriveOAuthStatus:
        if not self.is_configured():
            return self.get_status()
        existing = self.store.get_secret_record()
        if existing:
            refresh_token = self._decrypt_refresh_token(existing.refresh_token_encrypted)
            try:
                self.http_post(REVOKE_URL, data={"token": refresh_token}, timeout=30)
            except Exception:
                pass
            self.store.clear_connection()
        return self.get_status()

    def credentials_provider(self, scopes: Sequence[str]):
        config = self._require_config()
        record = self.store.get_secret_record()
        if not record:
            raise GoogleDriveOAuthError("Google Drive OAuth user credentials are not connected.")
        refresh_token = self._decrypt_refresh_token(record.refresh_token_encrypted)
        effective_scopes = list(record.scopes or config.scopes or scopes)
        return Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=record.token_uri or TOKEN_URL,
            client_id=config.client_id,
            client_secret=config.client_secret,
            scopes=effective_scopes,
        )

    def build_user_drive_store(self, *, folder_map: dict[str, str], core_doc_map: dict[str, str]) -> Optional[DriveStore]:
        if not self.is_configured():
            return None
        if not self.store.get_connection():
            return None
        return DriveStore(
            folder_map=folder_map,
            core_doc_map=core_doc_map,
            credentials_provider=self.credentials_provider,
        )

    def _encode_state(self) -> str:
        config = self._require_config()
        now = int(time.time())
        payload = {"iat": now, "exp": now + 900, "purpose": "google_drive_oauth"}
        return jwt.encode(payload, config.state_secret, algorithm="HS256")

    def _decode_state(self, state: str) -> dict:
        config = self._require_config()
        try:
            payload = jwt.decode(state, config.state_secret, algorithms=["HS256"])
        except Exception as exc:
            raise GoogleDriveOAuthError("Invalid OAuth state.") from exc
        if payload.get("purpose") != "google_drive_oauth":
            raise GoogleDriveOAuthError("Invalid OAuth state purpose.")
        return payload

    def _exchange_code_for_tokens(self, code: str) -> dict:
        config = self._require_config()
        response = self.http_post(
            TOKEN_URL,
            data={
                "code": code,
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "redirect_uri": config.redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise GoogleDriveOAuthError(f"Google token exchange failed: {response.text}")
        return response.json()

    def _fetch_userinfo(self, access_token: str) -> dict:
        response = self.http_get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        if response.status_code >= 400:
            raise GoogleDriveOAuthError(f"Google userinfo request failed: {response.text}")
        return response.json()

    def _encrypt_refresh_token(self, refresh_token: str) -> str:
        return self._fernet().encrypt(refresh_token.encode("utf-8")).decode("utf-8")

    def _decrypt_refresh_token(self, encrypted_token: str) -> str:
        try:
            return self._fernet().decrypt(encrypted_token.encode("utf-8")).decode("utf-8")
        except GoogleDriveOAuthError:
            raise
        except Exception as exc:
            raise GoogleDriveOAuthError("Stored Google Drive OAuth credentials could not be decrypted.") from exc

    def _fernet(self) -> Fernet:
        config = self._require_config()
        try:
            return Fernet(config.token_encryption_key.encode("utf-8"))
        except Exception as exc:
            raise GoogleDriveOAuthError("Google Drive OAuth token encryption key is invalid for this deployment.") from exc

    def _require_config(self) -> GoogleDriveOAuthConfig:
        if self.config is None:
            raise GoogleDriveOAuthError("Google Drive OAuth is not configured for this deployment.")
        return self.config
