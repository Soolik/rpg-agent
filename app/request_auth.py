from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jwt
from fastapi.responses import JSONResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token
from starlette.middleware.base import BaseHTTPMiddleware


GCLOUD_CLIENT_ID = "32555940559.apps.googleusercontent.com"


@dataclass
class VerifiedRequestIdentity:
    email: Optional[str]
    subject: Optional[str]
    audience: Optional[str]
    issuer: Optional[str]


class RequestAuthError(RuntimeError):
    pass


@dataclass
class SignedSessionAuth:
    secret: str
    cookie_name: str = "gm_session"
    ttl_seconds: int = 60 * 60 * 24 * 14

    def issue(self, *, email: str, subject: Optional[str] = None) -> str:
        now = int(time.time())
        payload = {
            "iat": now,
            "exp": now + self.ttl_seconds,
            "purpose": "web_session",
            "email": email,
            "sub": subject,
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def verify_cookie(self, cookie_value: Optional[str]) -> VerifiedRequestIdentity:
        if not cookie_value:
            raise RequestAuthError("Missing web session.")
        try:
            payload = jwt.decode(cookie_value, self.secret, algorithms=["HS256"])
        except Exception as exc:
            raise RequestAuthError("Web session is invalid or expired.") from exc
        if payload.get("purpose") != "web_session":
            raise RequestAuthError("Unexpected web session purpose.")
        email = payload.get("email")
        if not email:
            raise RequestAuthError("Web session is missing email.")
        return VerifiedRequestIdentity(
            email=email,
            subject=payload.get("sub"),
            audience="web_session",
            issuer="local_session",
        )


@dataclass
class GoogleRequestAuth:
    allowed_emails: Sequence[str]
    allowed_audiences: Sequence[str]
    verify_token_fn: Callable = id_token.verify_oauth2_token
    request_factory: Callable[[], object] = GoogleAuthRequest

    def verify_bearer(self, authorization_header: Optional[str]) -> VerifiedRequestIdentity:
        if not authorization_header or not authorization_header.lower().startswith("bearer "):
            raise RequestAuthError("Missing bearer token.")
        token = authorization_header.split(" ", 1)[1].strip()
        if not token:
            raise RequestAuthError("Missing bearer token.")
        try:
            payload = self.verify_token_fn(token, self.request_factory(), audience=None)
        except Exception as exc:
            raise RequestAuthError(f"Token verification failed: {exc}") from exc

        issuer = payload.get("iss")
        audience = payload.get("aud")
        email = payload.get("email")
        email_verified = bool(payload.get("email_verified"))

        if issuer not in {"https://accounts.google.com", "accounts.google.com"}:
            raise RequestAuthError("Unexpected token issuer.")
        if audience not in set(self.allowed_audiences):
            raise RequestAuthError("Unexpected token audience.")
        if not email or not email_verified:
            raise RequestAuthError("Verified email is required.")
        if self.allowed_emails and email.lower() not in {item.lower() for item in self.allowed_emails}:
            raise RequestAuthError("Email is not allowed for this API.")

        return VerifiedRequestIdentity(
            email=email,
            subject=payload.get("sub"),
            audience=audience,
            issuer=issuer,
        )


class RequestAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, auth: GoogleRequestAuth, public_paths: Sequence[str], session_auth: Optional[SignedSessionAuth] = None):
        super().__init__(app)
        self.auth = auth
        self.public_paths = tuple(public_paths)
        self.session_auth = session_auth

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in self.public_paths:
            return await call_next(request)
        try:
            identity = None
            authorization_header = request.headers.get("Authorization")
            if authorization_header:
                identity = self.auth.verify_bearer(authorization_header)
            elif self.session_auth:
                identity = self.session_auth.verify_cookie(request.cookies.get(self.session_auth.cookie_name))
            else:
                raise RequestAuthError("Missing bearer token.")
        except RequestAuthError as exc:
            return JSONResponse(
                status_code=401,
                content={
                    "code": "unauthorized",
                    "message": str(exc),
                },
            )
        request.state.identity_email = identity.email
        request.state.identity_subject = identity.subject
        request.state.identity_audience = identity.audience
        return await call_next(request)
