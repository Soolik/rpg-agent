from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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
    def __init__(self, app, *, auth: GoogleRequestAuth, public_paths: Sequence[str]):
        super().__init__(app)
        self.auth = auth
        self.public_paths = tuple(public_paths)

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in self.public_paths:
            return await call_next(request)
        try:
            identity = self.auth.verify_bearer(request.headers.get("Authorization"))
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
