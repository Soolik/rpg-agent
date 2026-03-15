import unittest
from types import SimpleNamespace

from starlette.responses import JSONResponse

from app.request_auth import GoogleRequestAuth, RequestAuthError, RequestAuthMiddleware, SignedSessionAuth


class GoogleRequestAuthTest(unittest.TestCase):
    def test_verify_bearer_accepts_allowed_google_identity(self):
        auth = GoogleRequestAuth(
            allowed_emails=["soolik1990@gmail.com"],
            allowed_audiences=["32555940559.apps.googleusercontent.com"],
            verify_token_fn=lambda token, request, audience=None: {
                "iss": "https://accounts.google.com",
                "aud": "32555940559.apps.googleusercontent.com",
                "email": "soolik1990@gmail.com",
                "email_verified": True,
                "sub": "user-123",
            },
            request_factory=lambda: object(),
        )

        identity = auth.verify_bearer("Bearer token-value")

        self.assertEqual(identity.email, "soolik1990@gmail.com")
        self.assertEqual(identity.subject, "user-123")

    def test_verify_bearer_rejects_wrong_audience(self):
        auth = GoogleRequestAuth(
            allowed_emails=["soolik1990@gmail.com"],
            allowed_audiences=["32555940559.apps.googleusercontent.com"],
            verify_token_fn=lambda token, request, audience=None: {
                "iss": "https://accounts.google.com",
                "aud": "unexpected-client-id",
                "email": "soolik1990@gmail.com",
                "email_verified": True,
                "sub": "user-123",
            },
            request_factory=lambda: object(),
        )

        with self.assertRaises(RequestAuthError):
            auth.verify_bearer("Bearer token-value")


class RequestAuthMiddlewareTest(unittest.TestCase):
    def build_middleware(self):
        return RequestAuthMiddleware(
            app=lambda scope, receive, send: None,
            auth=GoogleRequestAuth(
                allowed_emails=["soolik1990@gmail.com"],
                allowed_audiences=["32555940559.apps.googleusercontent.com"],
                verify_token_fn=lambda token, request, audience=None: {
                    "iss": "https://accounts.google.com",
                    "aud": "32555940559.apps.googleusercontent.com",
                    "email": "soolik1990@gmail.com",
                    "email_verified": True,
                    "sub": "user-123",
                },
                request_factory=lambda: object(),
            ),
            session_auth=SignedSessionAuth(secret="session-secret-that-is-definitely-long-enough"),
            public_paths=("/health", "/v1/health", "/v1/auth/google-drive/callback"),
        )

    async def call_dispatch(self, middleware, request, response=None):
        return await middleware.dispatch(
            request,
            lambda incoming_request: self._return_response(incoming_request, response=response),
        )

    async def _return_response(self, request, *, response=None):
        if response is not None:
            return response
        return JSONResponse(
            {
                "email": getattr(request.state, "identity_email", None),
                "subject": getattr(request.state, "identity_subject", None),
            }
        )

    def build_request(self, path, authorization=None, cookies=None):
        headers = {}
        if authorization:
            headers["Authorization"] = authorization
        return SimpleNamespace(
            url=SimpleNamespace(path=path),
            headers=headers,
            cookies=cookies or {},
            state=SimpleNamespace(),
        )

    def test_public_callback_path_bypasses_auth(self):
        middleware = self.build_middleware()
        request = self.build_request("/v1/auth/google-drive/callback")
        response = self._run(self.call_dispatch(middleware, request, response=JSONResponse({"ok": True})))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b'{"ok":true}')

    def test_protected_path_requires_bearer_token(self):
        middleware = self.build_middleware()
        request = self.build_request("/protected")
        response = self._run(self.call_dispatch(middleware, request))

        self.assertEqual(response.status_code, 401)
        self.assertIn(b'"code":"unauthorized"', response.body)

    def test_protected_path_accepts_valid_google_token(self):
        middleware = self.build_middleware()
        request = self.build_request("/protected", authorization="Bearer token-value")
        response = self._run(self.call_dispatch(middleware, request))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b'{"email":"soolik1990@gmail.com","subject":"user-123"}')

    def test_protected_path_accepts_valid_session_cookie(self):
        middleware = self.build_middleware()
        session_auth = middleware.session_auth
        cookie_value = session_auth.issue(email="soolik1990@gmail.com", subject="user-123")
        request = self.build_request("/protected", cookies={session_auth.cookie_name: cookie_value})
        response = self._run(self.call_dispatch(middleware, request))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b'{"email":"soolik1990@gmail.com","subject":"user-123"}')

    def _run(self, coroutine):
        import asyncio

        return asyncio.run(coroutine)


if __name__ == "__main__":
    unittest.main()
