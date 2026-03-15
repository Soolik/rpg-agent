import unittest

from app.google_drive_oauth_service import (
    GoogleDriveOAuthConfig,
    GoogleDriveOAuthService,
)


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeStore:
    def __init__(self):
        self.connection = None
        self.secret = None
        self.cleared = False

    def get_connection(self):
        return self.connection

    def get_secret_record(self):
        return self.secret

    def upsert_connection(self, *, subject_email, refresh_token_encrypted, scopes, token_uri):
        from types import SimpleNamespace

        self.secret = SimpleNamespace(
            campaign_id="kng",
            provider="google_drive",
            subject_email=subject_email,
            refresh_token_encrypted=refresh_token_encrypted,
            scopes=scopes,
            token_uri=token_uri,
            created_at="2026-03-15T00:00:00+00:00",
            updated_at="2026-03-15T00:00:00+00:00",
        )
        self.connection = SimpleNamespace(
            campaign_id="kng",
            provider="google_drive",
            subject_email=subject_email,
            scopes=scopes,
            token_uri=token_uri,
            created_at="2026-03-15T00:00:00+00:00",
            updated_at="2026-03-15T00:00:00+00:00",
        )
        return self.connection

    def clear_connection(self):
        self.connection = None
        self.secret = None
        self.cleared = True
        return True


class GoogleDriveOAuthServiceTest(unittest.TestCase):
    def build_service(self, store=None, http_post=None, http_get=None):
        return GoogleDriveOAuthService(
            store=store or FakeStore(),
            config=GoogleDriveOAuthConfig(
                client_id="client-id",
                client_secret="client-secret",
                redirect_uri="https://example.com/v1/auth/google-drive/callback",
                state_secret="state-secret-that-is-definitely-long-enough",
                token_encryption_key="Zzp_b4E_OuSTWb9MUZ4b8vg4LZ8YHUxe6GJrKUoUbeU=",
            ),
            http_post=http_post or (lambda *args, **kwargs: FakeResponse(payload={})),
            http_get=http_get or (lambda *args, **kwargs: FakeResponse(payload={})),
            allowed_emails=["soolik1990@gmail.com"],
        )

    def test_start_authorization_returns_google_url(self):
        service = self.build_service()

        started = service.start_authorization()

        self.assertIn("accounts.google.com", started.authorization_url)
        self.assertIn("client_id=client-id", started.authorization_url)
        self.assertEqual(started.redirect_uri, "https://example.com/v1/auth/google-drive/callback")

    def test_handle_callback_stores_refresh_token_and_email(self):
        store = FakeStore()
        token_calls = []
        user_calls = []

        def fake_post(url, data=None, timeout=None):
            token_calls.append((url, data, timeout))
            return FakeResponse(
                payload={
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "scope": "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents",
                }
            )

        def fake_get(url, headers=None, timeout=None):
            user_calls.append((url, headers, timeout))
            return FakeResponse(payload={"email": "soolik1990@gmail.com"})

        service = self.build_service(store=store, http_post=fake_post, http_get=fake_get)
        state = service.start_authorization().authorization_url.split("state=", 1)[1].split("&", 1)[0]

        result = service.handle_callback(code="auth-code", state=state)

        self.assertTrue(result.status.connected)
        self.assertEqual(result.status.subject_email, "soolik1990@gmail.com")
        self.assertEqual(result.subject_email, "soolik1990@gmail.com")
        self.assertIn("window.location.replace('/gm');", result.html_body)
        self.assertIn('content="0; url=/gm"', result.html_body)
        self.assertIn('href="/gm"', result.html_body)
        self.assertIsNotNone(store.secret)
        self.assertEqual(token_calls[0][1]["code"], "auth-code")
        self.assertEqual(user_calls[0][1]["Authorization"], "Bearer access-token")

    def test_handle_callback_rejects_disallowed_email(self):
        def fake_post(url, data=None, timeout=None):
            return FakeResponse(
                payload={
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "scope": "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents",
                }
            )

        def fake_get(url, headers=None, timeout=None):
            return FakeResponse(payload={"email": "other@example.com"})

        service = self.build_service(store=FakeStore(), http_post=fake_post, http_get=fake_get)
        state = service.start_authorization().authorization_url.split("state=", 1)[1].split("&", 1)[0]

        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            service.handle_callback(code="auth-code", state=state)

    def test_disconnect_clears_stored_credentials(self):
        store = FakeStore()
        service = self.build_service(store=store)
        encrypted = service._encrypt_refresh_token("refresh-token")
        store.upsert_connection(
            subject_email="soolik1990@gmail.com",
            refresh_token_encrypted=encrypted,
            scopes=["https://www.googleapis.com/auth/drive"],
            token_uri="https://oauth2.googleapis.com/token",
        )

        status = service.disconnect()

        self.assertFalse(status.connected)
        self.assertTrue(store.cleared)

    def test_invalid_fernet_key_raises_google_oauth_error(self):
        service = GoogleDriveOAuthService(
            store=FakeStore(),
            config=GoogleDriveOAuthConfig(
                client_id="client-id",
                client_secret="client-secret",
                redirect_uri="https://example.com/v1/auth/google-drive/callback",
                state_secret="state-secret-that-is-definitely-long-enough",
                token_encryption_key="not-a-valid-fernet-key",
            ),
            http_post=lambda *args, **kwargs: FakeResponse(payload={}),
            http_get=lambda *args, **kwargs: FakeResponse(payload={}),
        )

        with self.assertRaisesRegex(RuntimeError, "token encryption key is invalid"):
            service._encrypt_refresh_token("refresh-token")


if __name__ == "__main__":
    unittest.main()
