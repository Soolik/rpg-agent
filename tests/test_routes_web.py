import unittest
from types import SimpleNamespace

from app.routes_web import build_web_router
from app.request_auth import SignedSessionAuth


class RoutesWebTest(unittest.TestCase):
    def route_endpoint(self, router, path, method):
        for route in router.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        raise AssertionError(f"Route {method} {path} not found")

    def test_root_redirects_to_gm(self):
        router = build_web_router(google_client_id="client-id.apps.googleusercontent.com")

        response = self.route_endpoint(router, "/", "GET")()

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/gm")

    def test_gm_page_contains_title_and_session_ui(self):
        router = build_web_router(google_client_id="client-id.apps.googleusercontent.com")

        response = self.route_endpoint(router, "/gm", "GET")(request=SimpleNamespace(cookies={}))
        body = response.body.decode("utf-8")

        self.assertIn("AI Czat MG", body)
        self.assertIn("Polacz konto Google", body)
        self.assertIn("/auth/session/status", body)
        self.assertIn('const byId = (id) => document.getElementById(id);', body)
        self.assertIn('loginBtn: byId("loginBtn")', body)
        self.assertIn('"apiBase": "/v1"', body)
        self.assertIn('"initialSession"', body)
        self.assertNotIn("&quot;apiBase&quot;", body)
        self.assertIn('e.message.addEventListener("keydown"', body)
        self.assertIn('if (ev.key !== "Enter" || ev.shiftKey || ev.isComposing) return;', body)
        self.assertIn('include_sources: true', body)
        self.assertIn("window.location.assign('/v1/auth/google-drive/start'); return false;", body)
        self.assertIn('window.addEventListener("focus"', body)
        self.assertIn('window.addEventListener("pageshow"', body)
        self.assertIn("/v1", body)

    def test_gm_page_renders_authenticated_session_from_cookie(self):
        session_auth = SignedSessionAuth(secret="session-secret-that-is-definitely-long-enough")
        router = build_web_router(
            google_client_id="client-id.apps.googleusercontent.com",
            session_auth=session_auth,
        )
        request = SimpleNamespace(
            cookies={
                session_auth.cookie_name: session_auth.issue(
                    email="soolik1990@gmail.com",
                    subject="user-123",
                )
            }
        )

        response = self.route_endpoint(router, "/gm", "GET")(request=request)
        body = response.body.decode("utf-8")

        self.assertIn("Zalogowano: soolik1990@gmail.com", body)
        self.assertIn('"authenticated": true', body)
        self.assertIn('"email": "soolik1990@gmail.com"', body)


if __name__ == "__main__":
    unittest.main()
