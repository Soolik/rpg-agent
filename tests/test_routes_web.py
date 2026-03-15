import unittest

from app.routes_web import build_web_router


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

    def test_gm_page_contains_web_client_id_and_title(self):
        router = build_web_router(google_client_id="client-id.apps.googleusercontent.com")

        response = self.route_endpoint(router, "/gm", "GET")()
        body = response.body.decode("utf-8")

        self.assertIn("AI Czat MG", body)
        self.assertIn("client-id.apps.googleusercontent.com", body)
        self.assertIn("/v1", body)


if __name__ == "__main__":
    unittest.main()
