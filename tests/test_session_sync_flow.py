import unittest

import main
from app.models_v2 import WorldEntityRecord, WorldThreadRecord


class SessionSyncFlowTest(unittest.TestCase):
    def test_generate_session_patch_maps_model_output(self):
        original_generate = main.gemini_generate
        try:
            main.gemini_generate = lambda *args, **kwargs: """
            {
              "session_summary": "Captain Mira ujawnila sekret.",
              "thread_tracker_patch": [
                {"thread_id": "T01", "title": "Red Blade", "status": "active", "change": "Mira wspolpracuje z nimi."}
              ],
              "entities_patch": [
                {"kind": "npc", "name": "Captain Mira", "description": "Tajna wspolpracowniczka.", "tags": ["captain"]}
              ],
              "rag_additions": ["Captain Mira wspolpracuje z Red Blade."]
            }
            """

            patch = main.generate_session_patch("Raw notes")
        finally:
            main.gemini_generate = original_generate

        self.assertEqual(patch.session_summary, "Captain Mira ujawnila sekret.")
        self.assertEqual(len(patch.thread_tracker_patch), 1)
        self.assertEqual(patch.entities_patch[0].name, "Captain Mira")

    def test_generate_session_patch_includes_world_model_context(self):
        original_generate = main.gemini_generate
        original_store = main.world_model_store_v2
        captured = {}

        class FakeStore:
            def list_entities(self, limit=50, kind=None):
                return [
                    WorldEntityRecord(
                        id=1,
                        campaign_id="kng",
                        entity_kind="npc",
                        name="Captain Mira",
                        description="Desc",
                        tags=[],
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

            def list_threads(self, limit=50, status=None):
                return [
                    WorldThreadRecord(
                        id=1,
                        campaign_id="kng",
                        thread_key="T01",
                        thread_id="T01",
                        title="Red Blade",
                        status="active",
                        last_change="Change",
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

        def fake_generate(prompt, **kwargs):
            captured["prompt"] = prompt
            return """
            {
              "session_summary": "Captain Mira ujawnila sekret.",
              "thread_tracker_patch": [],
              "entities_patch": [],
              "rag_additions": []
            }
            """

        try:
            main.world_model_store_v2 = FakeStore()
            main.gemini_generate = fake_generate
            main.generate_session_patch("Raw notes")
        finally:
            main.gemini_generate = original_generate
            main.world_model_store_v2 = original_store

        self.assertIn("KNOWN ENTITIES:", captured["prompt"])
        self.assertIn("Captain Mira", captured["prompt"])
        self.assertIn("KNOWN THREADS:", captured["prompt"])
        self.assertIn("T01 | Red Blade", captured["prompt"])

    def test_ingest_session_and_sync_returns_patch_and_sync_result(self):
        original_generate = main.gemini_generate
        original_store = main.world_model_store_v2

        class FakeStore:
            def __init__(self):
                self.requests = []

            def sync_session_patch(self, request):
                self.requests.append(request)
                return main.SyncSessionPatchResponse(
                    session_id=7,
                    campaign_id="kng",
                    summary="Session patch synced into world model",
                    entity_count=1,
                    thread_count=1,
                )

        fake_store = FakeStore()

        try:
            main.gemini_generate = lambda *args, **kwargs: """
            {
              "session_summary": "Captain Mira ujawnila sekret.",
              "thread_tracker_patch": [
                {"thread_id": "T01", "title": "Red Blade", "status": "active", "change": "Mira wspolpracuje z nimi."}
              ],
              "entities_patch": [
                {"kind": "npc", "name": "Captain Mira", "description": "Tajna wspolpracowniczka.", "tags": ["captain"]}
              ],
              "rag_additions": ["Captain Mira wspolpracuje z Red Blade."]
            }
            """
            main.world_model_store_v2 = fake_store

            response = main.ingest_session_and_sync(
                main.IngestAndSyncSessionRequest(
                    raw_notes="Raw notes",
                    source_title="Session 05",
                )
            )
        finally:
            main.gemini_generate = original_generate
            main.world_model_store_v2 = original_store

        self.assertEqual(response.sync.session_id, 7)
        self.assertEqual(response.patch.entities_patch[0].name, "Captain Mira")
        self.assertEqual(fake_store.requests[0].source_title, "Session 05")

    def test_generate_session_patch_reuses_matching_known_thread(self):
        original_generate = main.gemini_generate
        original_store = main.world_model_store_v2

        class FakeStore:
            def list_entities(self, limit=50, kind=None):
                return []

            def list_threads(self, limit=50, status=None):
                return [
                    WorldThreadRecord(
                        id=2,
                        campaign_id="kng",
                        thread_key="mira's allegiances",
                        thread_id=None,
                        title="Mira's Allegiances",
                        status="Updated",
                        last_change="Old duplicate change",
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    ),
                    WorldThreadRecord(
                        id=1,
                        campaign_id="kng",
                        thread_key="T01",
                        thread_id="T01",
                        title="Red Blade",
                        status="active",
                        last_change="Old change",
                        last_session_id=None,
                        updated_at="2026-03-14T00:00:00+00:00",
                    )
                ]

        try:
            main.world_model_store_v2 = FakeStore()
            main.gemini_generate = lambda *args, **kwargs: """
            {
              "session_summary": "Captain Mira ujawnila sekret.",
              "thread_tracker_patch": [
                {"thread_id": null, "title": "Mira's Allegiances", "status": "Updated", "change": "Captain Mira ujawnila tajny kontakt z Red Blade."}
              ],
              "entities_patch": [],
              "rag_additions": []
            }
            """

            patch = main.generate_session_patch("Raw notes")
        finally:
            main.gemini_generate = original_generate
            main.world_model_store_v2 = original_store

        self.assertEqual(len(patch.thread_tracker_patch), 1)
        self.assertEqual(patch.thread_tracker_patch[0].thread_id, "T01")
        self.assertEqual(patch.thread_tracker_patch[0].title, "Red Blade")


if __name__ == "__main__":
    unittest.main()
