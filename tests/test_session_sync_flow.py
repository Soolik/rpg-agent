import unittest

import main


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


if __name__ == "__main__":
    unittest.main()
