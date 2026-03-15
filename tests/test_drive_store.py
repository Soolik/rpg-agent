import unittest

from app.drive_store import DriveStore, decode_google_export_text, replace_section_content
from app.models_v2 import WorldEntityType


class DriveStoreSectionReplaceTest(unittest.TestCase):
    def test_decode_google_export_text_strips_bom_and_normalizes_newlines(self):
        raw = b"\xef\xbb\xbfLine 1\r\nLine 2\rLine 3"

        decoded = decode_google_export_text(raw)

        self.assertEqual(decoded, "Line 1\nLine 2\nLine 3")

    def test_decode_google_export_text_repairs_common_mojibake(self):
        raw = "Captain Mira \u00c5\u00bcada lojalnosci wobec Red Blade.\r\n".encode("utf-8")

        decoded = decode_google_export_text(raw)

        self.assertEqual(decoded, "Captain Mira \u017cada lojalnosci wobec Red Blade.\n")

    def test_replace_existing_section_body(self):
        original = (
            "# NPC\n\n"
            "## Identity\n\n"
            "- Name: Mira\n\n"
            "## Secrets\n\n"
            "Old secret\n\n"
            "## Relationships\n\n"
            "Trusted crew\n"
        )

        updated = replace_section_content(original, "Secrets", "New secret line")

        self.assertIn("## Secrets\n\nNew secret line\n", updated)
        self.assertNotIn("Old secret", updated)
        self.assertIn("## Relationships\n\nTrusted crew\n", updated)

    def test_append_section_when_missing(self):
        original = "# NPC\n\n## Identity\n\n- Name: Mira\n"

        updated = replace_section_content(original, "Motivations", "Protect the crew")

        self.assertTrue(updated.endswith("## Motivations\n\nProtect the crew\n"))

    def test_keep_nested_headings_inside_section(self):
        original = (
            "# NPC\n\n"
            "## Secrets\n\n"
            "Old secret\n\n"
            "### Known by\n\n"
            "Nobody\n\n"
            "## Relationships\n\n"
            "Trusted crew\n"
        )

        updated = replace_section_content(original, "Secrets", "Replaced secret")

        self.assertIn("## Secrets\n\nReplaced secret\n", updated)
        self.assertNotIn("### Known by", updated)
        self.assertIn("## Relationships\n\nTrusted crew\n", updated)

    def test_create_doc_creates_google_doc_in_target_folder_via_drive(self):
        class FakeRequest:
            def __init__(self, payload):
                self.payload = payload

            def execute(self):
                return self.payload

        class FakeFilesApi:
            def __init__(self):
                self.create_calls = []

            def create(self, **kwargs):
                self.create_calls.append(kwargs)
                return FakeRequest({"id": "doc-123", "name": kwargs["body"]["name"], "parents": kwargs["body"].get("parents", [])})

        class FakeDriveApi:
            def __init__(self):
                self.files_api = FakeFilesApi()

            def files(self):
                return self.files_api

        class FakeDocsDocumentsApi:
            def __init__(self):
                self.batch_calls = []

            def batchUpdate(self, **kwargs):
                self.batch_calls.append(kwargs)
                return FakeRequest({"ok": True})

        class FakeDocsApi:
            def __init__(self):
                self.documents_api = FakeDocsDocumentsApi()

            def documents(self):
                return self.documents_api

        class TestDriveStore(DriveStore):
            def __init__(self):
                super().__init__(folder_map={"03 NPC": "folder-123"}, core_doc_map={})
                self.fake_drive = FakeDriveApi()
                self.fake_docs = FakeDocsApi()

            def _drive(self):
                return self.fake_drive

            def _docs(self):
                return self.fake_docs

        store = TestDriveStore()

        created = store.create_doc(
            folder="03 NPC",
            title="Captain Mira",
            content="# NPC\n\n## Secrets\nWspolpracuje z Czerwonym Ostrzem.",
            entity_type=WorldEntityType.npc,
        )

        create_call = store.fake_drive.files_api.create_calls[0]
        self.assertEqual(create_call["body"]["mimeType"], "application/vnd.google-apps.document")
        self.assertEqual(create_call["body"]["parents"], ["folder-123"])
        self.assertEqual(created.doc_id, "doc-123")
        self.assertEqual(store.fake_docs.documents_api.batch_calls[0]["documentId"], "doc-123")

    def test_replace_section_uses_targeted_docs_api_update(self):
        class FakeRequest:
            def __init__(self, payload):
                self.payload = payload

            def execute(self):
                return self.payload

        class FakeDocumentsApi:
            def __init__(self):
                self.batch_calls = []

            def get(self, **kwargs):
                document = {
                    "body": {
                        "content": [
                            {
                                "startIndex": 1,
                                "endIndex": 10,
                                "paragraph": {"elements": [{"textRun": {"content": "# NPC\n"}}]},
                            },
                            {
                                "startIndex": 10,
                                "endIndex": 19,
                                "paragraph": {"elements": [{"textRun": {"content": "## Notes\n"}}]},
                            },
                            {
                                "startIndex": 19,
                                "endIndex": 33,
                                "paragraph": {"elements": [{"textRun": {"content": "Stara notatka\n"}}]},
                            },
                            {
                                "startIndex": 33,
                                "endIndex": 44,
                                "paragraph": {"elements": [{"textRun": {"content": "## Sekrety\n"}}]},
                            },
                            {
                                "startIndex": 44,
                                "endIndex": 52,
                                "paragraph": {"elements": [{"textRun": {"content": "Sekret\n"}}]},
                            },
                            {
                                "startIndex": 52,
                                "endIndex": 53,
                                "paragraph": {"elements": [{"textRun": {"content": "\n"}}]},
                            },
                        ]
                    }
                }
                return FakeRequest(document)

            def batchUpdate(self, **kwargs):
                self.batch_calls.append(kwargs)
                return FakeRequest({"ok": True})

        class FakeDocsApi:
            def __init__(self):
                self.documents_api = FakeDocumentsApi()

            def documents(self):
                return self.documents_api

        class TestDriveStore(DriveStore):
            def __init__(self):
                super().__init__(folder_map={}, core_doc_map={})
                self.fake_docs = FakeDocsApi()

            def _docs(self):
                return self.fake_docs

        store = TestDriveStore()

        store.replace_section(
            doc_ref=type("DocRef", (), {"folder": "03 NPC", "title": "Captain Mira", "doc_id": "doc-1", "path_hint": None})(),
            section="Notes",
            content="Nowa notatka.",
        )

        batch_call = store.fake_docs.documents_api.batch_calls[0]
        requests = batch_call["body"]["requests"]
        self.assertEqual(batch_call["documentId"], "doc-1")
        self.assertTrue(any("deleteContentRange" in request for request in requests))
        self.assertTrue(any(request.get("insertText", {}).get("text") == "\nNowa notatka.\n" for request in requests))


if __name__ == "__main__":
    unittest.main()
