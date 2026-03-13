import unittest

from app.drive_store import DriveStore, replace_section_content
from app.models_v2 import WorldEntityType


class DriveStoreSectionReplaceTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
