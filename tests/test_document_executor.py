import unittest

from app.document_executor import DocumentExecutor
from app.models_v2 import ActionType, DocumentAction, DocumentRef, WorldEntityType


class FakeDriveStore:
    def __init__(self):
        self.docs = {
            "doc-1": "# NPC\n\n## Notes\n\nStara notatka.\n",
        }
        self.created = []
        self.section_updates = []

    def find_doc(self, *, folder=None, title=None, doc_id=None):
        if doc_id == "doc-1":
            return DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="doc-1", path_hint="03 NPC/Captain Mira")
        if folder == "03 NPC" and title == "Captain Mira":
            return DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="doc-1", path_hint="03 NPC/Captain Mira")
        return None

    def read_doc(self, doc_ref):
        return self.docs[doc_ref.doc_id]

    def create_doc(self, *, folder, title, content, entity_type):
        self.created.append({"folder": folder, "title": title, "content": content, "entity_type": entity_type})
        self.docs["doc-2"] = content
        return DocumentRef(folder=folder, title=title, doc_id="doc-2", path_hint=f"{folder}/{title}")

    def replace_section(self, doc_ref, section, content):
        self.section_updates.append({"doc_id": doc_ref.doc_id, "section": section, "content": content})
        self.docs[doc_ref.doc_id] = "# NPC\n\n## Notes\n\nNowa notatka.\n"

    def replace_doc(self, doc_ref, content):
        self.docs[doc_ref.doc_id] = content

    def append_doc(self, doc_ref, content):
        self.docs[doc_ref.doc_id] += content


class DocumentExecutorTest(unittest.TestCase):
    def test_preview_replace_section_contains_diff(self):
        executor = DocumentExecutor(drive_store=FakeDriveStore())
        action = DocumentAction(
            action_type=ActionType.replace_section,
            entity_type=WorldEntityType.npc,
            target=DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="doc-1"),
            section="Notes",
            content="Nowa notatka.",
        )

        prepared = executor.preview_action(action)

        self.assertIn("Nowa notatka.", prepared.proposed_text)
        self.assertIn("--- current", prepared.preview.diff_text)

    def test_apply_prepared_verifies_targeted_section_update(self):
        store = FakeDriveStore()
        executor = DocumentExecutor(drive_store=store)
        action = DocumentAction(
            action_type=ActionType.replace_section,
            entity_type=WorldEntityType.npc,
            target=DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="doc-1"),
            section="Notes",
            content="Nowa notatka.",
        )

        result = executor.apply_prepared(executor.preview_action(action))

        self.assertTrue(result.success)
        self.assertEqual(store.section_updates[0]["section"], "Notes")

    def test_apply_prepared_create_doc_returns_resolved_target(self):
        store = FakeDriveStore()
        executor = DocumentExecutor(drive_store=store)
        action = DocumentAction(
            action_type=ActionType.create_doc,
            entity_type=WorldEntityType.location,
            target=DocumentRef(folder="04 Locations", title="Port Peril"),
            content="# Port Peril",
        )

        result = executor.apply_prepared(executor.preview_action(action))

        self.assertTrue(result.success)
        self.assertEqual(result.target.doc_id, "doc-2")


if __name__ == "__main__":
    unittest.main()
