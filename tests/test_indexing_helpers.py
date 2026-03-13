import unittest

import main
from app.models_v2 import DocumentRef, WorldDocInfo, WorldEntityType


class IndexingHelpersTest(unittest.TestCase):
    def test_content_hash_is_stable(self):
        value = "Alpha Beta Gamma"
        self.assertEqual(main.content_hash(value), main.content_hash(value))

    def test_content_hash_changes_with_content(self):
        self.assertNotEqual(main.content_hash("alpha"), main.content_hash("beta"))

    def test_doc_type_for_thread_tracker_uses_threads(self):
        doc = WorldDocInfo(
            folder="06 Threads",
            title="Thread Tracker",
            doc_id="doc-1",
            entity_type=WorldEntityType.thread,
        )
        self.assertEqual(main.doc_type_for_indexing(doc), "threads")

    def test_doc_type_for_entity_uses_entity_type(self):
        doc = WorldDocInfo(
            folder="03 NPC",
            title="Captain Mira",
            doc_id="doc-2",
            entity_type=WorldEntityType.npc,
        )
        self.assertEqual(main.doc_type_for_indexing(doc), "npc")

    def test_chunk_text_splits_large_input(self):
        text = "a" * 5000
        chunks = main.chunk_text(text, max_chars=2400, overlap=400)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(chunk) for chunk in chunks), 2400)

    def test_resolve_reindex_docs_deduplicates_found_docs(self):
        class FakeDriveStore:
            def find_doc(self, *, folder=None, title=None, doc_id=None):
                if doc_id == "doc-1" or (folder == "03 NPC" and title == "Captain Mira"):
                    return WorldDocInfo(
                        folder="03 NPC",
                        title="Captain Mira",
                        doc_id="doc-1",
                        entity_type=WorldEntityType.npc,
                    )
                return None

        original = main.drive_store_v2
        try:
            main.drive_store_v2 = FakeDriveStore()
            docs = main.resolve_reindex_docs(
                [
                    DocumentRef(folder="03 NPC", title="Captain Mira"),
                    DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="doc-1"),
                ]
            )
        finally:
            main.drive_store_v2 = original

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].doc_id, "doc-1")

    def test_build_campaign_context_includes_doc_metadata(self):
        context = main.build_campaign_context(
            [
                {
                    "doc_type": "bible",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "chunk_text": "## Test Automation\n\nTen wpis potwierdza dzialanie.",
                    "distance": 0.1,
                    "title": "Campaign Bible",
                    "folder": "01 Bible",
                    "path_hint": "01 Bible/Campaign Bible",
                }
            ]
        )

        self.assertIn("title=Campaign Bible", context)
        self.assertIn("folder=01 Bible", context)
        self.assertIn("## Test Automation", context)

    def test_ctx_slice_keeps_tail_for_long_chunks(self):
        text = ("a" * 1200) + "## Test Automation\nNowa tresc" + ("b" * 1200)
        sliced = main.ctx_slice({"doc_type": "bible", "chunk_text": text})

        self.assertIn("## Test Automation", sliced)
        self.assertIn("...\n", sliced)


if __name__ == "__main__":
    unittest.main()
