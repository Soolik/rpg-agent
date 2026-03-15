import unittest

from app.doc_canon import extract_doc_backed_entities, strip_reference_block, sync_doc_backed_entities
from app.models_v2 import WorldDocInfo, WorldEntityType


class FakeCursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.statements.append((query, params))


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.commit_called = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commit_called += 1


class DocCanonTest(unittest.TestCase):
    def test_extract_doc_backed_entities_prefers_campaign_names_from_docs(self):
        doc = WorldDocInfo(
            folder="02 Sessions",
            title="Dossier Morna - sprawa Black Eel",
            doc_id="doc-morn",
            path_hint="02 Sessions / Dossier Morna - sprawa Black Eel",
            entity_type=WorldEntityType.session,
        )
        text = (
            "Tavin Morn zachowal kopie dokumentow.\n"
            "Black Eel byl celem papierowego rabunku.\n"
            "Mercy Tide wraca jako falszywa nazwa statku.\n"
            "Port Peril jest miejscem, gdzie zaczyna sie sprawa.\n"
        )

        entities = extract_doc_backed_entities(doc, text)
        names = {entity.name for entity in entities}

        self.assertIn("Black Eel", names)
        self.assertIn("Mercy Tide", names)
        self.assertIn("Port Peril", names)
        self.assertIn("Tavin Morn", names)

    def test_sync_doc_backed_entities_replaces_rows_for_touched_docs(self):
        doc = WorldDocInfo(
            folder="04 Locations",
            title="Places",
            doc_id="doc-places",
            path_hint="04 Locations / Places",
            entity_type=WorldEntityType.location,
        )
        cursor = FakeCursor()
        connection = FakeConnection(cursor)

        created = sync_doc_backed_entities(
            campaign_id="kng",
            connection_factory=lambda: connection,
            docs_with_content=[(doc, "Port Peril to glowny port w Shackles.\n")],
        )

        self.assertGreaterEqual(created, 1)
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("delete from world_entities", cursor.statements[0][0].lower())
        inserts = [statement for statement in cursor.statements if "insert into world_entities" in statement[0].lower()]
        self.assertTrue(inserts)

    def test_strip_reference_block_removes_sources_footer(self):
        text = (
            "- Port Peril jest wazny.\n"
            "- Tavin Morn uruchamia sprawe.\n\n"
            "Zrodla:\n"
            "- 02 Sessions / Dossier Morna - sprawa Black Eel\n"
        )

        stripped = strip_reference_block(text)

        self.assertIn("Port Peril", stripped)
        self.assertNotIn("Zrodla", stripped)
        self.assertNotIn("Dossier Morna", stripped)


if __name__ == "__main__":
    unittest.main()
