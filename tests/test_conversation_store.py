import unittest
from datetime import datetime, timezone

from app.conversation_store import ConversationStore


class FakeCursor:
    def __init__(self, fetchone_values=None):
        self.fetchone_values = list(fetchone_values or [])
        self.fetchall_result = []
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.statements.append((query, params))

    def fetchone(self):
        if self.fetchone_values:
            return self.fetchone_values.pop(0)
        return None

    def fetchall(self):
        return self.fetchall_result


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


class ConversationStoreTest(unittest.TestCase):
    def test_create_conversation_bootstraps_schema_and_returns_record(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(
            fetchone_values=[
                ("conv-1", "kng", "Plan sesji", {"source": "test"}, now, now, None),
            ]
        )
        connection = FakeConnection(cursor)
        store = ConversationStore(campaign_id="kng", connection_factory=lambda: connection)

        record = store.create_conversation(title="Plan sesji", metadata={"source": "test"})

        self.assertEqual(record.conversation_id, "conv-1")
        self.assertEqual(record.title, "Plan sesji")
        self.assertEqual(record.metadata["source"], "test")
        self.assertEqual(connection.commit_called, 2)
        self.assertTrue(any("create table if not exists conversations" in query.lower() for query, _ in cursor.statements))
        self.assertTrue(any("insert into conversations" in query.lower() for query, _ in cursor.statements))

    def test_append_message_updates_parent_conversation(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(
            fetchone_values=[
                (7, "conv-1", "kng", "assistant", "answer", "npc_brief", "OK", {"continuity_ok": True}, now),
            ]
        )
        connection = FakeConnection(cursor)
        store = ConversationStore(campaign_id="kng", connection_factory=lambda: connection)
        store._schema_ready = True

        row = store.append_message(
            "conv-1",
            role="assistant",
            content="OK",
            kind="answer",
            artifact_type="npc_brief",
            metadata={"continuity_ok": True},
        )

        self.assertEqual(row.message_id, 7)
        self.assertEqual(row.role, "assistant")
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("insert into conversation_messages", cursor.statements[0][0].lower())
        self.assertIn("update conversations", cursor.statements[1][0].lower())

    def test_list_messages_maps_rows(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor()
        cursor.fetchall_result = [
            (1, "conv-1", "kng", "user", "input", None, "Kim jest Mira?", {"source_title": None}, now),
            (2, "conv-1", "kng", "assistant", "answer", None, "Dowodzi garnizonem.", {}, now),
        ]
        connection = FakeConnection(cursor)
        store = ConversationStore(campaign_id="kng", connection_factory=lambda: connection)
        store._schema_ready = True

        rows = store.list_messages("conv-1", limit=10)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].role, "user")
        self.assertEqual(rows[1].content, "Dowodzi garnizonem.")
        self.assertIn("from conversation_messages", cursor.statements[0][0].lower())

    def test_update_conversation_metadata_merges_patch(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(
            fetchone_values=[
                ("conv-1", "kng", "Plan sesji", {"source": "test"}, now, now, None, 2),
                ("conv-1", "kng", "Plan sesji", {"source": "test", "summary_text": "PODSUMOWANIE"}, now, now, None, 2),
            ]
        )
        connection = FakeConnection(cursor)
        store = ConversationStore(campaign_id="kng", connection_factory=lambda: connection)
        store._schema_ready = True

        record = store.update_conversation_metadata("conv-1", metadata_patch={"summary_text": "PODSUMOWANIE"})

        self.assertEqual(record.metadata["source"], "test")
        self.assertEqual(record.metadata["summary_text"], "PODSUMOWANIE")
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("update conversations", cursor.statements[1][0].lower())


if __name__ == "__main__":
    unittest.main()
