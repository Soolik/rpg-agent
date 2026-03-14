import unittest
from datetime import datetime, timezone

from app.models_v2 import SessionPatchPayload, SyncSessionPatchRequest
from app.world_model_store import WorldModelStore, normalize_key


class FakeCursor:
    def __init__(self, fetchone_values):
        self.fetchone_values = list(fetchone_values)
        self.statements = []
        self.fetchall_result = []

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


class WorldModelStoreTest(unittest.TestCase):
    def test_normalize_key_lowercases_and_collapses_spaces(self):
        self.assertEqual(normalize_key("  Captain   Mira "), "captain mira")

    def test_sync_session_patch_inserts_session_and_upserts_world_rows(self):
        cursor = FakeCursor(fetchone_values=[(41,)])
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        request = SyncSessionPatchRequest(
            patch=SessionPatchPayload(
                session_summary="Mira revealed a secret.",
                thread_tracker_patch=[{"thread_id": "T01", "title": "Red Blade", "status": "active", "change": "Mira works with them."}],
                entities_patch=[{"kind": "npc", "name": "Captain Mira", "description": "Secret ally of Red Blade.", "tags": ["captain", "red-blade"]}],
                rag_additions=["Captain Mira works with Red Blade."],
            ),
            raw_notes="Raw notes",
            source_title="Session 05",
        )

        response = store.sync_session_patch(request)

        self.assertEqual(response.session_id, 41)
        self.assertEqual(response.entity_count, 1)
        self.assertEqual(response.thread_count, 1)
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("insert into world_sessions", cursor.statements[0][0].lower())
        self.assertIn("insert into world_entities", cursor.statements[1][0].lower())
        self.assertIn("insert into world_threads", cursor.statements[2][0].lower())

    def test_list_entities_maps_rows(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall_result = [
            (1, "kng", "npc", "Captain Mira", "Secret ally", ["tag-1"], 7, now),
        ]
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        rows = store.list_entities(limit=5, kind="npc")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].entity_kind, "npc")
        self.assertEqual(rows[0].last_session_id, 7)
        self.assertIn("from world_entities", cursor.statements[0][0].lower())

    def test_list_threads_maps_rows(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall_result = [
            (2, "kng", "T01", "T01", "Red Blade", "active", "Changed", 9, now),
        ]
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        rows = store.list_threads(limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].thread_key, "T01")
        self.assertEqual(rows[0].status, "active")
        self.assertIn("from world_threads", cursor.statements[0][0].lower())

    def test_list_sessions_maps_rows(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall_result = [
            (3, "kng", "Summary", 2, 1, "Session 05", now),
        ]
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        rows = store.list_sessions(limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].entity_count, 2)
        self.assertEqual(rows[0].source_title, "Session 05")
        self.assertIn("from world_sessions", cursor.statements[0][0].lower())

    def test_status_counts_rows(self):
        cursor = FakeCursor(fetchone_values=[(4,), (5,), (6,)])
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        status = store.status()

        self.assertEqual(status.entity_count, 4)
        self.assertEqual(status.thread_count, 5)
        self.assertEqual(status.session_count, 6)

    def test_cleanup_duplicate_threads_deletes_shadow_thread_when_canonical_match_exists(self):
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall_result = [
            (2, "mira's allegiances", None, "Mira's Allegiances", "Updated", "Captain Mira revealed a secret contact with Red Blade."),
            (1, "T01", "T01", "Red Blade", "active", "Captain Mira works with Red Blade."),
        ]
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        response = store.cleanup_duplicate_threads(dry_run=False)

        self.assertEqual(response.duplicate_thread_count, 1)
        self.assertEqual(response.deleted_thread_ids, [2])
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("delete from world_threads", cursor.statements[1][0].lower())
        self.assertEqual(cursor.statements[1][1], ("kng", [2]))

    def test_cleanup_duplicate_threads_dry_run_reports_without_deleting(self):
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall_result = [
            (2, "mira's allegiances", None, "Mira's Allegiances", "Updated", "Captain Mira revealed a secret contact with Red Blade."),
            (1, "T01", "T01", "Red Blade", "active", "Captain Mira works with Red Blade."),
        ]
        connection = FakeConnection(cursor)
        store = WorldModelStore(campaign_id="kng", connection_factory=lambda: connection)

        response = store.cleanup_duplicate_threads(dry_run=True)

        self.assertEqual(response.duplicate_thread_count, 1)
        self.assertEqual(response.deleted_thread_ids, [2])
        self.assertEqual(connection.commit_called, 0)
        self.assertEqual(len(cursor.statements), 1)


if __name__ == "__main__":
    unittest.main()
