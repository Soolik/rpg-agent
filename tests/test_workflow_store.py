import unittest
from datetime import datetime, timezone

from app.models_v2 import ApplyChangesRequest, ApplyChangesResponse, ChangeProposal, ProposeChangesRequest
from app.workflow_store import WorkflowStore


class FakeCursor:
    def __init__(self, fetchone_values):
        self.fetchone_values = list(fetchone_values)
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


class WorkflowStoreTest(unittest.TestCase):
    def test_save_proposal_returns_database_id(self):
        cursor = FakeCursor(fetchone_values=[(17,)])
        connection = FakeConnection(cursor)
        store = WorkflowStore(campaign_id="kng", connection_factory=lambda: connection)

        proposal = ChangeProposal(summary="Test", user_goal="Goal")
        proposal_id = store.save_proposal(
            request=ProposeChangesRequest(instruction="do it"),
            proposal=proposal,
        )

        self.assertEqual(proposal_id, 17)
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("insert into proposals", cursor.statements[0][0].lower())
        self.assertIn("update proposals", cursor.statements[1][0].lower())
        self.assertIn('"proposal_id": 17', cursor.statements[1][1][0])

    def test_save_apply_run_updates_linked_proposal(self):
        cursor = FakeCursor(fetchone_values=[(31,)])
        connection = FakeConnection(cursor)
        store = WorkflowStore(campaign_id="kng", connection_factory=lambda: connection)

        request = ApplyChangesRequest(
            proposal_id=12,
            proposal=ChangeProposal(summary="Test", user_goal="Goal", proposal_id=12),
            approved=True,
            approved_by="manager",
        )
        response = ApplyChangesResponse(
            ok=True,
            summary="Apply finished",
            proposal_id=12,
        )

        run_id = store.save_apply_run(request, response)

        self.assertEqual(run_id, 31)
        self.assertEqual(connection.commit_called, 1)
        self.assertIn("insert into apply_runs", cursor.statements[0][0].lower())
        self.assertIn("update proposals", cursor.statements[1][0].lower())

    def test_list_proposals_maps_rows(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall = lambda: [
            (1, "kng", "Summary", "Goal", True, "mgr", now, now),
        ]
        connection = FakeConnection(cursor)
        store = WorkflowStore(campaign_id="kng", connection_factory=lambda: connection)

        rows = store.list_proposals(limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, 1)
        self.assertTrue(rows[0].approved)
        self.assertIn("from proposals", cursor.statements[0][0].lower())

    def test_list_apply_runs_maps_rows(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(fetchone_values=[])
        cursor.fetchall = lambda: [
            (2, "kng", 1, True, "mgr", False, now),
        ]
        connection = FakeConnection(cursor)
        store = WorkflowStore(campaign_id="kng", connection_factory=lambda: connection)

        rows = store.list_apply_runs(limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].proposal_id, 1)
        self.assertFalse(rows[0].ok)
        self.assertIn("from apply_runs", cursor.statements[0][0].lower())

    def test_get_proposal_maps_detail_row(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(
            fetchone_values=[
                (
                    4,
                    "kng",
                    "Summary",
                    "Goal",
                    True,
                    "mgr",
                    now,
                    now,
                    {"instruction": "do it"},
                    {"summary": "Summary"},
                )
            ]
        )
        connection = FakeConnection(cursor)
        store = WorkflowStore(campaign_id="kng", connection_factory=lambda: connection)

        row = store.get_proposal(4)

        self.assertIsNotNone(row)
        self.assertEqual(row.id, 4)
        self.assertEqual(row.request["instruction"], "do it")
        self.assertIn("where campaign_id = %s and id = %s", cursor.statements[0][0].lower())

    def test_get_apply_run_maps_detail_row(self):
        now = datetime.now(timezone.utc)
        cursor = FakeCursor(
            fetchone_values=[
                (
                    5,
                    "kng",
                    4,
                    True,
                    "mgr",
                    True,
                    now,
                    {"approved": True},
                    {"ok": True},
                )
            ]
        )
        connection = FakeConnection(cursor)
        store = WorkflowStore(campaign_id="kng", connection_factory=lambda: connection)

        row = store.get_apply_run(5)

        self.assertIsNotNone(row)
        self.assertEqual(row.id, 5)
        self.assertTrue(row.response["ok"])
        self.assertIn("from apply_runs", cursor.statements[0][0].lower())


if __name__ == "__main__":
    unittest.main()
