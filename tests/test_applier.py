import unittest

from app.applier import ProposalApplier
from app.models_v2 import (
    ActionType,
    ApplyChangesRequest,
    ChangeProposal,
    DocumentAction,
    DocumentRef,
)


class FakeDriveStore:
    def __init__(self):
        self.appended = []

    def append_doc(self, target, content):
        self.appended.append((target, content))


class ProposalApplierTest(unittest.TestCase):
    def test_apply_uses_partial_reindex_targets(self):
        captured = {}

        def fake_reindex(targets):
            captured["targets"] = targets
            return {"ok": True, "mode": "partial", "indexed_docs": len(targets)}

        applier = ProposalApplier(drive_store=FakeDriveStore(), reindex_fn=fake_reindex)
        request = ApplyChangesRequest(
            proposal=ChangeProposal(
                summary="Append note",
                user_goal="Update NPC",
                actions=[
                    DocumentAction(
                        action_type=ActionType.append_doc,
                        target=DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="doc-1"),
                        content="New note",
                    )
                ],
            ),
            reindex_after_apply=True,
        )

        response = applier.apply(request)

        self.assertTrue(response.ok)
        self.assertEqual(response.reindex_result["mode"], "partial")
        self.assertEqual(len(captured["targets"]), 1)
        self.assertEqual(captured["targets"][0].doc_id, "doc-1")


if __name__ == "__main__":
    unittest.main()
