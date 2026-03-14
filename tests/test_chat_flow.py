import unittest

import main
from app.models_v2 import ChangeProposal, DocumentRef


class ChatFlowTest(unittest.TestCase):
    def test_detect_chat_intent_prefers_proposal_markers(self):
        intent = main.detect_chat_intent(
            "W dokumencie Campaign Bible podmien sekcje Test Automation na nowy tekst."
        )
        self.assertEqual(intent, "proposal")

    def test_chat_answer_returns_human_text_with_sources(self):
        original_ask = main.ask
        try:
            main.ask = lambda req: main.AskResponse(
                answer="Ten wpis potwierdza, ze apply_changes dziala poprawnie.",
                sources=[
                    {
                        "folder": "01 Bible",
                        "title": "Campaign Bible",
                        "doc_id": "doc-1",
                    }
                ],
            )

            response = main.chat(
                main.ChatRequest(
                    message="Co mowi sekcja Test Automation?",
                    intent="answer",
                    include_sources=True,
                )
            )
        finally:
            main.ask = original_ask

        self.assertEqual(response.kind, "answer")
        self.assertIn("Ten wpis potwierdza", response.reply)
        self.assertIn("Zrodla:", response.reply)
        self.assertEqual(response.references, ["01 Bible / Campaign Bible"])

    def test_chat_session_sync_returns_human_summary(self):
        original_sync = main.ingest_session_and_sync
        try:
            main.ingest_session_and_sync = lambda req: main.IngestAndSyncSessionResponse(
                patch=main.SessionPatch(
                    session_summary="Captain Mira ujawnila tajny kontakt z Red Blade.",
                    thread_tracker_patch=[
                        main.ThreadPatch(
                            thread_id="T01",
                            title="Red Blade",
                            status="Updated",
                            change="Ujawniono tajny kontakt Captain Miry.",
                        )
                    ],
                    entities_patch=[
                        main.EntityPatch(
                            kind="npc",
                            name="Captain Mira",
                            description="Tajny kontakt Red Blade.",
                            tags=[],
                        )
                    ],
                    rag_additions=[],
                ),
                sync=main.SyncSessionPatchResponse(
                    session_id=11,
                    campaign_id="kng",
                    summary="Session patch synced into world model",
                    entity_count=1,
                    thread_count=1,
                ),
            )

            response = main.chat(
                main.ChatRequest(
                    message="Captain Mira ujawnila tajny kontakt z Red Blade.\nTo zmienia watek frakcji.",
                    intent="session_sync",
                    source_title="Session 06",
                )
            )
        finally:
            main.ingest_session_and_sync = original_sync

        self.assertEqual(response.kind, "session_sync")
        self.assertEqual(response.session_id, 11)
        self.assertIn("Zaktualizowalem model swiata z notatek.", response.reply)
        self.assertIn("T01 / Red Blade", response.reply)

    def test_chat_proposal_returns_human_summary(self):
        original_drive_store = main.drive_store_v2
        original_planner = main.planner_v2
        original_workflow_store = main.workflow_store_v2
        original_build_context = main.build_context_for_planner

        class FakeDriveStore:
            def list_world_docs(self):
                return []

        class FakePlanner:
            def propose(self, request, world_docs, world_context):
                return ChangeProposal(
                    summary="Podmien sekcje Test Automation w Campaign Bible.",
                    user_goal=request.instruction,
                    impacted_docs=[DocumentRef(folder="01 Bible", title="Campaign Bible")],
                    actions=[],
                    needs_confirmation=True,
                )

        class FakeWorkflowStore:
            def save_proposal(self, request, proposal):
                return 17

        try:
            main.drive_store_v2 = FakeDriveStore()
            main.planner_v2 = FakePlanner()
            main.workflow_store_v2 = FakeWorkflowStore()
            main.build_context_for_planner = lambda drive_store: "world context"

            response = main.chat(
                main.ChatRequest(
                    message="W dokumencie Campaign Bible podmien sekcje Test Automation na nowy tekst.",
                    intent="proposal",
                )
            )
        finally:
            main.drive_store_v2 = original_drive_store
            main.planner_v2 = original_planner
            main.workflow_store_v2 = original_workflow_store
            main.build_context_for_planner = original_build_context

        self.assertEqual(response.kind, "proposal")
        self.assertEqual(response.proposal_id, 17)
        self.assertIn("Przygotowalem propozycje zmiany.", response.reply)
        self.assertIn("Proposal ID: 17", response.reply)


if __name__ == "__main__":
    unittest.main()
