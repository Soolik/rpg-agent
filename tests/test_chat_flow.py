import unittest

import main
from app.models_v2 import ChangeProposal, DocumentRef


class ChatFlowTest(unittest.TestCase):
    def test_detect_chat_intent_prefers_proposal_markers(self):
        intent = main.detect_chat_intent(
            "W dokumencie Campaign Bible podmien sekcje Test Automation na nowy tekst."
        )
        self.assertEqual(intent, "proposal")

    def test_detect_chat_intent_recognizes_creative_request(self):
        intent = main.detect_chat_intent(
            "Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade."
        )
        self.assertEqual(intent, "creative")

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

    def test_chat_answer_can_render_gm_brief_artifact(self):
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
                    artifact_type="gm_brief",
                )
            )
        finally:
            main.ask = original_ask

        self.assertEqual(response.artifact_type, "gm_brief")
        self.assertIn("# GM Brief", response.artifact_text)
        self.assertIn("## Sources", response.artifact_text)
        self.assertIn("01 Bible / Campaign Bible", response.artifact_text)

    def test_chat_creative_returns_generated_artifact(self):
        original_generate_creative_artifact = main.generate_creative_artifact
        try:
            main.generate_creative_artifact = lambda **kwargs: (
                "Tytul:\nCienie Red Blade\n\nHook 1:\nKupiec znika tej samej nocy co poslaniec frakcji.",
                ["01 Bible / Campaign Bible", "06 Threads / Thread Tracker"],
            )

            response = main.chat(
                main.ChatRequest(
                    message="Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade.",
                    artifact_type="session_hooks",
                )
            )
        finally:
            main.generate_creative_artifact = original_generate_creative_artifact

        self.assertEqual(response.kind, "creative")
        self.assertEqual(response.artifact_type, "session_hooks")
        self.assertIn("Cienie Red Blade", response.reply)
        self.assertEqual(response.references, ["01 Bible / Campaign Bible", "06 Threads / Thread Tracker"])

    def test_chat_creative_defaults_to_session_hooks_when_artifact_not_given(self):
        original_generate_creative_artifact = main.generate_creative_artifact
        captured = {}

        def fake_generate_creative_artifact(**kwargs):
            captured.update(kwargs)
            return "Tytul:\nCienie Red Blade", []

        try:
            main.generate_creative_artifact = fake_generate_creative_artifact

            response = main.chat(
                main.ChatRequest(
                    message="Wymysl 3 hooki na nastepna sesje zwiazane z Red Blade.",
                )
            )
        finally:
            main.generate_creative_artifact = original_generate_creative_artifact

        self.assertEqual(response.kind, "creative")
        self.assertEqual(response.artifact_type, "session_hooks")
        self.assertEqual(captured["artifact_type"], "session_hooks")

    def test_chat_answer_can_save_output_doc(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2

        class FakeDriveStore:
            def __init__(self):
                self.created = None

            def find_doc(self, folder=None, title=None, doc_id=None):
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                self.created = {
                    "folder": folder,
                    "title": title,
                    "content": content,
                    "entity_type": entity_type,
                }
                return main.WorldDocInfo(
                    folder=folder,
                    title=title,
                    doc_id="out-1",
                    path_hint=f"{folder}/{title}",
                    entity_type=main.WorldEntityType.output,
                )

        fake_drive_store = FakeDriveStore()

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = fake_drive_store

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    save_output=True,
                    output_title="Answer 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store

        self.assertEqual(response.output_doc_id, "out-1")
        self.assertEqual(response.output_title, "Answer 01")
        self.assertEqual(fake_drive_store.created["folder"], "08 Outputs")
        self.assertEqual(fake_drive_store.created["title"], "Answer 01")
        self.assertEqual(fake_drive_store.created["content"], "Gotowy tekst.")

    def test_chat_answer_saves_artifact_content_when_requested(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2

        class FakeDriveStore:
            def __init__(self):
                self.created = None

            def find_doc(self, folder=None, title=None, doc_id=None):
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                self.created = {
                    "folder": folder,
                    "title": title,
                    "content": content,
                    "entity_type": entity_type,
                }
                return main.WorldDocInfo(
                    folder=folder,
                    title=title,
                    doc_id="out-2",
                    path_hint=f"{folder}/{title}",
                    entity_type=main.WorldEntityType.output,
                )

        fake_drive_store = FakeDriveStore()

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = fake_drive_store

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    artifact_type="player_summary",
                    save_output=True,
                    output_title="Player Summary 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store

        self.assertEqual(response.output_doc_id, "out-2")
        self.assertEqual(response.output_title, "Player Summary 01")
        self.assertIn("# Player Summary", fake_drive_store.created["content"])
        self.assertNotEqual(fake_drive_store.created["content"], "Gotowy tekst.")

    def test_chat_answer_returns_warning_when_output_save_fails(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2

        class FakeDriveStore:
            def find_doc(self, folder=None, title=None, doc_id=None):
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                raise RuntimeError("storageQuotaExceeded")

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = FakeDriveStore()

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    save_output=True,
                    output_title="Answer 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store

        self.assertEqual(response.kind, "answer")
        self.assertEqual(response.reply, "Gotowy tekst.")
        self.assertEqual(response.output_doc_id, None)
        self.assertEqual(len(response.warnings), 1)
        self.assertIn("storageQuotaExceeded", response.warnings[0])

    def test_chat_answer_falls_back_to_rollup_doc_on_storage_quota(self):
        original_ask = main.ask
        original_drive_store = main.drive_store_v2
        original_rollup_doc_id = main.OUTPUT_ROLLUP_DOC_ID
        original_rollup_doc_title = main.OUTPUT_ROLLUP_DOC_TITLE
        original_rollup_mode = main.OUTPUT_ROLLUP_MODE

        class FakeDriveStore:
            def __init__(self):
                self.replaced = None

            def find_doc(self, folder=None, title=None, doc_id=None):
                if doc_id == "rollup-1":
                    return main.WorldDocInfo(
                        folder="08 Outputs",
                        title="Agent Inbox",
                        doc_id="rollup-1",
                        path_hint="08 Outputs/Agent Inbox",
                        entity_type=main.WorldEntityType.output,
                    )
                return None

            def create_doc(self, folder, title, content, entity_type=None):
                raise RuntimeError("storageQuotaExceeded")

            def replace_doc(self, doc_ref, content):
                self.replaced = {"doc_ref": doc_ref, "content": content}

        fake_drive_store = FakeDriveStore()

        try:
            main.ask = lambda req: main.AskResponse(answer="Gotowy tekst.", sources=[])
            main.drive_store_v2 = fake_drive_store
            main.OUTPUT_ROLLUP_DOC_ID = "rollup-1"
            main.OUTPUT_ROLLUP_DOC_TITLE = "Agent Inbox"
            main.OUTPUT_ROLLUP_MODE = "replace"

            response = main.chat(
                main.ChatRequest(
                    message="Odpowiedz krotko na pytanie.",
                    intent="answer",
                    save_output=True,
                    output_title="Answer 01",
                )
            )
        finally:
            main.ask = original_ask
            main.drive_store_v2 = original_drive_store
            main.OUTPUT_ROLLUP_DOC_ID = original_rollup_doc_id
            main.OUTPUT_ROLLUP_DOC_TITLE = original_rollup_doc_title
            main.OUTPUT_ROLLUP_MODE = original_rollup_mode

        self.assertEqual(response.output_doc_id, "rollup-1")
        self.assertEqual(response.output_path, "08 Outputs/Agent Inbox")
        self.assertEqual(len(response.warnings), 1)
        self.assertIn("fallback dokumentu 08 Outputs/Agent Inbox", response.warnings[0])
        self.assertEqual(fake_drive_store.replaced["doc_ref"].doc_id, "rollup-1")
        self.assertEqual(fake_drive_store.replaced["content"], "Gotowy tekst.")

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

    def test_chat_session_sync_can_render_session_report_artifact(self):
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
                    rag_additions=["Captain Mira ma tajny kontakt z Red Blade."],
                ),
                sync=main.SyncSessionPatchResponse(
                    session_id=12,
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
                    artifact_type="session_report",
                )
            )
        finally:
            main.ingest_session_and_sync = original_sync

        self.assertEqual(response.artifact_type, "session_report")
        self.assertIn("# Session Report", response.artifact_text)
        self.assertIn("## Executive Summary", response.artifact_text)
        self.assertIn("## Threads", response.artifact_text)
        self.assertIn("## Facts For Retrieval", response.artifact_text)
        self.assertIn("## Suggested Document Follow-ups", response.artifact_text)
        self.assertIn("## Prep For Next Session", response.artifact_text)

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
