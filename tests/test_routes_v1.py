import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from app.api_models import (
    AssistantActionRequest,
    AssistantActionType,
    AssistantMode,
    CanonicalImportRequest,
    ConversationCreateRequest,
    ConversationMessageCreateRequest,
    ProposalStatus,
    V1ArtifactGenerateRequest,
    V1ChatRequest,
    WorldModelChangeDecisionRequest,
    WorldModelChangeProposalRequest,
)
from app.conversation_store import ConversationMessageRecord, ConversationRecord
from app.conversation_store import NullConversationStore
from app.chat_models import ChatRequest, ChatResponse
from app.models_v2 import (
    ApplyChangesResponse,
    AppliedActionResult,
    ChangeProposal,
    DocumentAction,
    DocumentRef,
    ProposalDetail,
    WorldEntityRecord,
    WorldEntityType,
    WorldSessionRecord,
    WorldThreadRecord,
)
from app.routes_v1 import build_v1_router
from app.chat_service import DirectChatStream, StreamPlan
from app.drive_store import DriveFileInfo
from app.request_auth import SignedSessionAuth


class FakeDriveStore:
    def __init__(self):
        self.docs = {}
        self.created = []
        self.replaced = []
        self.drive_folder_files = {}
        self.drive_file_text = {}

    def list_world_docs(self):
        return list(self.docs.values())

    def find_doc(self, *, folder=None, title=None, doc_id=None):
        if doc_id:
            for doc in self.docs.values():
                if doc.doc_id == doc_id:
                    return doc
            return None
        if folder and title:
            return self.docs.get((folder, title))
        return None

    def create_doc(self, *, folder, title, content, entity_type):
        doc = DocumentRef(folder=folder, title=title, doc_id=f"doc-{len(self.created) + 1}", path_hint=f"{folder}/{title}")
        self.docs[(folder, title)] = doc
        self.created.append({"folder": folder, "title": title, "content": content, "entity_type": entity_type})
        return doc

    def replace_doc(self, doc_ref, content):
        self.replaced.append({"doc_ref": doc_ref, "content": content})

    def list_drive_folder_files(self, folder_id):
        return self.drive_folder_files.get(folder_id, [])

    def read_drive_file_text(self, file_id, mime_type):
        return self.drive_file_text[file_id]


class FailingCreateDriveStore(FakeDriveStore):
    def create_doc(self, *, folder, title, content, entity_type):
        raise RuntimeError("quota exceeded for test")


class FakeGoogleDriveOAuthService:
    def __init__(self, *, configured=True, connected=False, subject_email=None):
        self.configured = configured
        self.connected = connected
        self.subject_email = subject_email
        self.start_called = False
        self.disconnect_called = False
        self.callback_called = None

    def get_status(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            configured=self.configured,
            connected=self.connected,
            subject_email=self.subject_email,
            scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"] if self.configured else [],
            redirect_uri="https://example.com/v1/auth/google-drive/callback" if self.configured else None,
            write_mode="user_oauth" if self.configured else "service_account",
        )

    def start_authorization(self):
        self.start_called = True
        from types import SimpleNamespace

        return SimpleNamespace(
            authorization_url="https://accounts.google.com/o/oauth2/v2/auth?client_id=test-client",
            redirect_uri="https://example.com/v1/auth/google-drive/callback",
            scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"],
        )

    def handle_callback(self, *, code, state):
        self.callback_called = {"code": code, "state": state}
        self.connected = True
        self.subject_email = "soolik1990@gmail.com"
        from types import SimpleNamespace

        return SimpleNamespace(
            status=self.get_status(),
            html_body="<html><body>connected</body></html>",
            subject_email="soolik1990@gmail.com",
            subject_id="user-123",
        )

    def disconnect(self):
        self.disconnect_called = True
        self.connected = False
        self.subject_email = None
        return self.get_status()


class FakePlanner:
    def propose(self, request, world_docs, world_context):
        return ChangeProposal(
            summary="Dodaj wpis o Captain Mira.",
            user_goal=request.instruction,
            impacted_docs=[DocumentRef(folder="03 NPC", title="Captain Mira")],
            actions=[
                DocumentAction(
                    action_type="replace_section",
                    entity_type=WorldEntityType.npc,
                    target=DocumentRef(folder="03 NPC", title="Captain Mira", doc_id="npc-1"),
                    section="## Notes",
                    content="Nowa notatka.",
                    reason="Aktualizacja kanonu",
                )
            ],
            needs_confirmation=True,
        )

    def consistency_check(self, instruction, world_context):
        return "Konfliktow krytycznych brak, ale pojawia sie nowa nazwa wlasna do decyzji."


class FakeWorldModelStore:
    def list_entities(self, limit=20, kind=None):
        return [
            WorldEntityRecord(
                id=1,
                campaign_id="kng",
                entity_kind="npc",
                name="Captain Mira",
                description="Dowodzi garnizonem.",
                tags=[],
                last_session_id=None,
                updated_at="2026-03-14T00:00:00+00:00",
            )
        ]

    def list_threads(self, limit=20, status=None):
        return [
            WorldThreadRecord(
                id=2,
                campaign_id="kng",
                thread_key="T01",
                thread_id="T01",
                title="Red Blade",
                status="active",
                last_change="Red Blade naciska na Captain Mira.",
                last_session_id=None,
                updated_at="2026-03-14T00:00:00+00:00",
            )
        ]

    def list_sessions(self, limit=20):
        return [
            WorldSessionRecord(
                id=3,
                campaign_id="kng",
                session_summary="Captain Mira ujawnila kontakt z Red Blade.",
                entity_count=1,
                thread_count=1,
                source_title="Session 06",
                created_at="2026-03-14T00:00:00+00:00",
            )
        ]


class FakeWorkflowStore:
    def __init__(self):
        self.proposals = {
            12: ProposalDetail(
                id=12,
                campaign_id="kng",
                summary="Stara propozycja",
                user_goal="Poprzednia wersja",
                approved=False,
                approved_by=None,
                created_at="2026-03-14T00:00:00+00:00",
                updated_at="2026-03-14T00:00:00+00:00",
                request={"instruction": "Stara zmiana"},
                proposal={
                    "proposal_id": 12,
                    "proposal_type": "world_model_change",
                    "proposal_status": "proposed",
                    "summary": "Stara propozycja",
                    "user_goal": "Poprzednia wersja",
                    "impacted_docs": [
                        {"folder": "03 NPC", "title": "Captain Mira", "doc_id": "npc-1"}
                    ],
                    "actions": [
                        {
                            "action_type": "replace_section",
                            "entity_type": "npc",
                            "target": {"folder": "03 NPC", "title": "Captain Mira", "doc_id": "npc-1"},
                            "section": "## Notes",
                            "content": "Stara notatka.",
                            "reason": "Fixture testowa",
                        }
                    ],
                    "needs_confirmation": True,
                },
            )
        }
        self.next_id = 20
        self.next_apply_run_id = 50

    def save_proposal(self, request, proposal, *, proposal_type="general", proposal_status="proposed", supersedes_proposal_id=None):
        proposal_id = self.next_id
        self.next_id += 1
        payload = {
            **proposal.model_dump(mode="json"),
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "proposal_status": proposal_status,
            "supersedes_proposal_id": supersedes_proposal_id,
            "accepted_apply_run_id": None,
            "rejected_reason": None,
            "reviewed_by": None,
        }
        self.proposals[proposal_id] = ProposalDetail(
            id=proposal_id,
            campaign_id="kng",
            summary=proposal.summary,
            user_goal=proposal.user_goal,
            approved=False,
            approved_by=None,
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            request=request.model_dump(mode="json"),
            proposal=payload,
        )
        return proposal_id

    def get_proposal(self, proposal_id):
        return self.proposals.get(proposal_id)

    def list_proposal_details(self, limit=20, *, proposal_type=None, proposal_status=None):
        items = list(self.proposals.values())
        if proposal_type:
            items = [item for item in items if item.proposal.get("proposal_type") == proposal_type]
        if proposal_status:
            items = [item for item in items if item.proposal.get("proposal_status") == proposal_status]
        return items[:limit]

    def update_proposal_state(self, proposal_id, *, proposal_status, reviewed_by=None, rejected_reason=None, accepted_apply_run_id=None):
        detail = self.proposals.get(proposal_id)
        if not detail:
            return None
        detail.proposal["proposal_status"] = proposal_status
        detail.proposal["reviewed_by"] = reviewed_by
        detail.proposal["rejected_reason"] = rejected_reason
        detail.proposal["accepted_apply_run_id"] = accepted_apply_run_id
        detail.approved = proposal_status == "accepted"
        detail.approved_by = reviewed_by
        self.proposals[proposal_id] = detail
        return detail

    def save_apply_run(self, request, response):
        run_id = self.next_apply_run_id
        self.next_apply_run_id += 1
        return run_id


class FakeApplier:
    def apply(self, request):
        return ApplyChangesResponse(
            ok=True,
            summary="Apply finished",
            results=[
                AppliedActionResult(
                    action_type=request.proposal.actions[0].action_type,
                    success=True,
                    message="Section replaced",
                    target=request.proposal.actions[0].target,
                )
            ],
            reindex_result={"ok": True, "mode": "partial"},
        )


class FakeConversationStore:
    def __init__(self):
        self.conversations = {}
        self.messages = {}
        self.next_id = 1

    def create_conversation(self, *, title="", metadata=None):
        conversation_id = f"conv-{self.next_id}"
        self.next_id += 1
        record = ConversationRecord(
            conversation_id=conversation_id,
            campaign_id="kng",
            title=title or "Nowa rozmowa",
            message_count=0,
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            last_message_at=None,
            metadata=metadata or {},
        )
        self.conversations[conversation_id] = record
        self.messages[conversation_id] = []
        return record

    def get_conversation(self, conversation_id):
        return self.conversations.get(conversation_id)

    def list_conversations(self, limit=20):
        return list(self.conversations.values())[:limit]

    def append_message(self, conversation_id, *, role, content, kind=None, artifact_type=None, metadata=None):
        items = self.messages.setdefault(conversation_id, [])
        message = ConversationMessageRecord(
            message_id=len(items) + 1,
            conversation_id=conversation_id,
            campaign_id="kng",
            role=role,
            content=content,
            kind=kind,
            artifact_type=artifact_type,
            created_at="2026-03-14T00:00:00+00:00",
            metadata=metadata or {},
        )
        items.append(message)
        convo = self.conversations[conversation_id]
        convo.message_count = len(items)
        convo.last_message_at = message.created_at
        convo.updated_at = message.created_at
        return message

    def list_messages(self, conversation_id, limit=100):
        return self.messages.get(conversation_id, [])[:limit]

    def update_conversation_metadata(self, conversation_id, *, metadata_patch):
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None
        conversation.metadata = {
            **(conversation.metadata or {}),
            **(metadata_patch or {}),
        }
        return conversation


class RoutesV1Test(unittest.TestCase):
    def build_router(self, chat_fn=None, chat_stream_fn=None, workflow_store=None, conversation_store=None, drive_store=None, reindex_fn=None, oauth_service=None):
        return build_v1_router(
            chat_request_cls=ChatRequest,
            chat_fn=chat_fn or (lambda req: ChatResponse(kind="answer", reply="OK", references=[])),
            chat_stream_fn=chat_stream_fn,
            health_fn=lambda: {"ok": True, "campaign_id": "kng", "revision": "rev-1"},
            drive_store=drive_store or FakeDriveStore(),
            planner=FakePlanner(),
            workflow_store=workflow_store or FakeWorkflowStore(),
            world_model_store=FakeWorldModelStore(),
            conversation_store=conversation_store or FakeConversationStore(),
            applier=FakeApplier(),
            reindex_fn=reindex_fn,
            google_drive_oauth_service=oauth_service,
            session_auth=SignedSessionAuth(secret="session-secret-that-is-definitely-long-enough"),
        )

    def route_endpoint(self, router, path, method):
        for route in router.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        raise AssertionError(f"Route {method} {path} not found")

    async def collect_stream(self, response):
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    def test_v1_health_returns_trace_fields(self):
        router = self.build_router()

        body = self.route_endpoint(router, "/v1/health", "GET")().model_dump(mode="json")

        self.assertTrue(body["ok"])
        self.assertTrue(body["request_id"])
        self.assertEqual(body["request_id"], body["trace_id"])

    def test_v1_google_drive_auth_status_reports_connection(self):
        router = self.build_router(oauth_service=FakeGoogleDriveOAuthService(connected=True, subject_email="soolik1990@gmail.com"))

        body = self.route_endpoint(router, "/v1/auth/google-drive/status", "GET")().model_dump(mode="json")

        self.assertTrue(body["configured"])
        self.assertTrue(body["connected"])
        self.assertEqual(body["subject_email"], "soolik1990@gmail.com")
        self.assertEqual(body["write_mode"], "user_oauth")

    def test_v1_google_drive_auth_start_returns_authorization_url(self):
        oauth_service = FakeGoogleDriveOAuthService()
        router = self.build_router(oauth_service=oauth_service)

        body = self.route_endpoint(router, "/v1/auth/google-drive/start", "POST")().model_dump(mode="json")

        self.assertTrue(oauth_service.start_called)
        self.assertIn("accounts.google.com", body["authorization_url"])

    def test_v1_google_drive_auth_disconnect_returns_status(self):
        oauth_service = FakeGoogleDriveOAuthService(connected=True, subject_email="soolik1990@gmail.com")
        router = self.build_router(oauth_service=oauth_service)

        body = self.route_endpoint(router, "/v1/auth/google-drive/disconnect", "POST")().model_dump(mode="json")

        self.assertTrue(oauth_service.disconnect_called)
        self.assertFalse(body["connected"])

    def test_v1_session_status_reports_cookie_authentication(self):
        router = self.build_router()
        session_auth = SignedSessionAuth(secret="session-secret-that-is-definitely-long-enough")
        request = SimpleNamespace(cookies={session_auth.cookie_name: session_auth.issue(email="soolik1990@gmail.com", subject="user-123")})

        body = self.route_endpoint(router, "/v1/auth/session/status", "GET")(request=request).model_dump(mode="json")

        self.assertTrue(body["authenticated"])
        self.assertEqual(body["email"], "soolik1990@gmail.com")

    def test_v1_session_logout_clears_cookie(self):
        router = self.build_router()

        response = self.route_endpoint(router, "/v1/auth/session/logout", "POST")()

        self.assertEqual(response.status_code, 200)
        self.assertIn("gm_session=", response.headers.get("set-cookie", ""))

    def test_v1_chat_auto_creates_conversation_and_persists_messages(self):
        seen = {}

        def fake_chat(req):
            seen["message"] = req.message
            seen["conversation_id"] = req.conversation_id
            return ChatResponse(kind="answer", reply="Captain Mira pracuje z Red Blade.", references=[])

        conversation_store = FakeConversationStore()
        router = self.build_router(chat_fn=fake_chat, conversation_store=conversation_store)

        body = self.route_endpoint(router, "/v1/chat", "POST")(
            request=V1ChatRequest(message="Kim jest Captain Mira?")
        ).model_dump(mode="json")

        self.assertEqual(body["reply_markdown"], "Captain Mira pracuje z Red Blade.")
        self.assertTrue(body["conversation_id"])
        self.assertTrue(seen["conversation_id"])
        self.assertEqual(seen["message"], "Kim jest Captain Mira?")
        self.assertEqual(len(conversation_store.list_messages(body["conversation_id"])), 2)
        self.assertTrue(any(action["type"] == "continue_conversation" for action in body["next_actions"]))

    def test_v1_chat_uses_prior_messages_as_memory_context(self):
        seen = {}

        def fake_chat(req):
            seen["message"] = req.message
            return ChatResponse(kind="answer", reply="Kontynuacja.", references=[])

        conversation_store = FakeConversationStore()
        conversation = conversation_store.create_conversation(title="Red Blade", metadata={})
        conversation_store.append_message(conversation.conversation_id, role="user", content="Powiedz mi o Red Blade.", kind="input")
        conversation_store.append_message(conversation.conversation_id, role="assistant", content="To frakcja.", kind="answer")
        router = self.build_router(chat_fn=fake_chat, conversation_store=conversation_store)

        self.route_endpoint(router, "/v1/chat", "POST")(
            request=V1ChatRequest(
                conversation_id=conversation.conversation_id,
                message="A jak to sie ma do Captain Mira?",
            )
        )

        self.assertIn("KONTEKST ROZMOWY:", seen["message"])
        self.assertIn("Powiedz mi o Red Blade.", seen["message"])
        self.assertIn("A jak to sie ma do Captain Mira?", seen["message"])

    def test_v1_chat_with_unknown_conversation_returns_404(self):
        router = self.build_router(conversation_store=FakeConversationStore())

        with self.assertRaises(HTTPException) as caught:
            self.route_endpoint(router, "/v1/chat", "POST")(
                request=V1ChatRequest(
                    conversation_id="missing-conversation",
                    message="Kontynuuj ten watek.",
                )
            )

        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail["code"], "conversation_not_found")

    def test_v1_chat_with_conversation_id_without_storage_returns_503(self):
        router = self.build_router(conversation_store=NullConversationStore())

        with self.assertRaises(HTTPException) as caught:
            self.route_endpoint(router, "/v1/chat", "POST")(
                request=V1ChatRequest(
                    conversation_id="conv-1",
                    message="Kontynuuj ten watek.",
                )
            )

        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail["code"], "conversation_store_unavailable")

    def test_v1_chat_stream_returns_sse_events(self):
        router = self.build_router(
            chat_fn=lambda req: ChatResponse(kind="answer", reply="Pierwszy akapit.\n\nDrugi akapit.", references=[])
        )

        response = self.route_endpoint(router, "/v1/chat", "POST")(
            request=V1ChatRequest(message="Stresc to", stream=True)
        )
        body = asyncio.run(self.collect_stream(response))

        self.assertIn("event: start", body)
        self.assertIn("event: delta", body)
        self.assertIn("event: complete", body)
        self.assertIn("Pierwszy akapit.", body)

    def test_v1_chat_stream_uses_direct_stream_when_available(self):
        seen = {}

        def fake_stream(req):
            seen["message"] = req.message
            return StreamPlan(
                selected_mode="direct",
                reason="test_direct_stream",
                handle=DirectChatStream(chunks=iter(["Pierwszy ", "token ", "streamu."])),
            )

        def fail_chat(_req):
            raise AssertionError("chat_fn should not be called for direct stream")

        router = self.build_router(chat_fn=fail_chat, chat_stream_fn=fake_stream)

        response = self.route_endpoint(router, "/v1/chat", "POST")(
            request=V1ChatRequest(message="Opowiedz krotko o idei tego API.", stream=True)
        )
        body = asyncio.run(self.collect_stream(response))

        self.assertEqual(seen["message"], "Opowiedz krotko o idei tego API.")
        self.assertIn('"selected_mode": "direct"', body)
        self.assertIn('"reason": "test_direct_stream"', body)
        self.assertIn("Pierwszy token streamu.", body)
        self.assertIn("event: complete", body)

    def test_v1_guard_mode_returns_guard_report_without_calling_chat(self):
        def fail_chat(_req):
            raise AssertionError("chat_fn should not be called for guard mode")

        router = self.build_router(chat_fn=fail_chat)

        body = self.route_endpoint(router, "/v1/chat", "POST")(
            request=V1ChatRequest(
                message="Sprawdz zgodnosc tego opisu z kanonem.",
                mode=AssistantMode.guard,
                candidate_text=(
                    "* **Captain Mira** - zawarla pakt przeciw Red Blade.\n"
                    "* **Skup** - wchodzi do gry jako nowa frakcja."
                ),
            )
        ).model_dump(mode="json")

        self.assertEqual(body["mode"], AssistantMode.guard.value)
        self.assertEqual(body["kind"], "answer")
        self.assertFalse(body["continuity"]["ok"])
        self.assertIn("Skup", body["continuity"]["proposed_new_names"])
        self.assertIn("Guard Report", body["reply_markdown"])
        self.assertIn("Konfliktow krytycznych brak", body["reply_markdown"])

    def test_v1_editor_mode_forces_proposal_intent(self):
        seen = {}

        def fake_chat(req):
            seen["intent"] = req.intent
            return ChatResponse(
                kind="proposal",
                reply="Plan zmian gotowy.",
                proposal_id=44,
                references=[],
            )

        router = self.build_router(chat_fn=fake_chat)

        body = self.route_endpoint(router, "/v1/chat", "POST")(
            request=V1ChatRequest(
                message="Dodaj nowego NPC powiazanego z Red Blade.",
                mode=AssistantMode.editor,
            )
        ).model_dump(mode="json")

        self.assertEqual(seen["intent"], "proposal")
        self.assertEqual(body["mode"], AssistantMode.editor.value)
        self.assertEqual(body["proposal_id"], 44)
        self.assertTrue(any(action["type"] == "accept_world_change" for action in body["next_actions"]))
        self.assertTrue(any(action["type"] == "reject_world_change" for action in body["next_actions"]))

    def test_v1_conversation_routes_return_saved_history(self):
        conversation_store = FakeConversationStore()
        router = self.build_router(conversation_store=conversation_store)

        create_body = self.route_endpoint(router, "/v1/conversations", "POST")(
            request=ConversationCreateRequest(title="Plan sesji")
        ).model_dump(mode="json")
        conversation_id = create_body["conversation"]["conversation_id"]

        self.route_endpoint(router, "/v1/conversations/{conversation_id}/messages", "POST")(
            conversation_id=conversation_id,
            request=ConversationMessageCreateRequest(message="Przygotuj 3 hooki."),
        )

        message_body = self.route_endpoint(router, "/v1/conversations/{conversation_id}/messages", "GET")(
            conversation_id=conversation_id
        ).model_dump(mode="json")

        self.assertEqual(message_body["conversation_id"], conversation_id)
        self.assertEqual(len(message_body["items"]), 2)
        self.assertEqual(message_body["items"][0]["role"], "user")
        self.assertEqual(message_body["items"][1]["role"], "assistant")

    def test_v1_assistant_action_accept_executes_proposal(self):
        workflow_store = FakeWorkflowStore()
        router = self.build_router(workflow_store=workflow_store)

        body = self.route_endpoint(router, "/v1/assistant/actions", "POST")(
            request=AssistantActionRequest(
                action_type=AssistantActionType.accept_world_change,
                proposal_id=12,
                actor="mg",
            )
        ).model_dump(mode="json")

        self.assertEqual(body["action_type"], AssistantActionType.accept_world_change.value)
        self.assertTrue(body["ok"])
        self.assertEqual(body["proposal"]["status"], ProposalStatus.accepted.value)
        self.assertIsNotNone(body["apply_run_id"])

    def test_v1_assistant_action_revise_uses_chat_service(self):
        router = self.build_router(
            chat_fn=lambda req: ChatResponse(kind="creative", reply="Nowa wersja.", references=[], artifact_type="scene_seed", artifact_text="## Scena\nNowa wersja.")
        )

        body = self.route_endpoint(router, "/v1/assistant/actions", "POST")(
            request=AssistantActionRequest(
                action_type=AssistantActionType.revise,
                message="Przerob to na ostrzejsza scene.",
                mode=AssistantMode.create,
                artifact_type="scene_seed",
            )
        ).model_dump(mode="json")

        self.assertEqual(body["action_type"], AssistantActionType.revise.value)
        self.assertTrue(body["ok"])
        self.assertEqual(body["chat"]["kind"], "creative")
        self.assertEqual(body["chat"]["artifact_type"], "scene_seed")

    def test_v1_assistant_action_confirm_executes_inferred_request(self):
        router = self.build_router(
            chat_fn=lambda req: ChatResponse(kind="creative", reply="Hooki gotowe.", references=[], artifact_type="session_hooks", artifact_text="Hook 1\nHook 2")
        )

        body = self.route_endpoint(router, "/v1/assistant/actions", "POST")(
            request=AssistantActionRequest(
                action_type=AssistantActionType.confirm_inferred_action,
                message="Przygotuj hooki i zapisz je.",
                mode=AssistantMode.create,
                artifact_type="session_hooks",
                save_output=True,
                output_title="Hooki 01",
            )
        ).model_dump(mode="json")

        self.assertEqual(body["action_type"], AssistantActionType.confirm_inferred_action.value)
        self.assertTrue(body["ok"])
        self.assertEqual(body["chat"]["kind"], "creative")

    def test_v1_artifact_generate_returns_continuity_report(self):
        def fake_chat(req):
            return ChatResponse(
                kind="answer",
                reply="Brief gotowy.",
                artifact_type="pre_session_brief",
                artifact_text=(
                    "# Pre-Session Brief\n\n## Key NPCs and Factions\n\n"
                    "* **Captain Mira** - jest pod presja.\n"
                    "* **Red Blade** - eskaluje konflikt.\n"
                    "* **Skup** - wchodzi do gry politycznej."
                ),
                references=["06 Threads / Thread Tracker"],
            )

        router = self.build_router(chat_fn=fake_chat)

        body = self.route_endpoint(router, "/v1/artifacts/generate", "POST")(
            request=V1ArtifactGenerateRequest(
                message="Przygotuj briefing przed sesja o Red Blade i Captain Mira.",
                artifact_type="pre_session_brief",
            )
        )
        if isinstance(body, dict):
            body = body
        else:
            body = body.model_dump(mode="json")

        self.assertEqual(body["artifact_type"], "pre_session_brief")
        self.assertEqual(body["artifact"]["artifact_type"], "pre_session_brief")
        self.assertFalse(body["continuity"]["ok"])
        self.assertIn("Skup", body["continuity"]["proposed_new_names"])

    def test_v1_world_model_search_returns_hits(self):
        router = self.build_router()

        body = self.route_endpoint(router, "/v1/world-model/search", "GET")(q="Red Blade").model_dump(mode="json")

        self.assertEqual(body["query"], "Red Blade")
        self.assertTrue(any(item["record_type"] == "thread" for item in body["items"]))
        self.assertTrue(any(item["record_type"] == "session" for item in body["items"]))

    def test_v1_canonical_import_dry_run_maps_local_files(self):
        drive_store = FakeDriveStore()
        reindex_calls = []
        router = self.build_router(
            drive_store=drive_store,
            reindex_fn=lambda targets: reindex_calls.append(targets) or {"ok": True},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Campaign Bible_2026_03_08_2046.txt").write_text("# Campaign Bible\n\nTest.", encoding="utf-8")
            (root / "NPCs_2026_03_08_2046.txt").write_text("Captain Mira", encoding="utf-8")
            (root / "README.txt").write_text("skip me", encoding="utf-8")

            body = self.route_endpoint(router, "/v1/imports/canonical-files", "POST")(
                request=CanonicalImportRequest(
                    source_path=str(root),
                    dry_run=True,
                    replace_existing=True,
                    reindex_after_import=True,
                )
            ).model_dump(mode="json")

        self.assertEqual(body["created_count"], 0)
        self.assertEqual(body["updated_count"], 0)
        self.assertEqual(body["imported_count"], 0)
        self.assertEqual(body["skipped_count"], 1)
        self.assertTrue(any(item["title"] == "Campaign Bible" and item["action"] == "create_doc" for item in body["results"]))
        self.assertTrue(any(item["folder"] == "03 NPC" and item["title"] == "NPCs" for item in body["results"]))
        self.assertEqual(reindex_calls, [])

    def test_v1_canonical_import_from_drive_folder_dry_run(self):
        drive_store = FakeDriveStore()
        drive_store.drive_folder_files["folder-123"] = [
            DriveFileInfo(file_id="file-1", name="Campaign Bible_2026_03_08_2046.docx", mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", parents=["folder-123"]),
            DriveFileInfo(file_id="file-2", name="README.txt", mime_type="text/plain", parents=["folder-123"]),
        ]
        drive_store.drive_file_text["file-1"] = "# Campaign Bible\n\nTekst z Drive."
        router = self.build_router(drive_store=drive_store)

        body = self.route_endpoint(router, "/v1/imports/canonical-files", "POST")(
            request=CanonicalImportRequest(
                source_drive_folder_id="folder-123",
                dry_run=True,
            )
        ).model_dump(mode="json")

        self.assertEqual(body["source_path"], "gdrive://folder-123")
        self.assertTrue(any(item["title"] == "Campaign Bible" and item["action"] == "create_doc" for item in body["results"]))
        self.assertTrue(any(item["source_name"] == "README.txt" and item["status"] == "skipped" for item in body["results"]))

    def test_v1_canonical_import_returns_partial_results_on_write_error(self):
        drive_store = FailingCreateDriveStore()
        drive_store.drive_folder_files["folder-123"] = [
            DriveFileInfo(file_id="file-1", name="NPCs_2026_03_08_2046.docx", mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", parents=["folder-123"]),
        ]
        drive_store.drive_file_text["file-1"] = "Captain Mira"
        router = self.build_router(drive_store=drive_store)

        body = self.route_endpoint(router, "/v1/imports/canonical-files", "POST")(
            request=CanonicalImportRequest(
                source_drive_folder_id="folder-123",
                dry_run=False,
            )
        ).model_dump(mode="json")

        self.assertEqual(body["imported_count"], 0)
        self.assertEqual(body["error_count"], 1)
        self.assertTrue(any(item["status"] == "error" for item in body["results"]))
        self.assertTrue(any("create_doc failed for 03 NPC/NPCs" in warning for warning in body["warnings"]))

    def test_v1_world_model_change_accept_marks_old_proposal_superseded(self):
        workflow_store = FakeWorkflowStore()
        router = self.build_router(workflow_store=workflow_store)

        propose_body = self.route_endpoint(router, "/v1/world-model/changes/propose", "POST")(
            request=WorldModelChangeProposalRequest(
                instruction="Zaktualizuj wpis Captain Mira.",
                supersedes_proposal_id=12,
            )
        )
        if not isinstance(propose_body, dict):
            propose_body = propose_body.model_dump(mode="json")
        proposal_id = propose_body["proposal"]["proposal_id"]

        accept_body = self.route_endpoint(router, "/v1/world-model/changes/{proposal_id}/accept", "POST")(
            proposal_id=proposal_id,
            request=WorldModelChangeDecisionRequest(actor="mg", reindex_after_apply=True),
        )
        if not isinstance(accept_body, dict):
            accept_body = accept_body.model_dump(mode="json")

        self.assertEqual(accept_body["proposal"]["status"], ProposalStatus.accepted.value)
        self.assertEqual(workflow_store.proposals[12].proposal["proposal_status"], ProposalStatus.superseded.value)

    def test_v1_world_model_change_reject_sets_status(self):
        workflow_store = FakeWorkflowStore()
        router = self.build_router(workflow_store=workflow_store)

        body = self.route_endpoint(router, "/v1/world-model/changes/{proposal_id}/reject", "POST")(
            proposal_id=12,
            request=WorldModelChangeDecisionRequest(actor="mg", reason="Za slabe uzasadnienie"),
        )
        if not isinstance(body, dict):
            body = body.model_dump(mode="json")

        self.assertEqual(body["proposal"]["status"], ProposalStatus.rejected.value)
        self.assertEqual(body["proposal"]["rejected_reason"], "Za slabe uzasadnienie")


if __name__ == "__main__":
    unittest.main()
