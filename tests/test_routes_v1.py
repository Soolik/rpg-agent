import asyncio
import unittest

from app.api_models import (
    ConversationCreateRequest,
    ConversationMessageCreateRequest,
    ProposalStatus,
    V1ArtifactGenerateRequest,
    V1ChatRequest,
    WorldModelChangeDecisionRequest,
    WorldModelChangeProposalRequest,
)
from app.conversation_store import ConversationMessageRecord, ConversationRecord
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


class FakeDriveStore:
    def list_world_docs(self):
        return []


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
                    "impacted_docs": [],
                    "actions": [],
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


class RoutesV1Test(unittest.TestCase):
    def build_router(self, chat_fn=None, workflow_store=None, conversation_store=None):
        return build_v1_router(
            chat_request_cls=ChatRequest,
            chat_fn=chat_fn or (lambda req: ChatResponse(kind="answer", reply="OK", references=[])),
            health_fn=lambda: {"ok": True, "campaign_id": "kng", "revision": "rev-1"},
            drive_store=FakeDriveStore(),
            planner=FakePlanner(),
            workflow_store=workflow_store or FakeWorkflowStore(),
            world_model_store=FakeWorldModelStore(),
            conversation_store=conversation_store or FakeConversationStore(),
            applier=FakeApplier(),
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
