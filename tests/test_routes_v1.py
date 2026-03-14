import unittest

from app.api_models import (
    ProposalStatus,
    V1ArtifactGenerateRequest,
    WorldModelChangeDecisionRequest,
    WorldModelChangeProposalRequest,
)
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


class RoutesV1Test(unittest.TestCase):
    def build_router(self, chat_fn=None, workflow_store=None):
        return build_v1_router(
            chat_request_cls=ChatRequest,
            chat_fn=chat_fn or (lambda req: ChatResponse(kind="answer", reply="OK", references=[])),
            health_fn=lambda: {"ok": True, "campaign_id": "kng", "revision": "rev-1"},
            drive_store=FakeDriveStore(),
            planner=FakePlanner(),
            workflow_store=workflow_store or FakeWorkflowStore(),
            world_model_store=FakeWorldModelStore(),
            applier=FakeApplier(),
        )

    def route_endpoint(self, router, path, method):
        for route in router.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        raise AssertionError(f"Route {method} {path} not found")

    def test_v1_health_returns_trace_fields(self):
        router = self.build_router()

        body = self.route_endpoint(router, "/v1/health", "GET")().model_dump(mode="json")

        self.assertTrue(body["ok"])
        self.assertTrue(body["request_id"])
        self.assertEqual(body["request_id"], body["trace_id"])

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
