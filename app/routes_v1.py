from __future__ import annotations

import uuid
from typing import Callable, Optional, Type

from fastapi import APIRouter, HTTPException

from .api_models import (
    ContinuityReport,
    ProposalStatus,
    ProposalType,
    RequestTrace,
    SavedOutputRef,
    V1ArtifactGenerateRequest,
    V1ChatRequest,
    V1ChatResponse,
    V1HealthResponse,
    V1SessionPrepRequest,
    WorldModelChangeApplyResponse,
    WorldModelChangeDecisionRequest,
    WorldModelChangeListResponse,
    WorldModelChangeProposalRequest,
    WorldModelChangeResponse,
    WorldModelChangeView,
    WorldModelEntityListResponse,
    WorldModelSearchItem,
    WorldModelSearchResponse,
    WorldModelSessionListResponse,
    WorldModelThreadListResponse,
)
from .applier import ProposalApplier
from .canon_guard import build_continuity_report, normalize_key
from .chat_models import ChatRequest, ChatResponse
from .models_v2 import ApplyChangesRequest, ChangeProposal, ProposeChangesRequest
from .routes_v2 import build_context_for_planner
from .world_model_store import NullWorldModelStore, WorldModelStore
from .workflow_store import NullWorkflowStore, WorkflowStore


def _new_trace() -> RequestTrace:
    trace_id = uuid.uuid4().hex
    return RequestTrace(request_id=trace_id, trace_id=trace_id)


def _api_error(status_code: int, *, request_trace: RequestTrace, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "request_id": request_trace.request_id,
            "trace_id": request_trace.trace_id,
        },
    )


def _saved_output_from_chat(response: ChatResponse) -> Optional[SavedOutputRef]:
    if not (response.output_doc_id and response.output_title and response.output_path):
        return None
    return SavedOutputRef(
        doc_id=response.output_doc_id,
        title=response.output_title,
        path=response.output_path,
    )


def _continuity_for_response(
    *,
    message: str,
    response: ChatResponse,
    world_model_store: WorldModelStore | NullWorldModelStore,
) -> Optional[ContinuityReport]:
    text = response.artifact_text or response.reply
    if not text:
        return None
    entities = world_model_store.list_entities(limit=200)
    threads = world_model_store.list_threads(limit=200)
    allow_new_names = response.artifact_type == "npc_brief"
    return build_continuity_report(
        message=message,
        generated_text=text,
        known_entity_names=[entity.name for entity in entities],
        known_thread_names=[thread.title for thread in threads],
        extra_allowed_names=response.references,
        allow_proposed_new_names=allow_new_names,
    )


def _proposal_view_from_detail(detail) -> WorldModelChangeView:
    payload = detail.proposal or {}
    proposal = ChangeProposal.model_validate(payload)
    proposal_type = payload.get("proposal_type", ProposalType.general.value)
    proposal_status = payload.get("proposal_status", ProposalStatus.proposed.value)
    return WorldModelChangeView(
        proposal_id=payload.get("proposal_id") or detail.id,
        proposal_type=ProposalType(proposal_type),
        status=ProposalStatus(proposal_status),
        summary=detail.summary,
        user_goal=detail.user_goal,
        assumptions=proposal.assumptions,
        impacted_docs=proposal.impacted_docs,
        actions=proposal.actions,
        needs_confirmation=proposal.needs_confirmation,
        approved=detail.approved,
        approved_by=detail.approved_by,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
        supersedes_proposal_id=payload.get("supersedes_proposal_id"),
        accepted_apply_run_id=payload.get("accepted_apply_run_id"),
        rejected_reason=payload.get("rejected_reason"),
        reviewed_by=payload.get("reviewed_by"),
        request=detail.request,
        raw_proposal=payload,
    )


def _chat_response_v1(
    *,
    trace: RequestTrace,
    message: str,
    response: ChatResponse,
    world_model_store: WorldModelStore | NullWorldModelStore,
) -> V1ChatResponse:
    return V1ChatResponse(
        request_id=trace.request_id,
        trace_id=trace.trace_id,
        kind=response.kind,
        reply=response.reply,
        artifact_type=response.artifact_type,
        artifact_text=response.artifact_text,
        proposal_id=response.proposal_id,
        session_id=response.session_id,
        citations=response.references,
        warnings=response.warnings,
        output=_saved_output_from_chat(response),
        telemetry=response.telemetry,
        continuity=_continuity_for_response(
            message=message,
            response=response,
            world_model_store=world_model_store,
        ),
    )


def _search_world_model(
    query: str,
    *,
    world_model_store: WorldModelStore | NullWorldModelStore,
    limit: int,
) -> list[WorldModelSearchItem]:
    qkey = normalize_key(query)
    if not qkey:
        return []

    items: list[WorldModelSearchItem] = []

    for entity in world_model_store.list_entities(limit=max(limit * 3, 50)):
        haystack = normalize_key(" ".join(filter(None, [entity.name, entity.description, " ".join(entity.tags)])))
        if qkey not in haystack:
            continue
        score = 120 if qkey == normalize_key(entity.name) else 90 if qkey in normalize_key(entity.name) else 60
        items.append(
            WorldModelSearchItem(
                record_type="entity",
                record_id=entity.id,
                title=entity.name,
                snippet=entity.description[:220],
                entity_kind=entity.entity_kind,
                score=score,
            )
        )

    for thread in world_model_store.list_threads(limit=max(limit * 3, 50)):
        haystack = normalize_key(" ".join(filter(None, [thread.thread_id or "", thread.title, thread.last_change, thread.status or ""])))
        if qkey not in haystack:
            continue
        score = 115 if qkey == normalize_key(thread.title) else 85 if qkey in normalize_key(thread.title) else 55
        items.append(
            WorldModelSearchItem(
                record_type="thread",
                record_id=thread.id,
                title=thread.title,
                snippet=thread.last_change[:220],
                status=thread.status,
                score=score,
            )
        )

    for session in world_model_store.list_sessions(limit=max(limit * 3, 50)):
        haystack = normalize_key(" ".join(filter(None, [session.session_summary, session.source_title or ""])))
        if qkey not in haystack:
            continue
        score = 70 if qkey in normalize_key(session.source_title or "") else 50
        items.append(
            WorldModelSearchItem(
                record_type="session",
                record_id=session.id,
                title=session.source_title or f"Session {session.id}",
                snippet=session.session_summary[:220],
                source_title=session.source_title,
                score=score,
            )
        )

    items.sort(key=lambda item: (item.score, item.record_id), reverse=True)
    return items[:limit]


def build_v1_router(
    *,
    chat_request_cls: Type[ChatRequest],
    chat_fn: Callable[[ChatRequest], ChatResponse],
    health_fn: Callable[[], dict],
    drive_store,
    planner,
    workflow_store: Optional[WorkflowStore | NullWorkflowStore] = None,
    world_model_store: Optional[WorldModelStore | NullWorldModelStore] = None,
    applier: Optional[ProposalApplier] = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])
    store = workflow_store or NullWorkflowStore()
    model_store = world_model_store or NullWorldModelStore()
    proposal_applier = applier or ProposalApplier(drive_store=drive_store)

    @router.get("/health", response_model=V1HealthResponse)
    def v1_health():
        trace = _new_trace()
        payload = health_fn()
        return V1HealthResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            ok=bool(payload.get("ok")),
            campaign_id=payload.get("campaign_id") or "",
            revision=payload.get("revision") or "unknown",
        )

    @router.post("/chat", response_model=V1ChatResponse)
    def v1_chat(request: V1ChatRequest):
        trace = _new_trace()
        response = chat_fn(
            chat_request_cls(
                message=request.message,
                intent=request.intent,
                artifact_type=request.artifact_type,
                source_title=request.source_title,
                include_sources=request.include_sources,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
            )
        )
        return _chat_response_v1(
            trace=trace,
            message=request.message,
            response=response,
            world_model_store=model_store,
        )

    @router.post("/artifacts/generate", response_model=V1ChatResponse)
    def v1_generate_artifact(request: V1ArtifactGenerateRequest):
        trace = _new_trace()
        response = chat_fn(
            chat_request_cls(
                message=request.message,
                intent="auto",
                artifact_type=request.artifact_type,
                include_sources=request.include_sources,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
            )
        )
        return _chat_response_v1(
            trace=trace,
            message=request.message,
            response=response,
            world_model_store=model_store,
        )

    @router.post("/sessions/prep", response_model=V1ChatResponse)
    def v1_prepare_session(request: V1SessionPrepRequest):
        trace = _new_trace()
        response = chat_fn(
            chat_request_cls(
                message=request.message,
                intent="auto",
                artifact_type="pre_session_brief",
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
            )
        )
        return _chat_response_v1(
            trace=trace,
            message=request.message,
            response=response,
            world_model_store=model_store,
        )

    @router.get("/world-model/entities", response_model=WorldModelEntityListResponse)
    def v1_list_entities(limit: int = 20, kind: Optional[str] = None):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        return WorldModelEntityListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=model_store.list_entities(limit=safe_limit, kind=kind),
        )

    @router.get("/world-model/threads", response_model=WorldModelThreadListResponse)
    def v1_list_threads(limit: int = 20, status: Optional[str] = None):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        return WorldModelThreadListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=model_store.list_threads(limit=safe_limit, status=status),
        )

    @router.get("/world-model/sessions", response_model=WorldModelSessionListResponse)
    def v1_list_sessions(limit: int = 20):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        return WorldModelSessionListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=model_store.list_sessions(limit=safe_limit),
        )

    @router.get("/world-model/search", response_model=WorldModelSearchResponse)
    def v1_search_world_model(q: str, limit: int = 20):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 50))
        items = _search_world_model(q, world_model_store=model_store, limit=safe_limit)
        return WorldModelSearchResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            query=q,
            items=items,
        )

    @router.post("/world-model/changes/propose", response_model=WorldModelChangeResponse)
    def v1_propose_world_model_change(request: WorldModelChangeProposalRequest):
        trace = _new_trace()
        docs = drive_store.list_world_docs()
        context = build_context_for_planner(drive_store)
        proposal_request = ProposeChangesRequest(
            instruction=request.instruction,
            mode=request.mode,
            dry_run=request.dry_run,
        )
        proposal = planner.propose(request=proposal_request, world_docs=docs, world_context=context)
        proposal_id = store.save_proposal(
            proposal_request,
            proposal,
            proposal_type=ProposalType.world_model_change.value,
            proposal_status=ProposalStatus.proposed.value,
            supersedes_proposal_id=request.supersedes_proposal_id,
        )
        detail = store.get_proposal(proposal_id)
        if not detail:
            raise _api_error(
                500,
                request_trace=trace,
                code="proposal_not_persisted",
                message="Proposal was generated but could not be loaded back from workflow store.",
            )
        return WorldModelChangeResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=_proposal_view_from_detail(detail),
        )

    @router.get("/world-model/changes", response_model=WorldModelChangeListResponse)
    def v1_list_world_model_changes(
        limit: int = 20,
        status: Optional[ProposalStatus] = None,
    ):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        details = store.list_proposal_details(
            limit=safe_limit,
            proposal_type=ProposalType.world_model_change.value,
            proposal_status=status.value if status else None,
        )
        return WorldModelChangeListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=[_proposal_view_from_detail(detail) for detail in details],
        )

    @router.get("/world-model/changes/{proposal_id}", response_model=WorldModelChangeResponse)
    def v1_get_world_model_change(proposal_id: int):
        trace = _new_trace()
        detail = store.get_proposal(proposal_id)
        if not detail:
            raise _api_error(
                404,
                request_trace=trace,
                code="proposal_not_found",
                message="World model change proposal not found.",
            )
        return WorldModelChangeResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=_proposal_view_from_detail(detail),
        )

    @router.post("/world-model/changes/{proposal_id}/accept", response_model=WorldModelChangeApplyResponse)
    def v1_accept_world_model_change(proposal_id: int, request: WorldModelChangeDecisionRequest):
        trace = _new_trace()
        detail = store.get_proposal(proposal_id)
        if not detail:
            raise _api_error(
                404,
                request_trace=trace,
                code="proposal_not_found",
                message="World model change proposal not found.",
            )

        proposal = ChangeProposal.model_validate(detail.proposal)
        apply_request = ApplyChangesRequest(
            proposal_id=proposal_id,
            proposal=proposal,
            approved=True,
            approved_by=request.actor,
            reindex_after_apply=request.reindex_after_apply,
        )
        apply_response = proposal_applier.apply(apply_request)
        apply_response.proposal_id = proposal_id
        apply_run_id = store.save_apply_run(apply_request, apply_response)

        updated_detail = detail
        if apply_response.ok:
            updated_detail = store.update_proposal_state(
                proposal_id,
                proposal_status=ProposalStatus.accepted.value,
                reviewed_by=request.actor,
                accepted_apply_run_id=apply_run_id,
            ) or detail
            supersedes_proposal_id = detail.proposal.get("supersedes_proposal_id")
            if supersedes_proposal_id:
                store.update_proposal_state(
                    int(supersedes_proposal_id),
                    proposal_status=ProposalStatus.superseded.value,
                    reviewed_by=request.actor,
                )
        return WorldModelChangeApplyResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=_proposal_view_from_detail(updated_detail),
            apply_run_id=apply_run_id,
            ok=apply_response.ok,
            summary=apply_response.summary,
            results=apply_response.results,
            reindex_result=apply_response.reindex_result,
        )

    @router.post("/world-model/changes/{proposal_id}/reject", response_model=WorldModelChangeResponse)
    def v1_reject_world_model_change(proposal_id: int, request: WorldModelChangeDecisionRequest):
        trace = _new_trace()
        updated_detail = store.update_proposal_state(
            proposal_id,
            proposal_status=ProposalStatus.rejected.value,
            reviewed_by=request.actor,
            rejected_reason=request.reason,
        )
        if not updated_detail:
            raise _api_error(
                404,
                request_trace=trace,
                code="proposal_not_found",
                message="World model change proposal not found.",
            )
        return WorldModelChangeResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=_proposal_view_from_detail(updated_detail),
        )

    return router
