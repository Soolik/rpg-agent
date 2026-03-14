from __future__ import annotations

import uuid
from typing import Callable, Optional, Type

from fastapi import APIRouter, HTTPException

from .api_models import (
    AssistantActionRequest,
    AssistantActionResponse,
    AssistantActionType,
    AssistantMode,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationMessageCreateRequest,
    ConversationMessageListResponse,
    ConversationResponse,
    ProposalStatus,
    ProposalType,
    RequestTrace,
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
from .chat_service import ChatService, DirectChatStream
from .canon_guard import normalize_key
from .chat_models import ChatRequest, ChatResponse
from .conversation_store import ConversationStore, NullConversationStore
from .world_model_store import NullWorldModelStore, WorldModelStore
from .workflow_store import NullWorkflowStore, WorkflowStore
from .workflow_service import WorkflowService


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
    chat_stream_fn: Optional[Callable[[ChatRequest], Optional[DirectChatStream]]] = None,
    health_fn: Callable[[], dict],
    drive_store,
    planner,
    consistency_planner=None,
    workflow_store: Optional[WorkflowStore | NullWorkflowStore] = None,
    world_model_store: Optional[WorldModelStore | NullWorldModelStore] = None,
    conversation_store: Optional[ConversationStore | NullConversationStore] = None,
    applier: Optional[ProposalApplier] = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])
    store = workflow_store or NullWorkflowStore()
    model_store = world_model_store or NullWorldModelStore()
    convo_store = conversation_store or NullConversationStore()
    guard_planner = consistency_planner or planner
    proposal_applier = applier or ProposalApplier(drive_store=drive_store)
    chat_service = ChatService(
        chat_request_cls=chat_request_cls,
        chat_fn=chat_fn,
        chat_stream_fn=chat_stream_fn,
        drive_store=drive_store,
        planner=planner,
        consistency_planner=guard_planner,
        world_model_store=model_store,
        conversation_store=convo_store,
    )
    workflow_service = WorkflowService(
        drive_store=drive_store,
        planner=planner,
        workflow_store=store,
        applier=proposal_applier,
    )

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

    @router.get("/conversations", response_model=ConversationListResponse)
    def v1_list_conversations(limit: int = 20):
        trace = _new_trace()
        if not chat_service.conversation_storage_enabled():
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        safe_limit = max(1, min(limit, 100))
        return ConversationListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=chat_service.list_conversations(limit=safe_limit),
        )

    @router.post("/conversations", response_model=ConversationResponse)
    def v1_create_conversation(request: ConversationCreateRequest):
        trace = _new_trace()
        if not chat_service.conversation_storage_enabled():
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        conversation = chat_service.create_conversation(
            title=request.title,
            seed_message="",
            metadata={"source": "v1_conversations"},
        )
        return ConversationResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            conversation=conversation,
        )

    @router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
    def v1_get_conversation(conversation_id: str):
        trace = _new_trace()
        if not chat_service.conversation_storage_enabled():
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        conversation = chat_service.get_conversation(conversation_id)
        if not conversation:
            raise _api_error(
                404,
                request_trace=trace,
                code="conversation_not_found",
                message="Conversation not found.",
            )
        return ConversationResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            conversation=conversation,
        )

    @router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessageListResponse)
    def v1_list_conversation_messages(conversation_id: str, limit: int = 100):
        trace = _new_trace()
        if not chat_service.conversation_storage_enabled():
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        conversation = chat_service.get_conversation(conversation_id)
        if not conversation:
            raise _api_error(
                404,
                request_trace=trace,
                code="conversation_not_found",
                message="Conversation not found.",
            )
        safe_limit = max(1, min(limit, 200))
        return ConversationMessageListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            conversation_id=conversation_id,
            items=chat_service.list_messages(conversation_id, limit=safe_limit),
        )

    @router.post("/chat", response_model=V1ChatResponse)
    def v1_chat(request: V1ChatRequest):
        trace = _new_trace()
        if request.stream:
            return chat_service.stream_run(
                trace=trace,
                message=request.message,
                assistant_mode=request.mode,
                intent=request.intent,
                artifact_type=request.artifact_type,
                source_title=request.source_title,
                candidate_text=request.candidate_text,
                include_sources=request.include_sources,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
                conversation_id=request.conversation_id,
                conversation_title=request.conversation_title,
            )
        response = chat_service.run(
            trace=trace,
            message=request.message,
            assistant_mode=request.mode,
            intent=request.intent,
            artifact_type=request.artifact_type,
            source_title=request.source_title,
            candidate_text=request.candidate_text,
            include_sources=request.include_sources,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=request.conversation_id,
            conversation_title=request.conversation_title,
        )
        return response

    @router.post("/conversations/{conversation_id}/messages", response_model=V1ChatResponse)
    def v1_conversation_message(conversation_id: str, request: ConversationMessageCreateRequest):
        trace = _new_trace()
        if request.stream:
            return chat_service.stream_run(
                trace=trace,
                message=request.message,
                assistant_mode=request.mode,
                intent=request.intent,
                artifact_type=request.artifact_type,
                source_title=request.source_title,
                candidate_text=request.candidate_text,
                include_sources=request.include_sources,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
                conversation_id=conversation_id,
                conversation_title=None,
            )
        response = chat_service.run(
            trace=trace,
            message=request.message,
            assistant_mode=request.mode,
            intent=request.intent,
            artifact_type=request.artifact_type,
            source_title=request.source_title,
            candidate_text=request.candidate_text,
            include_sources=request.include_sources,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=conversation_id,
            conversation_title=None,
        )
        return response

    @router.post("/artifacts/generate", response_model=V1ChatResponse)
    def v1_generate_artifact(request: V1ArtifactGenerateRequest):
        trace = _new_trace()
        if request.stream:
            return chat_service.stream_run(
                trace=trace,
                message=request.message,
                assistant_mode=AssistantMode.create,
                intent="auto",
                artifact_type=request.artifact_type,
                source_title=None,
                candidate_text=None,
                include_sources=request.include_sources,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
                conversation_id=request.conversation_id,
                conversation_title=request.conversation_title,
            )
        response = chat_service.run(
            trace=trace,
            message=request.message,
            assistant_mode=AssistantMode.create,
            intent="auto",
            artifact_type=request.artifact_type,
            source_title=None,
            candidate_text=None,
            include_sources=request.include_sources,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=request.conversation_id,
            conversation_title=request.conversation_title,
        )
        return response

    @router.post("/sessions/prep", response_model=V1ChatResponse)
    def v1_prepare_session(request: V1SessionPrepRequest):
        trace = _new_trace()
        if request.stream:
            return chat_service.stream_run(
                trace=trace,
                message=request.message,
                assistant_mode=AssistantMode.create,
                intent="auto",
                artifact_type="pre_session_brief",
                source_title=None,
                candidate_text=None,
                include_sources=False,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
                conversation_id=request.conversation_id,
                conversation_title=request.conversation_title,
            )
        response = chat_service.run(
            trace=trace,
            message=request.message,
            assistant_mode=AssistantMode.create,
            intent="auto",
            artifact_type="pre_session_brief",
            source_title=None,
            candidate_text=None,
            include_sources=False,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=request.conversation_id,
            conversation_title=request.conversation_title,
        )
        return response

    @router.post("/assistant/actions", response_model=AssistantActionResponse)
    def v1_assistant_action(request: AssistantActionRequest):
        trace = _new_trace()

        if request.action_type == AssistantActionType.accept_world_change:
            if request.proposal_id is None:
                raise _api_error(
                    400,
                    request_trace=trace,
                    code="missing_proposal_id",
                    message="proposal_id is required for accept_world_change.",
                )
            accepted = accept_world_model_change_impl(
                trace=trace,
                proposal_id=request.proposal_id,
                actor=request.actor,
                reindex_after_apply=request.reindex_after_apply,
            )
            return AssistantActionResponse(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                action_type=request.action_type,
                proposal=accepted.proposal,
                apply_run_id=accepted.apply_run_id,
                ok=accepted.ok,
                summary=accepted.summary,
                results=accepted.results,
                reindex_result=accepted.reindex_result,
            )

        if request.action_type == AssistantActionType.reject_world_change:
            if request.proposal_id is None:
                raise _api_error(
                    400,
                    request_trace=trace,
                    code="missing_proposal_id",
                    message="proposal_id is required for reject_world_change.",
                )
            rejected = reject_world_model_change_impl(
                trace=trace,
                proposal_id=request.proposal_id,
                actor=request.actor,
                reason=request.reason,
            )
            return AssistantActionResponse(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                action_type=request.action_type,
                proposal=rejected.proposal,
                ok=True,
                summary="World model change rejected.",
            )

        if request.action_type == AssistantActionType.revise:
            if not request.message:
                raise _api_error(
                    400,
                    request_trace=trace,
                    code="missing_message",
                    message="message is required for revise.",
                )
            chat_response = chat_service.run(
                trace=trace,
                message=request.message,
                assistant_mode=request.mode,
                intent="auto",
                artifact_type=request.artifact_type,
                source_title=request.source_title,
                candidate_text=request.candidate_text,
                include_sources=request.include_sources,
                include_telemetry=request.include_telemetry,
                save_output=request.save_output,
                output_title=request.output_title,
                conversation_id=request.conversation_id,
                conversation_title=None,
            )
            return AssistantActionResponse(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                action_type=request.action_type,
                ok=True,
                summary="Revision response generated.",
                chat=chat_response,
            )

        raise _api_error(
            400,
            request_trace=trace,
            code="unsupported_action",
            message="Unsupported assistant action.",
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

    def accept_world_model_change_impl(
        *,
        trace: RequestTrace,
        proposal_id: int,
        actor: Optional[str],
        reindex_after_apply: bool,
    ) -> WorldModelChangeApplyResponse:
        accepted = workflow_service.accept_change(
            proposal_id=proposal_id,
            actor=actor,
            reindex_after_apply=reindex_after_apply,
        )
        if not accepted:
            raise _api_error(
                404,
                request_trace=trace,
                code="proposal_not_found",
                message="World model change proposal not found.",
            )
        return WorldModelChangeApplyResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=accepted.proposal,
            apply_run_id=accepted.apply_run_id,
            ok=accepted.ok,
            summary=accepted.summary,
            results=accepted.results,
            reindex_result=accepted.reindex_result,
        )

    def reject_world_model_change_impl(
        *,
        trace: RequestTrace,
        proposal_id: int,
        actor: Optional[str],
        reason: Optional[str],
    ) -> WorldModelChangeResponse:
        rejected = workflow_service.reject_change(
            proposal_id=proposal_id,
            actor=actor,
            reason=reason,
        )
        if not rejected:
            raise _api_error(
                404,
                request_trace=trace,
                code="proposal_not_found",
                message="World model change proposal not found.",
            )
        return WorldModelChangeResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=rejected,
        )

    @router.post("/world-model/changes/propose", response_model=WorldModelChangeResponse)
    def v1_propose_world_model_change(request: WorldModelChangeProposalRequest):
        trace = _new_trace()
        try:
            proposal = workflow_service.propose_change(
                instruction=request.instruction,
                mode=request.mode,
                dry_run=request.dry_run,
                supersedes_proposal_id=request.supersedes_proposal_id,
            )
        except RuntimeError as exc:
            raise _api_error(
                500,
                request_trace=trace,
                code="proposal_not_persisted",
                message=str(exc),
            )
        return WorldModelChangeResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=proposal,
        )

    @router.get("/world-model/changes", response_model=WorldModelChangeListResponse)
    def v1_list_world_model_changes(
        limit: int = 20,
        status: Optional[ProposalStatus] = None,
    ):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        return WorldModelChangeListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=workflow_service.list_changes(limit=safe_limit, status=status),
        )

    @router.get("/world-model/changes/{proposal_id}", response_model=WorldModelChangeResponse)
    def v1_get_world_model_change(proposal_id: int):
        trace = _new_trace()
        proposal = workflow_service.get_change(proposal_id)
        if not proposal:
            raise _api_error(
                404,
                request_trace=trace,
                code="proposal_not_found",
                message="World model change proposal not found.",
            )
        return WorldModelChangeResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            proposal=proposal,
        )

    @router.post("/world-model/changes/{proposal_id}/accept", response_model=WorldModelChangeApplyResponse)
    def v1_accept_world_model_change(proposal_id: int, request: WorldModelChangeDecisionRequest):
        trace = _new_trace()
        return accept_world_model_change_impl(
            trace=trace,
            proposal_id=proposal_id,
            actor=request.actor,
            reindex_after_apply=request.reindex_after_apply,
        )

    @router.post("/world-model/changes/{proposal_id}/reject", response_model=WorldModelChangeResponse)
    def v1_reject_world_model_change(proposal_id: int, request: WorldModelChangeDecisionRequest):
        trace = _new_trace()
        return reject_world_model_change_impl(
            trace=trace,
            proposal_id=proposal_id,
            actor=request.actor,
            reason=request.reason,
        )

    return router
