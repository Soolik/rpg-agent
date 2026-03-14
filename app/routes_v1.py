from __future__ import annotations

import json
import uuid
from typing import Callable, Optional, Type

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .api_models import (
    ChatArtifact,
    ContinuityReport,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationMessageCreateRequest,
    ConversationMessageListResponse,
    ConversationResponse,
    NextAction,
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
from .conversation_store import ConversationMessageRecord, ConversationRecord, ConversationStore, NullConversationStore
from .models_v2 import ApplyChangesRequest, ChangeProposal, ProposeChangesRequest
from .routes_v2 import build_context_for_planner
from .world_model_store import NullWorldModelStore, WorldModelStore
from .workflow_store import NullWorkflowStore, WorkflowStore


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 4000
STREAM_CHUNK_CHARS = 500


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


def _artifact_from_chat(response: ChatResponse) -> Optional[ChatArtifact]:
    if not (response.artifact_type and response.artifact_text):
        return None
    return ChatArtifact(
        artifact_type=response.artifact_type,
        text=response.artifact_text,
        format="markdown",
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


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _compact_title(value: str, *, fallback: str) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        return fallback
    text = text.lstrip("#*- ").strip()
    if ":" in text and len(text.split(":", 1)[0]) <= 24:
        text = text.split(":", 1)[1].strip() or text
    return text[:96].rstrip(" .:-") or fallback


def _derive_conversation_title(message: str, explicit_title: Optional[str]) -> str:
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()[:96]
    return _compact_title(_first_nonempty_line(message), fallback="Nowa rozmowa")


def _derive_response_title(response: ChatResponse) -> str:
    if response.artifact_text:
        return _compact_title(_first_nonempty_line(response.artifact_text), fallback="Odpowiedz")
    if response.artifact_type:
        return response.artifact_type.replace("_", " ").title()
    return _compact_title(_first_nonempty_line(response.reply), fallback="Odpowiedz")


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


def _next_actions_for_response(
    *,
    response: ChatResponse,
    continuity: Optional[ContinuityReport],
    conversation_id: Optional[str],
) -> list[NextAction]:
    actions: list[NextAction] = [
        NextAction(
            type="continue_conversation",
            label="Kontynuuj rozmowe",
            payload={"conversation_id": conversation_id} if conversation_id else {},
        )
    ]

    if response.kind == "proposal" and response.proposal_id is not None:
        actions.append(
            NextAction(
                type="accept_world_change",
                label="Zaakceptuj zmiane",
                payload={"proposal_id": response.proposal_id},
            )
        )
        actions.append(
            NextAction(
                type="reject_world_change",
                label="Odrzuc zmiane",
                payload={"proposal_id": response.proposal_id},
            )
        )

    if response.artifact_type:
        actions.append(
            NextAction(
                type="revise",
                label="Przerob odpowiedz",
                payload={"artifact_type": response.artifact_type},
            )
        )

    if continuity and not continuity.ok:
        actions.append(
            NextAction(
                type="review_continuity",
                label="Sprawdz ciaglosc",
                payload={"issue_count": len(continuity.issues)},
            )
        )

    if response.output_doc_id and response.output_path:
        actions.append(
            NextAction(
                type="open_output_doc",
                label="Otworz zapisany output",
                payload={"doc_id": response.output_doc_id, "path": response.output_path},
            )
        )

    deduped: list[NextAction] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        key = (action.type, json.dumps(action.payload, ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _chat_response_v1(
    *,
    trace: RequestTrace,
    message: str,
    response: ChatResponse,
    world_model_store: WorldModelStore | NullWorldModelStore,
    conversation_id: Optional[str] = None,
    conversation_title: Optional[str] = None,
) -> V1ChatResponse:
    continuity = _continuity_for_response(
        message=message,
        response=response,
        world_model_store=world_model_store,
    )
    reply_markdown = response.artifact_text or response.reply
    return V1ChatResponse(
        request_id=trace.request_id,
        trace_id=trace.trace_id,
        kind=response.kind,
        reply=response.reply,
        reply_markdown=reply_markdown,
        title=_derive_response_title(response),
        conversation_id=conversation_id,
        conversation_title=conversation_title,
        artifact_type=response.artifact_type,
        artifact_text=response.artifact_text,
        artifact=_artifact_from_chat(response),
        proposal_id=response.proposal_id,
        session_id=response.session_id,
        citations=response.references,
        warnings=response.warnings,
        next_actions=_next_actions_for_response(
            response=response,
            continuity=continuity,
            conversation_id=conversation_id,
        ),
        output=_saved_output_from_chat(response),
        telemetry=response.telemetry,
        continuity=continuity,
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


def _conversation_storage_enabled(store: ConversationStore | NullConversationStore) -> bool:
    return not isinstance(store, NullConversationStore)


def _resolve_conversation(
    *,
    trace: RequestTrace,
    conversation_store: ConversationStore | NullConversationStore,
    conversation_id: Optional[str],
    conversation_title: Optional[str],
    seed_message: str,
) -> Optional[ConversationRecord]:
    if not conversation_id:
        if not _conversation_storage_enabled(conversation_store):
            return None
        return conversation_store.create_conversation(
            title=_derive_conversation_title(seed_message, conversation_title),
            metadata={"source": "v1_chat"},
        )

    if not _conversation_storage_enabled(conversation_store):
        raise _api_error(
            503,
            request_trace=trace,
            code="conversation_store_unavailable",
            message="Conversation storage is not configured for this deployment.",
        )

    conversation = conversation_store.get_conversation(conversation_id)
    if not conversation:
        raise _api_error(
            404,
            request_trace=trace,
            code="conversation_not_found",
            message="Conversation not found.",
        )
    return conversation


def _prompt_history(messages: list[ConversationMessageRecord]) -> list[ConversationMessageRecord]:
    if not messages:
        return []

    selected: list[ConversationMessageRecord] = []
    total_chars = 0
    for message in reversed(messages):
        rendered = f"{message.role}: {message.content}"
        if selected and total_chars + len(rendered) > MAX_HISTORY_CHARS:
            break
        selected.append(message)
        total_chars += len(rendered)
        if len(selected) >= MAX_HISTORY_MESSAGES:
            break
    selected.reverse()
    return selected


def _compose_message_with_history(message: str, history: list[ConversationMessageRecord]) -> str:
    if not history or len(message) > MAX_HISTORY_CHARS:
        return message

    lines = ["KONTEKST ROZMOWY:"]
    for item in history:
        role_label = "Uzytkownik" if item.role == "user" else "Asystent" if item.role == "assistant" else item.role.title()
        lines.append(f"{role_label}: {item.content}")
    lines.extend(
        [
            "",
            "NOWA WIADOMOSC UZYTKOWNIKA:",
            message,
            "",
            "Odpowiedz na ostatnia wiadomosc, zachowujac ciaglosc rozmowy i kanonu.",
        ]
    )
    return "\n".join(lines)


def _stream_chunks(text: str) -> list[str]:
    compact = text or ""
    if not compact:
        return []
    chunks: list[str] = []
    cursor = 0
    while cursor < len(compact):
        chunks.append(compact[cursor : cursor + STREAM_CHUNK_CHARS])
        cursor += STREAM_CHUNK_CHARS
    return chunks


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_chat_response(response: V1ChatResponse) -> StreamingResponse:
    payload = response.model_dump(mode="json")

    def iterator():
        yield _sse_event(
            "start",
            {
                "request_id": response.request_id,
                "trace_id": response.trace_id,
                "conversation_id": response.conversation_id,
                "title": response.title,
            },
        )
        for chunk in _stream_chunks(response.reply_markdown):
            yield _sse_event("delta", {"text": chunk})
        yield _sse_event("complete", payload)

    return StreamingResponse(iterator(), media_type="text/event-stream")


def _run_chat_request(
    *,
    trace: RequestTrace,
    chat_request_cls: Type[ChatRequest],
    chat_fn: Callable[[ChatRequest], ChatResponse],
    world_model_store: WorldModelStore | NullWorldModelStore,
    conversation_store: ConversationStore | NullConversationStore,
    message: str,
    intent: str,
    artifact_type: Optional[str],
    source_title: Optional[str],
    include_sources: bool,
    include_telemetry: bool,
    save_output: bool,
    output_title: Optional[str],
    conversation_id: Optional[str],
    conversation_title: Optional[str],
) -> V1ChatResponse:
    conversation = _resolve_conversation(
        trace=trace,
        conversation_store=conversation_store,
        conversation_id=conversation_id,
        conversation_title=conversation_title,
        seed_message=message,
    )
    history: list[ConversationMessageRecord] = []
    if conversation:
        history = _prompt_history(conversation_store.list_messages(conversation.conversation_id, limit=50))
        conversation_store.append_message(
            conversation.conversation_id,
            role="user",
            content=message,
            kind="input",
            artifact_type=artifact_type,
            metadata={"source_title": source_title},
        )

    response = chat_fn(
        chat_request_cls(
            message=_compose_message_with_history(message, history),
            intent=intent,
            artifact_type=artifact_type,
            source_title=source_title,
            conversation_id=conversation.conversation_id if conversation else None,
            conversation_title=conversation.title if conversation else conversation_title,
            include_sources=include_sources,
            include_telemetry=include_telemetry,
            save_output=save_output,
            output_title=output_title,
        )
    )
    rendered = _chat_response_v1(
        trace=trace,
        message=message,
        response=response,
        world_model_store=world_model_store,
        conversation_id=conversation.conversation_id if conversation else None,
        conversation_title=conversation.title if conversation else None,
    )

    if conversation:
        conversation_store.append_message(
            conversation.conversation_id,
            role="assistant",
            content=rendered.reply_markdown,
            kind=rendered.kind,
            artifact_type=rendered.artifact_type,
            metadata={
                "proposal_id": rendered.proposal_id,
                "session_id": rendered.session_id,
                "continuity_ok": rendered.continuity.ok if rendered.continuity else None,
            },
        )

    return rendered


def build_v1_router(
    *,
    chat_request_cls: Type[ChatRequest],
    chat_fn: Callable[[ChatRequest], ChatResponse],
    health_fn: Callable[[], dict],
    drive_store,
    planner,
    workflow_store: Optional[WorkflowStore | NullWorkflowStore] = None,
    world_model_store: Optional[WorldModelStore | NullWorldModelStore] = None,
    conversation_store: Optional[ConversationStore | NullConversationStore] = None,
    applier: Optional[ProposalApplier] = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])
    store = workflow_store or NullWorkflowStore()
    model_store = world_model_store or NullWorldModelStore()
    convo_store = conversation_store or NullConversationStore()
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

    @router.get("/conversations", response_model=ConversationListResponse)
    def v1_list_conversations(limit: int = 20):
        trace = _new_trace()
        if not _conversation_storage_enabled(convo_store):
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
            items=convo_store.list_conversations(limit=safe_limit),
        )

    @router.post("/conversations", response_model=ConversationResponse)
    def v1_create_conversation(request: ConversationCreateRequest):
        trace = _new_trace()
        if not _conversation_storage_enabled(convo_store):
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        conversation = convo_store.create_conversation(
            title=_derive_conversation_title("", request.title),
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
        if not _conversation_storage_enabled(convo_store):
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        conversation = convo_store.get_conversation(conversation_id)
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
        if not _conversation_storage_enabled(convo_store):
            raise _api_error(
                503,
                request_trace=trace,
                code="conversation_store_unavailable",
                message="Conversation storage is not configured for this deployment.",
            )
        conversation = convo_store.get_conversation(conversation_id)
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
            items=convo_store.list_messages(conversation_id, limit=safe_limit),
        )

    @router.post("/chat", response_model=V1ChatResponse)
    def v1_chat(request: V1ChatRequest):
        trace = _new_trace()
        response = _run_chat_request(
            trace=trace,
            chat_request_cls=chat_request_cls,
            chat_fn=chat_fn,
            world_model_store=model_store,
            conversation_store=convo_store,
            message=request.message,
            intent=request.intent,
            artifact_type=request.artifact_type,
            source_title=request.source_title,
            include_sources=request.include_sources,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=request.conversation_id,
            conversation_title=request.conversation_title,
        )
        if request.stream:
            return _stream_chat_response(response)
        return response

    @router.post("/conversations/{conversation_id}/messages", response_model=V1ChatResponse)
    def v1_conversation_message(conversation_id: str, request: ConversationMessageCreateRequest):
        trace = _new_trace()
        response = _run_chat_request(
            trace=trace,
            chat_request_cls=chat_request_cls,
            chat_fn=chat_fn,
            world_model_store=model_store,
            conversation_store=convo_store,
            message=request.message,
            intent=request.intent,
            artifact_type=request.artifact_type,
            source_title=request.source_title,
            include_sources=request.include_sources,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=conversation_id,
            conversation_title=None,
        )
        if request.stream:
            return _stream_chat_response(response)
        return response

    @router.post("/artifacts/generate", response_model=V1ChatResponse)
    def v1_generate_artifact(request: V1ArtifactGenerateRequest):
        trace = _new_trace()
        response = _run_chat_request(
            trace=trace,
            chat_request_cls=chat_request_cls,
            chat_fn=chat_fn,
            world_model_store=model_store,
            conversation_store=convo_store,
            message=request.message,
            intent="auto",
            artifact_type=request.artifact_type,
            source_title=None,
            include_sources=request.include_sources,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=request.conversation_id,
            conversation_title=request.conversation_title,
        )
        if request.stream:
            return _stream_chat_response(response)
        return response

    @router.post("/sessions/prep", response_model=V1ChatResponse)
    def v1_prepare_session(request: V1SessionPrepRequest):
        trace = _new_trace()
        response = _run_chat_request(
            trace=trace,
            chat_request_cls=chat_request_cls,
            chat_fn=chat_fn,
            world_model_store=model_store,
            conversation_store=convo_store,
            message=request.message,
            intent="auto",
            artifact_type="pre_session_brief",
            source_title=None,
            include_sources=False,
            include_telemetry=request.include_telemetry,
            save_output=request.save_output,
            output_title=request.output_title,
            conversation_id=request.conversation_id,
            conversation_title=request.conversation_title,
        )
        if request.stream:
            return _stream_chat_response(response)
        return response

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
