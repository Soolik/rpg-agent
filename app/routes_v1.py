from __future__ import annotations

import uuid
from typing import Callable, Optional, Type

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .api_models import (
    AssistantActionRequest,
    AssistantActionResponse,
    AssistantActionType,
    AssistantMode,
    CampaignResetRequest,
    CampaignResetResponse,
    CanonicalImportFileView,
    CanonicalImportRequest,
    CanonicalImportResponse,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationMessageCreateRequest,
    ConversationMessageListResponse,
    ConversationResponse,
    EntityRelationListResponse,
    GoogleDriveOAuthStartResponse,
    GoogleDriveOAuthStatusResponse,
    ProposalStatus,
    ProposalType,
    RequestTrace,
    V1ArtifactGenerateRequest,
    V1ChatRequest,
    V1ChatResponse,
    V1HealthResponse,
    V1SessionPrepRequest,
    WebSessionStatusResponse,
    WorldModelChangeApplyResponse,
    WorldModelChangeDecisionRequest,
    WorldModelChangeListResponse,
    WorldModelChangeProposalRequest,
    WorldModelChangeResponse,
    WorldModelChangeView,
    WorldModelEntityListResponse,
    WorldFactListResponse,
    WorldModelSearchItem,
    WorldModelSearchResponse,
    WorldModelSessionListResponse,
    WorldModelThreadListResponse,
)
from .applier import ProposalApplier
from .canonical_import_service import CanonicalImportService
from .chat_service import ChatService, StreamPlan
from .chat_models import ChatRequest, ChatResponse
from .consistency_service import ConsistencyService
from .conversation_store import ConversationStore, NullConversationStore
from .google_drive_oauth_service import GoogleDriveOAuthError, GoogleDriveOAuthService
from .request_auth import RequestAuthError, SignedSessionAuth
from .routed_drive_store import DriveWriteAccessError
from .task_router import TaskRouter
from .world_fact_service import WorldFactService
from .world_fact_store import NullWorldFactStore, WorldFactStore
from .world_model_service import WorldModelService
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

def build_v1_router(
    *,
    chat_request_cls: Type[ChatRequest],
    chat_fn: Callable[[ChatRequest], ChatResponse],
    chat_stream_fn: Optional[Callable[[ChatRequest], StreamPlan]] = None,
    health_fn: Callable[[], dict],
    drive_store,
    planner,
    consistency_planner=None,
    planner_context_builder: Optional[Callable[[str], str]] = None,
    workflow_store: Optional[WorkflowStore | NullWorkflowStore] = None,
    world_model_store: Optional[WorldModelStore | NullWorldModelStore] = None,
    world_fact_store: Optional[WorldFactStore | NullWorldFactStore] = None,
    conversation_store: Optional[ConversationStore | NullConversationStore] = None,
    applier: Optional[ProposalApplier] = None,
    reindex_fn: Optional[Callable[[list], dict]] = None,
    campaign_reset_fn: Optional[Callable[..., dict]] = None,
    google_drive_oauth_service: Optional[GoogleDriveOAuthService] = None,
    session_auth: Optional[SignedSessionAuth] = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])
    store = workflow_store or NullWorkflowStore()
    model_store = world_model_store or NullWorldModelStore()
    fact_store = world_fact_store or NullWorldFactStore()
    convo_store = conversation_store or NullConversationStore()
    guard_planner = consistency_planner or planner
    consistency_service = ConsistencyService(world_model_store=model_store, world_fact_store=fact_store)
    proposal_applier = applier or ProposalApplier(
        drive_store=drive_store,
        consistency_service=consistency_service,
    )
    chat_service = ChatService(
        chat_request_cls=chat_request_cls,
        chat_fn=chat_fn,
        chat_stream_fn=chat_stream_fn,
        drive_store=drive_store,
        planner=planner,
        consistency_planner=guard_planner,
        world_model_store=model_store,
        conversation_store=convo_store,
        task_router=TaskRouter(),
        consistency_service=consistency_service,
    )
    workflow_service = WorkflowService(
        drive_store=drive_store,
        planner=planner,
        workflow_store=store,
        applier=proposal_applier,
        context_builder=planner_context_builder,
    )
    world_model_service = WorldModelService(world_model_store=model_store)
    world_fact_service = WorldFactService(world_model_store=model_store, world_fact_store=fact_store)
    canonical_import_service = CanonicalImportService(drive_store=drive_store, reindex_fn=reindex_fn)
    oauth_service = google_drive_oauth_service

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

    @router.post("/admin/reset-campaign-data", response_model=CampaignResetResponse)
    def v1_reset_campaign_data(request: CampaignResetRequest):
        trace = _new_trace()
        if not campaign_reset_fn:
            raise _api_error(
                503,
                request_trace=trace,
                code="campaign_reset_unavailable",
                message="Campaign reset is not configured for this deployment.",
            )
        result = campaign_reset_fn(
            clear_index=request.clear_index,
            clear_world_model=request.clear_world_model,
            clear_workflows=request.clear_workflows,
            clear_conversations=request.clear_conversations,
            clear_snapshots=request.clear_snapshots,
        )
        return CampaignResetResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            campaign_id=result.get("campaign_id") or "",
            deleted=result.get("deleted") or {},
        )

    @router.get("/auth/google-drive/status", response_model=GoogleDriveOAuthStatusResponse)
    def v1_google_drive_auth_status():
        trace = _new_trace()
        status = oauth_service.get_status() if oauth_service else None
        return GoogleDriveOAuthStatusResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            configured=bool(status.configured) if status else False,
            connected=bool(status.connected) if status else False,
            subject_email=status.subject_email if status else None,
            scopes=status.scopes if status else [],
            redirect_uri=status.redirect_uri if status else None,
            write_mode=status.write_mode if status else "service_account",
        )

    @router.get("/auth/session/status", response_model=WebSessionStatusResponse)
    def v1_session_status(request: Request):
        trace = _new_trace()
        identity = None
        if session_auth:
            try:
                identity = session_auth.verify_cookie(request.cookies.get(session_auth.cookie_name))
            except RequestAuthError:
                identity = None
        return WebSessionStatusResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            authenticated=identity is not None,
            email=identity.email if identity else None,
        )

    @router.post("/auth/google-drive/start", response_model=GoogleDriveOAuthStartResponse)
    def v1_google_drive_auth_start():
        trace = _new_trace()
        if not oauth_service:
            raise _api_error(
                503,
                request_trace=trace,
                code="google_drive_oauth_unavailable",
                message="Google Drive OAuth is not configured for this deployment.",
            )
        try:
            started = oauth_service.start_authorization()
        except GoogleDriveOAuthError as exc:
            raise _api_error(
                503,
                request_trace=trace,
                code="google_drive_oauth_unavailable",
                message=str(exc),
            )
        return GoogleDriveOAuthStartResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            authorization_url=started.authorization_url,
            redirect_uri=started.redirect_uri,
            scopes=started.scopes,
        )

    @router.get("/auth/google-drive/start", include_in_schema=False)
    def v1_google_drive_auth_start_redirect():
        trace = _new_trace()
        if not oauth_service:
            raise _api_error(
                503,
                request_trace=trace,
                code="google_drive_oauth_unavailable",
                message="Google Drive OAuth is not configured for this deployment.",
            )
        try:
            started = oauth_service.start_authorization()
        except GoogleDriveOAuthError as exc:
            raise _api_error(
                503,
                request_trace=trace,
                code="google_drive_oauth_unavailable",
                message=str(exc),
            )
        return RedirectResponse(url=started.authorization_url, status_code=307)

    @router.get("/auth/google-drive/callback", response_class=HTMLResponse)
    def v1_google_drive_auth_callback(code: str, state: str):
        trace = _new_trace()
        if not oauth_service:
            raise _api_error(
                503,
                request_trace=trace,
                code="google_drive_oauth_unavailable",
                message="Google Drive OAuth is not configured for this deployment.",
            )
        try:
            result = oauth_service.handle_callback(code=code, state=state)
        except GoogleDriveOAuthError as exc:
            raise _api_error(
                400,
                request_trace=trace,
                code="google_drive_oauth_callback_failed",
                message=str(exc),
            )
        response = HTMLResponse(content=result.html_body)
        if session_auth and result.subject_email:
            response.set_cookie(
                session_auth.cookie_name,
                session_auth.issue(email=result.subject_email, subject=result.subject_id),
                httponly=True,
                secure=True,
                samesite="lax",
                max_age=session_auth.ttl_seconds,
                path="/",
            )
        return response

    @router.post("/auth/session/logout", response_model=WebSessionStatusResponse)
    def v1_session_logout():
        trace = _new_trace()
        response = WebSessionStatusResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            authenticated=False,
            email=None,
        )
        transport = JSONResponse(response.model_dump(mode="json"))
        if session_auth:
            transport.delete_cookie(session_auth.cookie_name, path="/")
        return transport

    @router.post("/auth/google-drive/disconnect", response_model=GoogleDriveOAuthStatusResponse)
    def v1_google_drive_auth_disconnect():
        trace = _new_trace()
        if not oauth_service:
            raise _api_error(
                503,
                request_trace=trace,
                code="google_drive_oauth_unavailable",
                message="Google Drive OAuth is not configured for this deployment.",
            )
        status = oauth_service.disconnect()
        return GoogleDriveOAuthStatusResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            configured=status.configured,
            connected=status.connected,
            subject_email=status.subject_email,
            scopes=status.scopes,
            redirect_uri=status.redirect_uri,
            write_mode=status.write_mode,
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
                validation=accepted.validation,
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

        if request.action_type == AssistantActionType.confirm_inferred_action:
            if not request.message:
                raise _api_error(
                    400,
                    request_trace=trace,
                    code="missing_message",
                    message="message is required for confirm_inferred_action.",
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
                confirmed=True,
            )
            return AssistantActionResponse(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                action_type=request.action_type,
                ok=True,
                summary="Confirmed action executed.",
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
            items=world_model_service.list_entities(limit=safe_limit, kind=kind),
        )

    @router.get("/world-model/threads", response_model=WorldModelThreadListResponse)
    def v1_list_threads(limit: int = 20, status: Optional[str] = None):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        return WorldModelThreadListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=world_model_service.list_threads(limit=safe_limit, status=status),
        )

    @router.get("/world-model/sessions", response_model=WorldModelSessionListResponse)
    def v1_list_sessions(limit: int = 20):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 100))
        return WorldModelSessionListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=world_model_service.list_sessions(limit=safe_limit),
        )

    @router.get("/world-model/search", response_model=WorldModelSearchResponse)
    def v1_search_world_model(q: str, limit: int = 20):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 50))
        items = world_model_service.search(q, limit=safe_limit)
        return WorldModelSearchResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            query=q,
            items=items,
        )

    @router.get("/world-model/facts", response_model=WorldFactListResponse)
    def v1_list_world_facts(
        limit: int = 20,
        subject_name: Optional[str] = None,
        predicate: Optional[str] = None,
    ):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 200))
        return WorldFactListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=world_fact_service.list_facts(
                limit=safe_limit,
                subject_name=subject_name,
                predicate=predicate,
            ),
        )

    @router.get("/world-model/relations", response_model=EntityRelationListResponse)
    def v1_list_world_relations(
        limit: int = 20,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
        relation_type: Optional[str] = None,
    ):
        trace = _new_trace()
        safe_limit = max(1, min(limit, 200))
        return EntityRelationListResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            items=world_fact_service.list_relations(
                limit=safe_limit,
                source_name=source_name,
                target_name=target_name,
                relation_type=relation_type,
            ),
        )

    @router.post("/imports/canonical-files", response_model=CanonicalImportResponse)
    def v1_import_canonical_files(request: CanonicalImportRequest):
        trace = _new_trace()
        if bool(request.source_path) == bool(request.source_drive_folder_id):
            raise _api_error(
                400,
                request_trace=trace,
                code="invalid_import_source",
                message="Provide exactly one of source_path or source_drive_folder_id.",
            )
        try:
            if request.source_drive_folder_id:
                imported = canonical_import_service.import_drive_folder(
                    folder_id=request.source_drive_folder_id,
                    dry_run=request.dry_run,
                    replace_existing=request.replace_existing,
                    reindex_after_import=request.reindex_after_import,
                )
            else:
                imported = canonical_import_service.import_folder(
                    source_path=request.source_path or "",
                    dry_run=request.dry_run,
                    replace_existing=request.replace_existing,
                    reindex_after_import=request.reindex_after_import,
                )
        except FileNotFoundError as exc:
            raise _api_error(
                404,
                request_trace=trace,
                code="source_path_not_found",
                message=str(exc),
            )
        except ValueError as exc:
            raise _api_error(
                400,
                request_trace=trace,
                code="invalid_source_path",
                message=str(exc),
            )
        except PermissionError as exc:
            raise _api_error(
                403,
                request_trace=trace,
                code="drive_access_denied",
                message=str(exc),
            )
        except DriveWriteAccessError as exc:
            raise _api_error(
                409,
                request_trace=trace,
                code="google_drive_user_oauth_required",
                message=str(exc),
            )

        return CanonicalImportResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            source_path=imported.source_path,
            dry_run=imported.dry_run,
            imported_count=imported.imported_count,
            created_count=imported.created_count,
            updated_count=imported.updated_count,
            skipped_count=imported.skipped_count,
            error_count=imported.error_count,
            warnings=imported.warnings or [],
            reindex_result=imported.reindex_result,
            results=[
                CanonicalImportFileView(
                    source_path=item.source_path,
                    source_name=item.source_name,
                    format=item.format,
                    folder=item.folder,
                    title=item.title,
                    entity_type=item.entity_type,
                    action=item.action,
                    status=item.status,  # type: ignore[arg-type]
                    chars=item.chars,
                    doc_id=item.doc_id,
                    path=item.path,
                    message=item.message,
                )
                for item in imported.results
            ],
        )

    def accept_world_model_change_impl(
        *,
        trace: RequestTrace,
        proposal_id: int,
        actor: Optional[str],
        reindex_after_apply: bool,
    ) -> WorldModelChangeApplyResponse:
        try:
            accepted = workflow_service.accept_change(
                proposal_id=proposal_id,
                actor=actor,
                reindex_after_apply=reindex_after_apply,
            )
        except DriveWriteAccessError as exc:
            raise _api_error(
                409,
                request_trace=trace,
                code="google_drive_user_oauth_required",
                message=str(exc),
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
            validation=accepted.validation,
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
