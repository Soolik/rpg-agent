from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .chat_models import ArtifactType, ChatIntent
from .conversation_store import ConversationMessageRecord, ConversationRecord
from .models_v2 import AppliedActionResult, DocumentAction, DocumentRef, WorldEntityRecord, WorldSessionRecord, WorldThreadRecord


class ProposalStatus(str, Enum):
    proposed = "proposed"
    accepted = "accepted"
    rejected = "rejected"
    superseded = "superseded"


class ProposalType(str, Enum):
    general = "general"
    world_model_change = "world_model_change"


class RequestTrace(BaseModel):
    request_id: str
    trace_id: str


class SavedOutputRef(BaseModel):
    doc_id: str
    title: str
    path: str


class ChatArtifact(BaseModel):
    artifact_type: ArtifactType
    text: str
    format: Literal["markdown", "plain_text"] = "markdown"


class NextAction(BaseModel):
    type: Literal[
        "continue_conversation",
        "revise",
        "accept_world_change",
        "reject_world_change",
        "review_continuity",
        "open_output_doc",
    ]
    label: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class ContinuityIssue(BaseModel):
    code: Literal["new_proper_noun", "possible_name_conflict", "world_model_mismatch", "inferred_claim"] = "new_proper_noun"
    severity: Literal["info", "warning", "error"] = "warning"
    message: str
    related_name: Optional[str] = None
    evidence: Optional[str] = None
    source: Optional[Literal["world_model", "request", "generated"]] = None


class ContinuityReport(BaseModel):
    ok: bool = True
    source_backed_names: List[str] = Field(default_factory=list)
    inferred_names: List[str] = Field(default_factory=list)
    proposed_new_names: List[str] = Field(default_factory=list)
    possible_conflicts: List[str] = Field(default_factory=list)
    issues: List[ContinuityIssue] = Field(default_factory=list)


class V1HealthResponse(RequestTrace):
    ok: bool
    campaign_id: str
    revision: str


class V1ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    intent: ChatIntent = "auto"
    artifact_type: Optional[ArtifactType] = None
    source_title: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    stream: bool = False
    include_sources: bool = False
    include_telemetry: bool = False
    save_output: bool = False
    output_title: Optional[str] = None


class V1ArtifactGenerateRequest(BaseModel):
    message: str = Field(..., min_length=1)
    artifact_type: ArtifactType
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    stream: bool = False
    include_telemetry: bool = False
    save_output: bool = False
    output_title: Optional[str] = None
    include_sources: bool = False


class V1SessionPrepRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    stream: bool = False
    include_telemetry: bool = False
    save_output: bool = False
    output_title: Optional[str] = None


class V1ChatResponse(RequestTrace):
    kind: Literal["answer", "proposal", "session_sync", "creative"]
    reply: str
    reply_markdown: str
    title: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    artifact_type: Optional[ArtifactType] = None
    artifact_text: Optional[str] = None
    artifact: Optional[ChatArtifact] = None
    proposal_id: Optional[int] = None
    session_id: Optional[int] = None
    citations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    next_actions: List[NextAction] = Field(default_factory=list)
    output: Optional[SavedOutputRef] = None
    telemetry: Optional[Dict[str, Any]] = None
    continuity: Optional[ContinuityReport] = None


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None


class ConversationMessageCreateRequest(BaseModel):
    message: str = Field(..., min_length=1)
    intent: ChatIntent = "auto"
    artifact_type: Optional[ArtifactType] = None
    source_title: Optional[str] = None
    stream: bool = False
    include_sources: bool = False
    include_telemetry: bool = False
    save_output: bool = False
    output_title: Optional[str] = None


class ConversationResponse(RequestTrace):
    conversation: ConversationRecord


class ConversationListResponse(RequestTrace):
    items: List[ConversationRecord] = Field(default_factory=list)


class ConversationMessageListResponse(RequestTrace):
    conversation_id: str
    items: List[ConversationMessageRecord] = Field(default_factory=list)


class WorldModelEntityListResponse(RequestTrace):
    items: List[WorldEntityRecord] = Field(default_factory=list)


class WorldModelThreadListResponse(RequestTrace):
    items: List[WorldThreadRecord] = Field(default_factory=list)


class WorldModelSessionListResponse(RequestTrace):
    items: List[WorldSessionRecord] = Field(default_factory=list)


class WorldModelSearchItem(BaseModel):
    record_type: Literal["entity", "thread", "session"]
    record_id: int
    title: str
    snippet: str
    entity_kind: Optional[str] = None
    status: Optional[str] = None
    source_title: Optional[str] = None
    score: int = 0


class WorldModelSearchResponse(RequestTrace):
    query: str
    items: List[WorldModelSearchItem] = Field(default_factory=list)


class WorldModelChangeProposalRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    mode: Literal["auto", "create", "update", "session", "consistency", "player_output"] = "auto"
    dry_run: bool = True
    supersedes_proposal_id: Optional[int] = None


class WorldModelChangeDecisionRequest(BaseModel):
    actor: Optional[str] = None
    reason: Optional[str] = None
    reindex_after_apply: bool = True


class WorldModelChangeView(BaseModel):
    proposal_id: int
    proposal_type: ProposalType = ProposalType.general
    status: ProposalStatus = ProposalStatus.proposed
    summary: str
    user_goal: str
    assumptions: List[str] = Field(default_factory=list)
    impacted_docs: List[DocumentRef] = Field(default_factory=list)
    actions: List[DocumentAction] = Field(default_factory=list)
    needs_confirmation: bool = True
    approved: bool = False
    approved_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    supersedes_proposal_id: Optional[int] = None
    accepted_apply_run_id: Optional[int] = None
    rejected_reason: Optional[str] = None
    reviewed_by: Optional[str] = None
    request: Dict[str, Any] = Field(default_factory=dict)
    raw_proposal: Dict[str, Any] = Field(default_factory=dict)


class WorldModelChangeResponse(RequestTrace):
    proposal: WorldModelChangeView


class WorldModelChangeListResponse(RequestTrace):
    items: List[WorldModelChangeView] = Field(default_factory=list)


class WorldModelChangeApplyResponse(RequestTrace):
    proposal: WorldModelChangeView
    apply_run_id: Optional[int] = None
    ok: bool = False
    summary: str = ""
    results: List[AppliedActionResult] = Field(default_factory=list)
    reindex_result: Optional[Dict[str, Any]] = None
