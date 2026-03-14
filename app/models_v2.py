from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class WorldEntityType(str, Enum):
    npc = "npc"
    location = "location"
    faction = "faction"
    thread = "thread"
    secret = "secret"
    session = "session"
    output = "output"
    bible = "bible"
    glossary = "glossary"
    rules = "rules"
    other = "other"


class ActionType(str, Enum):
    create_doc = "create_doc"
    append_doc = "append_doc"
    replace_doc = "replace_doc"
    replace_section = "replace_section"
    create_if_missing = "create_if_missing"
    update_tracker_row = "update_tracker_row"
    reindex = "reindex"


class DocumentRef(BaseModel):
    folder: str = Field(..., description="Logical world folder, e.g. '03 NPC'")
    title: str = Field(..., description="Human-readable document title")
    path_hint: Optional[str] = Field(default=None, description="Optional path hint or file path")
    doc_id: Optional[str] = Field(default=None, description="Google Doc / Drive file id if known")


class DocumentAction(BaseModel):
    action_type: ActionType
    entity_type: WorldEntityType = WorldEntityType.other
    target: Optional[DocumentRef] = None
    content: Optional[str] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class ChangeProposal(BaseModel):
    proposal_id: Optional[int] = None
    summary: str
    user_goal: str
    assumptions: List[str] = Field(default_factory=list)
    impacted_docs: List[DocumentRef] = Field(default_factory=list)
    actions: List[DocumentAction] = Field(default_factory=list)
    needs_confirmation: bool = True


class ProposeChangesRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    mode: Literal["auto", "create", "update", "session", "consistency", "player_output"] = "auto"
    campaign_id: Optional[str] = None
    dry_run: bool = True


class ApplyChangesRequest(BaseModel):
    proposal_id: Optional[int] = None
    proposal: ChangeProposal
    approved: bool = True
    approved_by: Optional[str] = None
    reindex_after_apply: bool = True


class AppliedActionResult(BaseModel):
    action_type: ActionType
    success: bool
    message: str
    target: Optional[DocumentRef] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ApplyChangesResponse(BaseModel):
    proposal_id: Optional[int] = None
    apply_run_id: Optional[int] = None
    ok: bool
    summary: str
    results: List[AppliedActionResult] = Field(default_factory=list)
    reindex_result: Optional[Dict[str, Any]] = None


class WorldDocInfo(BaseModel):
    folder: str
    title: str
    doc_id: Optional[str] = None
    path_hint: Optional[str] = None
    entity_type: WorldEntityType = WorldEntityType.other


class WorldStatusResponse(BaseModel):
    campaign_id: Optional[str] = None
    folders: Dict[str, int] = Field(default_factory=dict)
    docs: List[WorldDocInfo] = Field(default_factory=list)
    indexed_chunks: Optional[int] = None
    notes: List[str] = Field(default_factory=list)


class ProposalRecord(BaseModel):
    id: int
    campaign_id: str
    summary: str
    user_goal: str
    approved: bool
    approved_by: Optional[str] = None
    created_at: str
    updated_at: str


class ProposalDetail(ProposalRecord):
    request: Dict[str, Any] = Field(default_factory=dict)
    proposal: Dict[str, Any] = Field(default_factory=dict)


class ApplyRunRecord(BaseModel):
    id: int
    campaign_id: str
    proposal_id: Optional[int] = None
    approved: bool
    approved_by: Optional[str] = None
    ok: bool
    created_at: str


class ApplyRunDetail(ApplyRunRecord):
    request: Dict[str, Any] = Field(default_factory=dict)
    response: Dict[str, Any] = Field(default_factory=dict)


class ReadWorldDocRequest(BaseModel):
    doc_id: Optional[str] = None
    folder: Optional[str] = None
    title: Optional[str] = None


class ReadWorldDocResponse(BaseModel):
    doc: WorldDocInfo
    content: str


class ConsistencyCheckRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    candidate_text: Optional[str] = None


class ConsistencyIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    title: str
    description: str
    related_docs: List[DocumentRef] = Field(default_factory=list)


class ConsistencyCheckResponse(BaseModel):
    summary: str
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class SessionThreadPatchPayload(BaseModel):
    thread_id: Optional[str] = None
    title: str
    status: Optional[str] = None
    change: str


class SessionEntityPatchPayload(BaseModel):
    kind: Literal["npc", "location", "faction", "item", "other"] = "other"
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)


class SessionPatchPayload(BaseModel):
    session_summary: str
    thread_tracker_patch: List[SessionThreadPatchPayload] = Field(default_factory=list)
    entities_patch: List[SessionEntityPatchPayload] = Field(default_factory=list)
    rag_additions: List[str] = Field(default_factory=list)


class SyncSessionPatchRequest(BaseModel):
    patch: SessionPatchPayload
    raw_notes: Optional[str] = None
    campaign_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    source_title: Optional[str] = None


class SyncSessionPatchResponse(BaseModel):
    session_id: int
    campaign_id: str
    summary: str
    entity_count: int
    thread_count: int


class WorldEntityRecord(BaseModel):
    id: int
    campaign_id: str
    entity_kind: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    last_session_id: Optional[int] = None
    updated_at: str


class WorldThreadRecord(BaseModel):
    id: int
    campaign_id: str
    thread_key: str
    thread_id: Optional[str] = None
    title: str
    status: Optional[str] = None
    last_change: str
    last_session_id: Optional[int] = None
    updated_at: str


class WorldSessionRecord(BaseModel):
    id: int
    campaign_id: str
    session_summary: str
    entity_count: int
    thread_count: int
    source_title: Optional[str] = None
    created_at: str


class WorldModelStatusResponse(BaseModel):
    campaign_id: str
    entity_count: int = 0
    thread_count: int = 0
    session_count: int = 0
