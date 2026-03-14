from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .models_v2 import SyncSessionPatchResponse


AskMode = Literal["auto", "campaign", "general", "scene"]
ChatIntent = Literal["auto", "answer", "proposal", "session_sync", "creative"]
ArtifactType = Literal[
    "gm_brief",
    "session_report",
    "player_summary",
    "pre_session_brief",
    "session_hooks",
    "scene_seed",
    "npc_brief",
    "twist_pack",
]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    include_sources: bool = False
    mode: AskMode = "auto"


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    intent: ChatIntent = "auto"
    artifact_type: Optional[ArtifactType] = None
    source_title: Optional[str] = None
    include_sources: bool = False
    include_telemetry: bool = False
    save_output: bool = False
    output_title: Optional[str] = None


class ChatResponse(BaseModel):
    kind: Literal["answer", "proposal", "session_sync", "creative"]
    reply: str
    artifact_type: Optional[ArtifactType] = None
    artifact_text: Optional[str] = None
    proposal_id: Optional[int] = None
    session_id: Optional[int] = None
    references: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    output_doc_id: Optional[str] = None
    output_title: Optional[str] = None
    output_path: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None


class ReindexRequest(BaseModel):
    clean: bool = False


class IngestSessionRequest(BaseModel):
    raw_notes: str = Field(..., min_length=1)
    campaign_id: Optional[str] = None


class ThreadPatch(BaseModel):
    thread_id: Optional[str] = None
    title: str
    status: Optional[str] = None
    change: str


class EntityPatch(BaseModel):
    kind: Literal["npc", "location", "faction", "item", "other"] = "other"
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)


class SessionPatch(BaseModel):
    session_summary: str
    thread_tracker_patch: List[ThreadPatch] = Field(default_factory=list)
    entities_patch: List[EntityPatch] = Field(default_factory=list)
    rag_additions: List[str] = Field(default_factory=list)


class IngestAndSyncSessionRequest(IngestSessionRequest):
    source_doc_id: Optional[str] = None
    source_title: Optional[str] = None


class IngestAndSyncSessionResponse(BaseModel):
    patch: SessionPatch
    sync: SyncSessionPatchResponse


class CampaignOut(BaseModel):
    format: Literal["bullets", "table"] = "bullets"
    bullets: List[str] = Field(default_factory=list)
    table: Optional[Dict[str, Any]] = None
    used_context: List[int] = Field(default_factory=list)
