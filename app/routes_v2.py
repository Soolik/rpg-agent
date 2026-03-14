from __future__ import annotations

from typing import Callable, Dict, Optional

from fastapi import APIRouter, HTTPException

from .applier import ProposalApplier
from .drive_store import DriveStore
from .models_v2 import (
    ApplyChangesRequest,
    ApplyChangesResponse,
    ApplyRunDetail,
    ApplyRunRecord,
    ChangeProposal,
    ConsistencyCheckRequest,
    ConsistencyCheckResponse,
    DocumentRef,
    ProposalDetail,
    ProposalRecord,
    ProposeChangesRequest,
    ReadWorldDocRequest,
    ReadWorldDocResponse,
    SyncSessionPatchRequest,
    SyncSessionPatchResponse,
    WorldEntityRecord,
    WorldModelStatusResponse,
    WorldSessionRecord,
    WorldStatusResponse,
    WorldThreadRecord,
)
from .planner import PlannerService
from .world_model_store import NullWorldModelStore, WorldModelStore
from .workflow_store import NullWorkflowStore, WorkflowStore


CORE_DOC_TITLES = {"Campaign Bible", "Glossary", "Rules And Tone", "Thread Tracker"}


def build_context_for_planner(drive_store: DriveStore) -> str:
    docs = drive_store.list_world_docs()
    if not docs:
        return "No world docs available yet."

    chunks = []
    for doc in docs[:50]:
        line = f"## DOC: {doc.folder}/{doc.title}"
        if doc.title in CORE_DOC_TITLES and doc.doc_id:
            try:
                text = drive_store.read_doc(doc)
                text = (text or "").strip()[:4000]
                chunks.append(f"{line}\n{text}")
                continue
            except Exception:
                pass
        chunks.append(line)
    return "\n\n".join(chunks)


def build_v2_router(
    drive_store: DriveStore,
    planner: PlannerService,
    reindex_fn: Optional[Callable[[list[DocumentRef]], Dict]] = None,
    indexed_chunks_fn: Optional[Callable[[], Optional[int]]] = None,
    campaign_id: Optional[str] = None,
    workflow_store: Optional[WorkflowStore | NullWorkflowStore] = None,
    world_model_store: Optional[WorldModelStore | NullWorldModelStore] = None,
) -> APIRouter:
    router = APIRouter(tags=["world-v2"])
    applier = ProposalApplier(drive_store=drive_store, reindex_fn=reindex_fn)
    store = workflow_store or NullWorkflowStore()
    model_store = world_model_store or NullWorldModelStore()

    @router.get("/world_status", response_model=WorldStatusResponse)
    def world_status() -> WorldStatusResponse:
        docs = drive_store.list_world_docs()
        folders: Dict[str, int] = {}
        for doc in docs:
            folders[doc.folder] = folders.get(doc.folder, 0) + 1
        indexed_chunks = indexed_chunks_fn() if indexed_chunks_fn else None
        return WorldStatusResponse(
            campaign_id=campaign_id,
            folders=folders,
            docs=docs,
            indexed_chunks=indexed_chunks,
            notes=[
                "Docs come from Google Drive / Docs through DriveStore.",
                "replace_section updates an existing section body or appends it if missing.",
            ],
        )

    @router.get("/list_world_docs")
    def list_world_docs():
        return drive_store.list_world_docs()

    @router.get("/world_model_status", response_model=WorldModelStatusResponse)
    def world_model_status():
        status = model_store.status()
        if not status.campaign_id and campaign_id:
            return WorldModelStatusResponse(
                campaign_id=campaign_id,
                entity_count=status.entity_count,
                thread_count=status.thread_count,
                session_count=status.session_count,
            )
        return status

    @router.get("/entities", response_model=list[WorldEntityRecord])
    def list_entities(limit: int = 20, kind: Optional[str] = None):
        safe_limit = max(1, min(limit, 100))
        return model_store.list_entities(limit=safe_limit, kind=kind)

    @router.get("/threads", response_model=list[WorldThreadRecord])
    def list_threads(limit: int = 20, status: Optional[str] = None):
        safe_limit = max(1, min(limit, 100))
        return model_store.list_threads(limit=safe_limit, status=status)

    @router.get("/sessions", response_model=list[WorldSessionRecord])
    def list_sessions(limit: int = 20):
        safe_limit = max(1, min(limit, 100))
        return model_store.list_sessions(limit=safe_limit)

    @router.get("/proposals", response_model=list[ProposalRecord])
    def list_proposals(limit: int = 20):
        safe_limit = max(1, min(limit, 100))
        return store.list_proposals(limit=safe_limit)

    @router.get("/proposals/{proposal_id}", response_model=ProposalDetail)
    def get_proposal(proposal_id: int):
        proposal = store.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
        return proposal

    @router.get("/apply_runs", response_model=list[ApplyRunRecord])
    def list_apply_runs(limit: int = 20):
        safe_limit = max(1, min(limit, 100))
        return store.list_apply_runs(limit=safe_limit)

    @router.get("/apply_runs/{apply_run_id}", response_model=ApplyRunDetail)
    def get_apply_run(apply_run_id: int):
        apply_run = store.get_apply_run(apply_run_id)
        if not apply_run:
            raise HTTPException(status_code=404, detail="Apply run not found")
        return apply_run

    @router.post("/read_world_doc", response_model=ReadWorldDocResponse)
    def read_world_doc(request: ReadWorldDocRequest) -> ReadWorldDocResponse:
        found = drive_store.find_doc(folder=request.folder, title=request.title, doc_id=request.doc_id)
        if not found:
            raise HTTPException(status_code=404, detail="World document not found")
        content = drive_store.read_doc(found)
        return ReadWorldDocResponse(doc=found, content=content)

    @router.post("/sync_session_patch", response_model=SyncSessionPatchResponse)
    def sync_session_patch(request: SyncSessionPatchRequest) -> SyncSessionPatchResponse:
        if request.campaign_id and campaign_id and request.campaign_id != campaign_id:
            raise HTTPException(status_code=400, detail="campaign_id does not match configured campaign")
        response = model_store.sync_session_patch(request)
        if response is None:
            raise HTTPException(status_code=503, detail="World model store is not configured")
        return response

    @router.post("/propose_changes")
    def propose_changes(request: ProposeChangesRequest):
        docs = drive_store.list_world_docs()
        context = build_context_for_planner(drive_store)
        proposal = planner.propose(request=request, world_docs=docs, world_context=context)
        proposal_id = store.save_proposal(request, proposal)
        if proposal_id is not None:
            proposal = ChangeProposal.model_validate(
                {
                    **proposal.model_dump(mode="json"),
                    "proposal_id": proposal_id,
                }
            )
        return proposal

    @router.post("/apply_changes", response_model=ApplyChangesResponse)
    def apply_changes(request: ApplyChangesRequest) -> ApplyChangesResponse:
        proposal_id = request.proposal_id or request.proposal.proposal_id
        response = applier.apply(request)
        response.proposal_id = proposal_id
        apply_run_id = store.save_apply_run(request, response)
        response.apply_run_id = apply_run_id
        return response

    @router.post("/consistency_check", response_model=ConsistencyCheckResponse)
    def consistency_check(request: ConsistencyCheckRequest) -> ConsistencyCheckResponse:
        context = build_context_for_planner(drive_store)
        raw = planner.consistency_check(instruction=request.instruction, world_context=context)
        return ConsistencyCheckResponse(
            summary="Consistency check completed",
            suggestions=[raw],
        )

    return router
