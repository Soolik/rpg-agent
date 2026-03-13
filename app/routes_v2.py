from __future__ import annotations

from typing import Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from .applier import ProposalApplier
from .drive_store import DriveStore
from .models_v2 import (
    ApplyChangesRequest,
    ApplyChangesResponse,
    ConsistencyCheckRequest,
    ConsistencyCheckResponse,
    ProposeChangesRequest,
    ReadWorldDocRequest,
    ReadWorldDocResponse,
    WorldStatusResponse,
)
from .planner import PlannerService


def build_context_for_planner(drive_store: DriveStore) -> str:
    """
    Placeholder. In production this should read the highest-value documents:
    Campaign Bible, Glossary, Rules And Tone, Thread Tracker and a slice of important entity docs.
    """
    docs = drive_store.list_world_docs()
    if not docs:
        return "No world docs available yet."
    return "\n".join(f"- {doc.folder}/{doc.title}" for doc in docs[:200])


def build_v2_router(
    drive_store: DriveStore,
    planner: PlannerService,
    reindex_fn: Optional[Callable[[], Dict]] = None,
) -> APIRouter:
    router = APIRouter(tags=["world-v2"])
    applier = ProposalApplier(drive_store=drive_store, reindex_fn=reindex_fn)

    @router.get("/world_status", response_model=WorldStatusResponse)
    def world_status() -> WorldStatusResponse:
        docs = drive_store.list_world_docs()
        folders: Dict[str, int] = {}
        for doc in docs:
            folders[doc.folder] = folders.get(doc.folder, 0) + 1
        return WorldStatusResponse(
            folders=folders,
            docs=docs,
            notes=[
                "This response comes from DriveStore.list_world_docs().",
                "Replace DriveStore stub with real Google Drive traversal.",
            ],
        )

    @router.get("/list_world_docs")
    def list_world_docs():
        return drive_store.list_world_docs()

    @router.post("/read_world_doc", response_model=ReadWorldDocResponse)
    def read_world_doc(request: ReadWorldDocRequest) -> ReadWorldDocResponse:
        found = drive_store.find_doc(folder=request.folder, title=request.title, doc_id=request.doc_id)
        if not found:
            raise HTTPException(status_code=404, detail="World document not found")
        content = drive_store.read_doc(found)
        return ReadWorldDocResponse(doc=found, content=content)

    @router.post("/propose_changes")
    def propose_changes(request: ProposeChangesRequest):
        docs = drive_store.list_world_docs()
        context = build_context_for_planner(drive_store)
        proposal = planner.propose(request=request, world_docs=docs, world_context=context)
        return proposal

    @router.post("/apply_changes", response_model=ApplyChangesResponse)
    def apply_changes(request: ApplyChangesRequest) -> ApplyChangesResponse:
        return applier.apply(request)

    @router.post("/consistency_check", response_model=ConsistencyCheckResponse)
    def consistency_check(request: ConsistencyCheckRequest) -> ConsistencyCheckResponse:
        context = build_context_for_planner(drive_store)
        raw = planner.consistency_check(instruction=request.instruction, world_context=context)
        return ConsistencyCheckResponse(
            summary="Consistency check completed",
            suggestions=[raw],
        )

    return router
