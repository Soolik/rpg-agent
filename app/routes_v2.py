from __future__ import annotations

from typing import Callable, Dict, Optional

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
    reindex_fn: Optional[Callable[[], Dict]] = None,
    indexed_chunks_fn: Optional[Callable[[], Optional[int]]] = None,
    campaign_id: Optional[str] = None,
) -> APIRouter:
    router = APIRouter(tags=["world-v2"])
    applier = ProposalApplier(drive_store=drive_store, reindex_fn=reindex_fn)

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
                "replace_section currently appends a marked section block in MVP mode.",
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
