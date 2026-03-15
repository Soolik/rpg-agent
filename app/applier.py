from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .consistency_service import ConsistencyService
from .document_executor import DocumentExecutor
from .drive_store import DriveStore
from .models_v2 import (
    ApplyChangesRequest,
    ApplyChangesResponse,
    AppliedActionResult,
    DocumentRef,
)


class ProposalApplier:
    def __init__(
        self,
        drive_store: DriveStore,
        reindex_fn: Optional[Callable[[List[DocumentRef]], Dict]] = None,
        consistency_service: Optional[ConsistencyService] = None,
        document_executor: Optional[DocumentExecutor] = None,
    ):
        self.drive_store = drive_store
        self.reindex_fn = reindex_fn
        self.consistency_service = consistency_service
        self.document_executor = document_executor or DocumentExecutor(drive_store=drive_store)

    def apply(self, request: ApplyChangesRequest) -> ApplyChangesResponse:
        if not request.approved:
            return ApplyChangesResponse(ok=False, summary="Changes were not approved.", results=[])

        results: List[AppliedActionResult] = []
        reindex_targets: List[DocumentRef] = []
        prepared_actions = [self.document_executor.preview_action(action) for action in request.proposal.actions]
        previews = [prepared.preview for prepared in prepared_actions]

        validation = None
        if self.consistency_service:
            validation = self.consistency_service.hard_validate(previews)
            if not validation.ok:
                return ApplyChangesResponse(
                    ok=False,
                    summary="Apply blocked by hard validation.",
                    results=[],
                    previews=previews,
                    validation=validation,
                )

        for prepared in prepared_actions:
            try:
                result = self.document_executor.apply_prepared(prepared)
                results.append(result)
                if result.target and result.target.doc_id:
                    reindex_targets.append(result.target)
            except Exception as exc:
                results.append(AppliedActionResult(
                    action_type=prepared.action.action_type,
                    success=False,
                    message=str(exc),
                    target=prepared.action.target,
                    details={"preview": prepared.preview.model_dump(mode="json"), "verified": False},
                ))

        reindex_result = None
        if request.reindex_after_apply and self.reindex_fn:
            try:
                reindex_result = self.reindex_fn(reindex_targets)
            except Exception as exc:
                reindex_result = {"ok": False, "error": str(exc)}

        ok = all(r.success for r in results)
        return ApplyChangesResponse(
            ok=ok,
            summary="Apply finished" if ok else "Apply finished with errors",
            results=results,
            reindex_result=reindex_result,
            previews=previews,
            validation=validation,
        )
