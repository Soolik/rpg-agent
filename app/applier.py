from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .drive_store import DriveStore
from .models_v2 import (
    ActionType,
    ApplyChangesRequest,
    ApplyChangesResponse,
    AppliedActionResult,
    DocumentRef,
)


class ProposalApplier:
    def __init__(self, drive_store: DriveStore, reindex_fn: Optional[Callable[[], Dict]] = None):
        self.drive_store = drive_store
        self.reindex_fn = reindex_fn

    def apply(self, request: ApplyChangesRequest) -> ApplyChangesResponse:
        if not request.approved:
            return ApplyChangesResponse(ok=False, summary="Changes were not approved.", results=[])

        results: List[AppliedActionResult] = []

        for action in request.proposal.actions:
            try:
                target = action.target

                if action.action_type == ActionType.create_doc:
                    if not target:
                        raise ValueError("create_doc requires target")
                    created = self.drive_store.create_doc(
                        folder=target.folder,
                        title=target.title,
                        content=action.content or "",
                        entity_type=action.entity_type,
                    )
                    results.append(AppliedActionResult(
                        action_type=action.action_type,
                        success=True,
                        message="Document created",
                        target=DocumentRef(folder=created.folder, title=created.title, doc_id=created.doc_id, path_hint=created.path_hint),
                    ))

                elif action.action_type == ActionType.append_doc:
                    if not target:
                        raise ValueError("append_doc requires target")
                    self.drive_store.append_doc(target, action.content or "")
                    results.append(AppliedActionResult(
                        action_type=action.action_type,
                        success=True,
                        message="Content appended",
                        target=target,
                    ))

                elif action.action_type == ActionType.replace_doc:
                    if not target:
                        raise ValueError("replace_doc requires target")
                    self.drive_store.replace_doc(target, action.content or "")
                    results.append(AppliedActionResult(
                        action_type=action.action_type,
                        success=True,
                        message="Document replaced",
                        target=target,
                    ))

                elif action.action_type == ActionType.replace_section:
                    if not target:
                        raise ValueError("replace_section requires target")
                    if not action.section:
                        raise ValueError("replace_section requires section")
                    self.drive_store.replace_section(target, action.section, action.content or "")
                    results.append(AppliedActionResult(
                        action_type=action.action_type,
                        success=True,
                        message=f"Section '{action.section}' replaced",
                        target=target,
                    ))

                elif action.action_type in {ActionType.create_if_missing, ActionType.update_tracker_row, ActionType.reindex}:
                    results.append(AppliedActionResult(
                        action_type=action.action_type,
                        success=True,
                        message="Action acknowledged but requires concrete implementation.",
                        target=target,
                    ))

                else:
                    raise ValueError(f"Unsupported action type: {action.action_type}")

            except Exception as exc:
                results.append(AppliedActionResult(
                    action_type=action.action_type,
                    success=False,
                    message=str(exc),
                    target=action.target,
                ))

        reindex_result = None
        if request.reindex_after_apply and self.reindex_fn:
            try:
                reindex_result = self.reindex_fn()
            except Exception as exc:
                reindex_result = {"ok": False, "error": str(exc)}

        ok = all(r.success for r in results)
        return ApplyChangesResponse(
            ok=ok,
            summary="Apply finished" if ok else "Apply finished with errors",
            results=results,
            reindex_result=reindex_result,
        )
