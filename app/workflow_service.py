from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .api_models import ProposalStatus, ProposalType, WorldModelChangeView
from .applier import ProposalApplier
from .models_v2 import ApplyChangesRequest, ChangeProposal, ProposeChangesRequest
from .routes_v2 import build_context_for_planner
from .workflow_store import NullWorkflowStore, WorkflowStore


@dataclass
class WorkflowApplyResult:
    proposal: WorldModelChangeView
    apply_run_id: Optional[int]
    ok: bool
    summary: str
    results: list
    reindex_result: Optional[dict]


def proposal_view_from_detail(detail) -> WorldModelChangeView:
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


@dataclass
class WorkflowService:
    drive_store: object
    planner: object
    workflow_store: WorkflowStore | NullWorkflowStore
    applier: ProposalApplier

    def propose_change(
        self,
        *,
        instruction: str,
        mode: str,
        dry_run: bool,
        supersedes_proposal_id: Optional[int] = None,
    ) -> WorldModelChangeView:
        docs = self.drive_store.list_world_docs()
        context = build_context_for_planner(self.drive_store)
        proposal_request = ProposeChangesRequest(
            instruction=instruction,
            mode=mode,
            dry_run=dry_run,
        )
        proposal = self.planner.propose(request=proposal_request, world_docs=docs, world_context=context)
        proposal_id = self.workflow_store.save_proposal(
            proposal_request,
            proposal,
            proposal_type=ProposalType.world_model_change.value,
            proposal_status=ProposalStatus.proposed.value,
            supersedes_proposal_id=supersedes_proposal_id,
        )
        if proposal_id is None:
            raise RuntimeError("Proposal could not be persisted.")
        detail = self.workflow_store.get_proposal(proposal_id)
        if not detail:
            raise RuntimeError("Proposal was generated but could not be loaded back from workflow store.")
        return proposal_view_from_detail(detail)

    def list_changes(
        self,
        *,
        limit: int,
        status: Optional[ProposalStatus] = None,
    ) -> list[WorldModelChangeView]:
        details = self.workflow_store.list_proposal_details(
            limit=limit,
            proposal_type=ProposalType.world_model_change.value,
            proposal_status=status.value if status else None,
        )
        return [proposal_view_from_detail(detail) for detail in details]

    def get_change(self, proposal_id: int) -> Optional[WorldModelChangeView]:
        detail = self.workflow_store.get_proposal(proposal_id)
        if not detail:
            return None
        return proposal_view_from_detail(detail)

    def accept_change(
        self,
        *,
        proposal_id: int,
        actor: Optional[str],
        reindex_after_apply: bool,
    ) -> Optional[WorkflowApplyResult]:
        detail = self.workflow_store.get_proposal(proposal_id)
        if not detail:
            return None

        proposal = ChangeProposal.model_validate(detail.proposal)
        apply_request = ApplyChangesRequest(
            proposal_id=proposal_id,
            proposal=proposal,
            approved=True,
            approved_by=actor,
            reindex_after_apply=reindex_after_apply,
        )
        apply_response = self.applier.apply(apply_request)
        apply_response.proposal_id = proposal_id
        apply_run_id = self.workflow_store.save_apply_run(apply_request, apply_response)

        updated_detail = detail
        if apply_response.ok:
            updated_detail = self.workflow_store.update_proposal_state(
                proposal_id,
                proposal_status=ProposalStatus.accepted.value,
                reviewed_by=actor,
                accepted_apply_run_id=apply_run_id,
            ) or detail
            supersedes_proposal_id = detail.proposal.get("supersedes_proposal_id")
            if supersedes_proposal_id:
                self.workflow_store.update_proposal_state(
                    int(supersedes_proposal_id),
                    proposal_status=ProposalStatus.superseded.value,
                    reviewed_by=actor,
                )

        return WorkflowApplyResult(
            proposal=proposal_view_from_detail(updated_detail),
            apply_run_id=apply_run_id,
            ok=apply_response.ok,
            summary=apply_response.summary,
            results=apply_response.results,
            reindex_result=apply_response.reindex_result,
        )

    def reject_change(
        self,
        *,
        proposal_id: int,
        actor: Optional[str],
        reason: Optional[str],
    ) -> Optional[WorldModelChangeView]:
        updated_detail = self.workflow_store.update_proposal_state(
            proposal_id,
            proposal_status=ProposalStatus.rejected.value,
            reviewed_by=actor,
            rejected_reason=reason,
        )
        if not updated_detail:
            return None
        return proposal_view_from_detail(updated_detail)
