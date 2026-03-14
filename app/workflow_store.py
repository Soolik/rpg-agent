from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from .models_v2 import (
    ApplyChangesRequest,
    ApplyChangesResponse,
    ApplyRunDetail,
    ApplyRunRecord,
    ChangeProposal,
    ProposalDetail,
    ProposalRecord,
    ProposeChangesRequest,
)


@dataclass
class WorkflowStore:
    campaign_id: str
    connection_factory: Callable[[], object]

    def save_proposal(
        self,
        request: ProposeChangesRequest,
        proposal: ChangeProposal,
        *,
        proposal_type: str = "general",
        proposal_status: str = "proposed",
        supersedes_proposal_id: Optional[int] = None,
    ) -> int:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into proposals (
                        campaign_id,
                        summary,
                        user_goal,
                        request_json,
                        proposal_json,
                        approved,
                        approved_by
                    )
                    values (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    returning id
                    """,
                    (
                        self.campaign_id,
                        proposal.summary,
                        proposal.user_goal,
                        json.dumps(request.model_dump(mode="json")),
                        json.dumps(proposal.model_dump(mode="json")),
                        False,
                        None,
                    ),
                )
                row = cur.fetchone()
                proposal_id = int(row[0])
                payload = {
                    **proposal.model_dump(mode="json"),
                    "proposal_id": proposal_id,
                    "proposal_type": proposal_type,
                    "proposal_status": proposal_status,
                    "supersedes_proposal_id": supersedes_proposal_id,
                    "accepted_apply_run_id": None,
                    "rejected_reason": None,
                    "reviewed_by": None,
                }
                cur.execute(
                    """
                    update proposals
                    set proposal_json = %s::jsonb,
                        updated_at = now()
                    where campaign_id = %s and id = %s
                    """,
                    (
                        json.dumps(payload),
                        self.campaign_id,
                        proposal_id,
                    ),
                )
            conn.commit()
        return proposal_id

    def save_apply_run(
        self,
        request: ApplyChangesRequest,
        response: ApplyChangesResponse,
    ) -> int:
        proposal_id = request.proposal_id or request.proposal.proposal_id or response.proposal_id
        request_payload = request.model_dump(mode="json")
        response_payload = response.model_dump(mode="json")
        approved_flag = bool(request.approved and response.ok)

        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into apply_runs (
                        campaign_id,
                        proposal_id,
                        approved,
                        approved_by,
                        ok,
                        request_json,
                        response_json
                    )
                    values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    returning id
                    """,
                    (
                        self.campaign_id,
                        proposal_id,
                        request.approved,
                        request.approved_by,
                        response.ok,
                        json.dumps(request_payload),
                        json.dumps(response_payload),
                    ),
                )
                row = cur.fetchone()

                if proposal_id is not None:
                    cur.execute(
                        """
                        update proposals
                        set approved = %s,
                            approved_by = %s,
                            updated_at = now()
                        where campaign_id = %s and id = %s
                        """,
                        (
                            approved_flag,
                            request.approved_by,
                            self.campaign_id,
                            proposal_id,
                        ),
                    )
            conn.commit()
        return int(row[0])

    def list_proposals(self, limit: int = 20) -> list[ProposalRecord]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, campaign_id, summary, user_goal, approved, approved_by, created_at, updated_at
                    from proposals
                    where campaign_id = %s
                    order by created_at desc
                    limit %s
                    """,
                    (self.campaign_id, limit),
                )
                rows = cur.fetchall()
        return [
            ProposalRecord(
                id=row[0],
                campaign_id=row[1],
                summary=row[2],
                user_goal=row[3],
                approved=bool(row[4]),
                approved_by=row[5],
                created_at=row[6].isoformat(),
                updated_at=row[7].isoformat(),
            )
            for row in rows
        ]

    def list_apply_runs(self, limit: int = 20) -> list[ApplyRunRecord]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, campaign_id, proposal_id, approved, approved_by, ok, created_at
                    from apply_runs
                    where campaign_id = %s
                    order by created_at desc
                    limit %s
                    """,
                    (self.campaign_id, limit),
                )
                rows = cur.fetchall()
        return [
            ApplyRunRecord(
                id=row[0],
                campaign_id=row[1],
                proposal_id=row[2],
                approved=bool(row[3]),
                approved_by=row[4],
                ok=bool(row[5]),
                created_at=row[6].isoformat(),
            )
            for row in rows
        ]

    def get_proposal(self, proposal_id: int) -> Optional[ProposalDetail]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, campaign_id, summary, user_goal, approved, approved_by, created_at, updated_at,
                           request_json, proposal_json
                    from proposals
                    where campaign_id = %s and id = %s
                    limit 1
                    """,
                    (self.campaign_id, proposal_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return ProposalDetail(
            id=row[0],
            campaign_id=row[1],
            summary=row[2],
            user_goal=row[3],
            approved=bool(row[4]),
            approved_by=row[5],
            created_at=row[6].isoformat(),
            updated_at=row[7].isoformat(),
            request=row[8] or {},
            proposal={
                **(row[9] or {}),
                "proposal_id": (row[9] or {}).get("proposal_id") or row[0],
            },
        )

    def list_proposal_details(
        self,
        limit: int = 20,
        *,
        proposal_type: Optional[str] = None,
        proposal_status: Optional[str] = None,
    ) -> list[ProposalDetail]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, campaign_id, summary, user_goal, approved, approved_by, created_at, updated_at,
                           request_json, proposal_json
                    from proposals
                    where campaign_id = %s
                    order by created_at desc
                    limit %s
                    """,
                    (self.campaign_id, limit),
                )
                rows = cur.fetchall()

        details: list[ProposalDetail] = []
        for row in rows:
            payload = {
                **(row[9] or {}),
                "proposal_id": (row[9] or {}).get("proposal_id") or row[0],
            }
            if proposal_type and payload.get("proposal_type", "general") != proposal_type:
                continue
            if proposal_status and payload.get("proposal_status", "proposed") != proposal_status:
                continue
            details.append(
                ProposalDetail(
                    id=row[0],
                    campaign_id=row[1],
                    summary=row[2],
                    user_goal=row[3],
                    approved=bool(row[4]),
                    approved_by=row[5],
                    created_at=row[6].isoformat(),
                    updated_at=row[7].isoformat(),
                    request=row[8] or {},
                    proposal=payload,
                )
            )
        return details

    def update_proposal_state(
        self,
        proposal_id: int,
        *,
        proposal_status: str,
        reviewed_by: Optional[str] = None,
        rejected_reason: Optional[str] = None,
        accepted_apply_run_id: Optional[int] = None,
    ) -> Optional[ProposalDetail]:
        current = self.get_proposal(proposal_id)
        if not current:
            return None

        payload = {
            **current.proposal,
            "proposal_id": current.proposal.get("proposal_id") or current.id,
            "proposal_status": proposal_status,
            "reviewed_by": reviewed_by or current.proposal.get("reviewed_by"),
            "rejected_reason": rejected_reason,
            "accepted_apply_run_id": accepted_apply_run_id,
        }

        approved = current.approved
        approved_by = current.approved_by
        if proposal_status == "accepted":
            approved = True
            approved_by = reviewed_by or approved_by
        elif proposal_status in {"rejected", "superseded"}:
            approved = False
            if reviewed_by:
                approved_by = reviewed_by

        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update proposals
                    set approved = %s,
                        approved_by = %s,
                        proposal_json = %s::jsonb,
                        updated_at = now()
                    where campaign_id = %s and id = %s
                    """,
                    (
                        approved,
                        approved_by,
                        json.dumps(payload),
                        self.campaign_id,
                        proposal_id,
                    ),
                )
            conn.commit()

        return self.get_proposal(proposal_id)

    def get_apply_run(self, apply_run_id: int) -> Optional[ApplyRunDetail]:
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, campaign_id, proposal_id, approved, approved_by, ok, created_at,
                           request_json, response_json
                    from apply_runs
                    where campaign_id = %s and id = %s
                    limit 1
                    """,
                    (self.campaign_id, apply_run_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return ApplyRunDetail(
            id=row[0],
            campaign_id=row[1],
            proposal_id=row[2],
            approved=bool(row[3]),
            approved_by=row[4],
            ok=bool(row[5]),
            created_at=row[6].isoformat(),
            request=row[7] or {},
            response=row[8] or {},
        )


class NullWorkflowStore:
    def save_proposal(
        self,
        request: ProposeChangesRequest,
        proposal: ChangeProposal,
        *,
        proposal_type: str = "general",
        proposal_status: str = "proposed",
        supersedes_proposal_id: Optional[int] = None,
    ) -> Optional[int]:
        return None

    def save_apply_run(
        self,
        request: ApplyChangesRequest,
        response: ApplyChangesResponse,
    ) -> Optional[int]:
        return None

    def list_proposals(self, limit: int = 20) -> list[ProposalRecord]:
        return []

    def list_apply_runs(self, limit: int = 20) -> list[ApplyRunRecord]:
        return []

    def get_proposal(self, proposal_id: int) -> Optional[ProposalDetail]:
        return None

    def list_proposal_details(
        self,
        limit: int = 20,
        *,
        proposal_type: Optional[str] = None,
        proposal_status: Optional[str] = None,
    ) -> list[ProposalDetail]:
        return []

    def update_proposal_state(
        self,
        proposal_id: int,
        *,
        proposal_status: str,
        reviewed_by: Optional[str] = None,
        rejected_reason: Optional[str] = None,
        accepted_apply_run_id: Optional[int] = None,
    ) -> Optional[ProposalDetail]:
        return None

    def get_apply_run(self, apply_run_id: int) -> Optional[ApplyRunDetail]:
        return None
