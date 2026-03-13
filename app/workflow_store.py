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

    def save_proposal(self, request: ProposeChangesRequest, proposal: ChangeProposal) -> int:
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
                            request.approved,
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
    def save_proposal(self, request: ProposeChangesRequest, proposal: ChangeProposal) -> Optional[int]:
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

    def get_apply_run(self, apply_run_id: int) -> Optional[ApplyRunDetail]:
        return None
