from __future__ import annotations

import json
from textwrap import dedent
from typing import Callable, List, Optional

from .templates import FACTION_TEMPLATE, LOCATION_TEMPLATE, NPC_TEMPLATE, SECRET_TEMPLATE, THREAD_TEMPLATE
from .models_v2 import (
    ChangeProposal,
    DocumentAction,
    DocumentRef,
    ProposeChangesRequest,
    WorldDocInfo,
)


class PlannerService:
    def __init__(self, generate_text_fn: Callable[[str], str]):
        self.generate_text_fn = generate_text_fn

    def propose(self, request: ProposeChangesRequest, world_docs: List[WorldDocInfo], world_context: str) -> ChangeProposal:
        prompt = self._build_prompt(request=request, world_docs=world_docs, world_context=world_context)
        raw = self.generate_text_fn(prompt)
        return self._parse_or_fallback(raw=raw, request=request)

    def consistency_check(self, instruction: str, world_context: str) -> str:
        prompt = dedent(f"""
        You are a campaign continuity editor.
        Analyze the instruction against the world context.
        Return a concise report with:
        1. conflicts
        2. overlaps
        3. missing links
        4. suggestions

        INSTRUCTION:
        {instruction}

        WORLD CONTEXT:
        {world_context}
        """).strip()
        return self.generate_text_fn(prompt)

    def _build_prompt(self, request: ProposeChangesRequest, world_docs: List[WorldDocInfo], world_context: str) -> str:
        docs_preview = "\n".join(f"- {d.folder}/{d.title}" for d in world_docs[:100]) or "- No docs found"
        allowed_folders = ", ".join(sorted({d.folder for d in world_docs if d.folder})) or "00 Admin, 01 Bible, 02 Sessions, 03 NPC, 04 Locations, 05 Factions, 06 Threads, 07 Secrets, 08 Outputs"
        return dedent(f"""
        You are a zero-touch worldbuilding planner for an RPG campaign.
        Your job is to convert a user instruction into a structured JSON change proposal.

        Rules:
        - Do not execute changes.
        - Propose only file/document operations.
        - If the user asks to create a new world entity, choose the right world folder.
        - Update Campaign Bible only for durable world facts.
        - Update Thread Tracker only for plot-level changes.
        - Do not invent certainty where there is not enough context. Put assumptions into `assumptions`.
        - Always set `needs_confirmation` to true.
        - Use only exact logical folder names from this allowed list: {allowed_folders}
        - Never invent folder names such as "03 NPCs". Use exact names like "03 NPC".
        - If a target document already exists in Existing world docs, reuse its exact title and folder.
        - If a target document does not exist, create it in the exact canonical folder name.
        - Keep proposed document content in Polish.
        - Prefer human-readable titles like "Captain Mira", unless existing docs clearly use another naming convention.

        Allowed action_type values:
        - create_doc
        - append_doc
        - replace_doc
        - replace_section
        - create_if_missing
        - update_tracker_row
        - reindex

        User request mode: {request.mode}
        User instruction: {request.instruction}

        Existing world docs:
        {docs_preview}

        World context:
        {world_context}

        Recommended document templates:
        NPC:
        {NPC_TEMPLATE}

        Location:
        {LOCATION_TEMPLATE}

        Faction:
        {FACTION_TEMPLATE}

        Thread:
        {THREAD_TEMPLATE}

        Secret:
        {SECRET_TEMPLATE}

        Return JSON only matching this rough structure:
        {{
          "summary": "...",
          "user_goal": "...",
          "assumptions": ["..."],
          "impacted_docs": [{{"folder": "...", "title": "..."}}],
          "actions": [
            {{
              "action_type": "create_doc",
              "entity_type": "faction",
              "target": {{"folder": "05 Factions", "title": "Faction_..."}},
              "content": "...",
              "reason": "..."
            }}
          ],
          "needs_confirmation": true
        }}
        """).strip()

    def _parse_or_fallback(self, raw: str, request: ProposeChangesRequest) -> ChangeProposal:
        try:
            data = json.loads(raw)
            return ChangeProposal.model_validate(data)
        except Exception:
            return ChangeProposal(
                summary="Fallback proposal generated because planner did not return valid JSON.",
                user_goal=request.instruction,
                assumptions=["Planner output was not valid JSON and needs manual review."],
                impacted_docs=[],
                actions=[
                    DocumentAction(
                        action_type="reindex",  # type: ignore[arg-type]
                        reason="Fallback no-op action",
                    )
                ],
                needs_confirmation=True,
            )
