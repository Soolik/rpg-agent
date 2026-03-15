from __future__ import annotations

from dataclasses import dataclass
import re

from .api_models import ContinuityIssue, ContinuityReport
from .canon_guard import build_continuity_report
from .models_v2 import ChangeValidationReport, DocumentExecutionPreview, ValidationIssue
from .world_fact_store import NullWorldFactStore, WorldFactStore, normalize_key
from .world_model_store import NullWorldModelStore, WorldModelStore


STRUCTURED_FACT_PATTERNS = (
    ("status", re.compile(r"(?im)^\s*(?:status|stan)\s*:\s*(.+?)\s*$")),
    ("thread_id", re.compile(r"(?im)^\s*(?:thread id|id watku|watek id)\s*:\s*(.+?)\s*$")),
)


def _extract_subject_name(text: str, fallback: str = "") -> str:
    for pattern in (
        re.compile(r"(?im)^\s*(?:name|imie|tytul)\s*:\s*(.+?)\s*$"),
        re.compile(r"(?im)^\s*#*\s*([A-Z][^\n]{2,80})\s*$"),
    ):
        match = pattern.search(text or "")
        if match:
            return match.group(1).strip()
    return fallback.strip()


def _candidate_facts(subject_name: str, text: str) -> list[tuple[str, str, str]]:
    facts: list[tuple[str, str, str]] = []
    if not subject_name:
        return facts
    for predicate, pattern in STRUCTURED_FACT_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = " ".join(match.group(1).split()).strip(" .")
            if value:
                facts.append((subject_name, predicate, value))
    return facts


def _fact_conflicts(
    *,
    subject_name: str,
    text: str,
    world_fact_store: WorldFactStore | NullWorldFactStore,
) -> list[ContinuityIssue]:
    if not subject_name:
        return []
    issues: list[ContinuityIssue] = []
    known_by_predicate: dict[str, list[str]] = {}
    for fact in world_fact_store.list_facts(limit=100, subject_name=subject_name):
        known_by_predicate.setdefault(fact.predicate, []).append(fact.object_value)
    for _, predicate, value in _candidate_facts(subject_name, text):
        known_values = known_by_predicate.get(predicate, [])
        if not known_values:
            continue
        if any(normalize_key(value) == normalize_key(candidate) for candidate in known_values):
            continue
        issues.append(
            ContinuityIssue(
                code="fact_conflict",
                severity="error",
                message=f"Pole `{predicate}` dla `{subject_name}` ma wartosc `{value}`, ale znany stan to `{known_values[0]}`.",
                related_name=subject_name,
                evidence=known_values[0],
                source="world_model",
            )
        )
    return issues


@dataclass
class ConsistencyService:
    world_model_store: WorldModelStore | NullWorldModelStore
    world_fact_store: WorldFactStore | NullWorldFactStore

    def soft_guard(
        self,
        *,
        message: str,
        generated_text: str,
        artifact_type: str | None = None,
        extra_allowed_names: list[str] | None = None,
    ) -> ContinuityReport:
        entities = self.world_model_store.list_entities(limit=1000)
        threads = self.world_model_store.list_threads(limit=500)
        allow_new_names = artifact_type == "npc_brief"
        report = build_continuity_report(
            message=message,
            generated_text=generated_text,
            known_entity_names=[entity.name for entity in entities],
            known_thread_names=[thread.title for thread in threads],
            extra_allowed_names=extra_allowed_names or [],
            allow_proposed_new_names=allow_new_names,
        )
        subject_name = _extract_subject_name(generated_text)
        report.issues.extend(
            _fact_conflicts(subject_name=subject_name, text=generated_text, world_fact_store=self.world_fact_store)
        )
        report.ok = not any(issue.severity in {"warning", "error"} for issue in report.issues)
        return report

    def hard_validate(self, previews: list[DocumentExecutionPreview]) -> ChangeValidationReport:
        issues: list[ValidationIssue] = []
        for preview in previews:
            if not preview.proposed_excerpt.strip():
                continue
            subject_name = preview.target.title if preview.target else ""
            for continuity_issue in _fact_conflicts(
                subject_name=subject_name,
                text=preview.proposed_excerpt,
                world_fact_store=self.world_fact_store,
            ):
                issues.append(
                    ValidationIssue(
                        code=continuity_issue.code,
                        severity=continuity_issue.severity,
                        message=continuity_issue.message,
                        target=preview.target,
                        evidence=continuity_issue.evidence,
                    )
                )
            guard_report = self.soft_guard(
                message=preview.summary or subject_name,
                generated_text=preview.proposed_excerpt,
                artifact_type=None,
            )
            for continuity_issue in guard_report.issues:
                if continuity_issue.severity not in {"warning", "error"}:
                    continue
                issues.append(
                    ValidationIssue(
                        code=continuity_issue.code,
                        severity=continuity_issue.severity,
                        message=continuity_issue.message,
                        target=preview.target,
                        evidence=continuity_issue.evidence,
                    )
                )
        deduped: list[ValidationIssue] = []
        seen = set()
        for issue in issues:
            key = (
                issue.code,
                issue.severity,
                issue.message,
                issue.target.doc_id if issue.target else "",
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return ChangeValidationReport(
            ok=not any(issue.severity == "error" for issue in deduped),
            issues=deduped,
        )
