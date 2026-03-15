from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .api_models import ContinuityReport
from .models_v2 import ChangeProposal


def _normalize(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _contains_all(text: str, expected: list[str]) -> list[str]:
    haystack = _normalize(text)
    return [item for item in expected if _normalize(item) not in haystack]


def _contains_forbidden(text: str, forbidden: list[str]) -> list[str]:
    haystack = _normalize(text)
    return [item for item in forbidden if _normalize(item) in haystack]


@dataclass(frozen=True)
class QuestionEvalCase:
    name: str
    prompt: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlannerEvalCase:
    name: str
    instruction: str
    expected_action_types: list[str] = field(default_factory=list)
    expected_titles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContinuityEvalCase:
    name: str
    message: str
    generated_text: str
    expected_issue_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalSuite:
    questions: list[QuestionEvalCase] = field(default_factory=list)
    planner_cases: list[PlannerEvalCase] = field(default_factory=list)
    continuity_cases: list[ContinuityEvalCase] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCaseResult:
    suite: str
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalRunResult:
    passed: int
    failed: int
    results: list[EvalCaseResult]

    @property
    def ok(self) -> bool:
        return self.failed == 0


@dataclass
class EvalHarness:
    answer_fn: Optional[Callable[[str], str]] = None
    planner_fn: Optional[Callable[[str], ChangeProposal]] = None
    continuity_fn: Optional[Callable[[str, str], ContinuityReport]] = None

    @staticmethod
    def load_suite(path: str | Path) -> EvalSuite:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return EvalSuite(
            questions=[
                QuestionEvalCase(
                    name=item["name"],
                    prompt=item["prompt"],
                    must_include=list(item.get("must_include") or []),
                    must_not_include=list(item.get("must_not_include") or []),
                )
                for item in payload.get("questions", [])
            ],
            planner_cases=[
                PlannerEvalCase(
                    name=item["name"],
                    instruction=item["instruction"],
                    expected_action_types=list(item.get("expected_action_types") or []),
                    expected_titles=list(item.get("expected_titles") or []),
                )
                for item in payload.get("planner_cases", [])
            ],
            continuity_cases=[
                ContinuityEvalCase(
                    name=item["name"],
                    message=item["message"],
                    generated_text=item["generated_text"],
                    expected_issue_codes=list(item.get("expected_issue_codes") or []),
                )
                for item in payload.get("continuity_cases", [])
            ],
        )

    def run_suite(self, suite: EvalSuite) -> EvalRunResult:
        results: list[EvalCaseResult] = []

        if self.answer_fn:
            for case in suite.questions:
                answer = self.answer_fn(case.prompt)
                missing = _contains_all(answer, case.must_include)
                forbidden = _contains_forbidden(answer, case.must_not_include)
                details = [f"missing:{item}" for item in missing] + [f"forbidden:{item}" for item in forbidden]
                results.append(
                    EvalCaseResult(
                        suite="questions",
                        name=case.name,
                        passed=not details,
                        details=details,
                    )
                )

        if self.planner_fn:
            for case in suite.planner_cases:
                proposal = self.planner_fn(case.instruction)
                action_types = [action.action_type.value for action in proposal.actions]
                titles = [action.target.title for action in proposal.actions if action.target and action.target.title]
                missing_actions = [item for item in case.expected_action_types if item not in action_types]
                missing_titles = [item for item in case.expected_titles if item not in titles]
                details = [f"missing_action:{item}" for item in missing_actions] + [f"missing_title:{item}" for item in missing_titles]
                results.append(
                    EvalCaseResult(
                        suite="planner",
                        name=case.name,
                        passed=not details,
                        details=details,
                    )
                )

        if self.continuity_fn:
            for case in suite.continuity_cases:
                report = self.continuity_fn(case.message, case.generated_text)
                issue_codes = [issue.code for issue in report.issues]
                missing_codes = [code for code in case.expected_issue_codes if code not in issue_codes]
                details = [f"missing_issue:{code}" for code in missing_codes]
                results.append(
                    EvalCaseResult(
                        suite="continuity",
                        name=case.name,
                        passed=not details,
                        details=details,
                    )
                )

        passed = sum(1 for result in results if result.passed)
        failed = len(results) - passed
        return EvalRunResult(passed=passed, failed=failed, results=results)
