from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main
from app.eval_harness import EvalHarness
from app.chat_models import AskRequest
from app.models_v2 import ProposeChangesRequest


def build_harness() -> EvalHarness:
    def answer_fn(prompt: str) -> str:
        response = main.ask(AskRequest(question=prompt, mode="auto", include_sources=False))
        return response.answer

    def planner_fn(instruction: str):
        docs = main.drive_store_v2.list_world_docs()
        context = main.build_planner_context_for_instruction(instruction)
        request = ProposeChangesRequest(instruction=instruction, mode="auto", dry_run=True)
        return main.planner_v2.propose(request=request, world_docs=docs, world_context=context)

    def continuity_fn(message: str, generated_text: str):
        if not main.consistency_service_v1:
            raise RuntimeError("Consistency service is not configured.")
        return main.consistency_service_v1.soft_guard(message=message, generated_text=generated_text)

    return EvalHarness(
        answer_fn=answer_fn,
        planner_fn=planner_fn,
        continuity_fn=continuity_fn,
    )


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Run the RPG agent eval harness.")
    parser.add_argument("suite", help="Path to eval suite JSON file.")
    args = parser.parse_args()

    harness = build_harness()
    suite = harness.load_suite(Path(args.suite))
    result = harness.run_suite(suite)
    print(json.dumps(
        {
            "ok": result.ok,
            "passed": result.passed,
            "failed": result.failed,
            "results": [
                {
                    "suite": item.suite,
                    "name": item.name,
                    "passed": item.passed,
                    "details": item.details,
                }
                for item in result.results
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
