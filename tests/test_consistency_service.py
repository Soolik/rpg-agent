import unittest

from app.consistency_service import ConsistencyService
from app.models_v2 import DocumentExecutionPreview, DocumentRef, WorldEntityRecord, WorldThreadRecord
from app.world_fact_store import ProjectionFact, WorldFactStore


class FakeWorldModelStore:
    def list_entities(self, limit=1000):
        return [
            WorldEntityRecord(
                id=1,
                campaign_id="kng",
                entity_kind="npc",
                name="Captain Mira",
                description="Captain Mira wspolpracuje z Red Blade.",
                tags=[],
                last_session_id=None,
                updated_at="2026-03-15T00:00:00+00:00",
            )
        ]

    def list_threads(self, limit=500):
        return [
            WorldThreadRecord(
                id=2,
                campaign_id="kng",
                thread_key="t01",
                thread_id="T01",
                title="Red Blade",
                status="active",
                last_change="Captain Mira podtrzymuje kontakt.",
                last_session_id=None,
                updated_at="2026-03-15T00:00:00+00:00",
            )
        ]


class FakeWorldFactStore:
    def list_facts(self, *, limit=50, subject_name=None, predicate=None):
        facts = [
            ProjectionFact(
                subject_type="thread",
                subject_name="Red Blade",
                predicate="status",
                object_value="active",
                source_type="world_model.thread",
            )
        ]
        rows = []
        for index, fact in enumerate(facts, start=1):
            if subject_name and fact.subject_name != subject_name:
                continue
            if predicate and fact.predicate != predicate:
                continue
            rows.append(
                type(
                    "WorldFact",
                    (),
                    {
                        "id": index,
                        "campaign_id": "kng",
                        "subject_type": fact.subject_type,
                        "subject_name": fact.subject_name,
                        "predicate": fact.predicate,
                        "object_value": fact.object_value,
                        "source_type": fact.source_type,
                        "source_ref": None,
                        "confidence": 1.0,
                        "updated_at": "2026-03-15T00:00:00+00:00",
                    },
                )()
            )
        return rows[:limit]


class ConsistencyServiceTest(unittest.TestCase):
    def test_soft_guard_detects_fact_conflict(self):
        service = ConsistencyService(
            world_model_store=FakeWorldModelStore(),
            world_fact_store=FakeWorldFactStore(),
        )

        report = service.soft_guard(
            message="Sprawdz continuity.",
            generated_text="Name: Red Blade\nStatus: closed",
        )

        self.assertFalse(report.ok)
        self.assertTrue(any(issue.code == "fact_conflict" for issue in report.issues))

    def test_hard_validate_blocks_conflicting_preview(self):
        service = ConsistencyService(
            world_model_store=FakeWorldModelStore(),
            world_fact_store=FakeWorldFactStore(),
        )
        preview = DocumentExecutionPreview(
            action_type="replace_section",
            target=DocumentRef(folder="06 Threads", title="Red Blade", doc_id="doc-1"),
            summary="Preview replace section Status",
            current_excerpt="Status: active",
            proposed_excerpt="Name: Red Blade\nStatus: closed",
            diff_text="",
        )

        report = service.hard_validate([preview])

        self.assertFalse(report.ok)
        self.assertTrue(any(issue.code == "fact_conflict" for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
