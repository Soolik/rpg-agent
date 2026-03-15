import unittest

from app.models_v2 import WorldEntityRecord, WorldSessionRecord, WorldThreadRecord
from app.world_model_service import WorldModelService


class FakeWorldModelStore:
    def list_entities(self, limit=20, kind=None):
        return [
            WorldEntityRecord(
                id=1,
                campaign_id="kng",
                entity_kind="npc",
                name="Captain Mira",
                description="Captain tied to Red Blade.",
                tags=["red-blade"],
                last_session_id=None,
                updated_at="2026-03-15T00:00:00+00:00",
            )
        ]

    def list_threads(self, limit=20, status=None):
        return [
            WorldThreadRecord(
                id=2,
                campaign_id="kng",
                thread_key="T01",
                thread_id="T01",
                title="Red Blade",
                status="active",
                last_change="Captain Mira negotiated with them.",
                last_session_id=None,
                updated_at="2026-03-15T00:00:00+00:00",
            )
        ]

    def list_sessions(self, limit=20):
        return [
            WorldSessionRecord(
                id=3,
                campaign_id="kng",
                session_summary="Captain Mira met Red Blade.",
                entity_count=1,
                thread_count=1,
                source_title="Session 06",
                created_at="2026-03-15T00:00:00+00:00",
            )
        ]


class WorldModelServiceTest(unittest.TestCase):
    def test_search_returns_ranked_hits(self):
        service = WorldModelService(world_model_store=FakeWorldModelStore())

        hits = service.search("Red Blade", limit=10)

        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].record_type, "thread")
        self.assertTrue(any(hit.record_type == "entity" for hit in hits))
        self.assertTrue(any(hit.record_type == "session" for hit in hits))

    def test_search_scans_large_doc_seeded_entity_sets(self):
        class ManyEntitiesStore(FakeWorldModelStore):
            def list_entities(self, limit=20, kind=None):
                rows = []
                for idx in range(150):
                    rows.append(
                        WorldEntityRecord(
                            id=idx + 1,
                            campaign_id="kng",
                            entity_kind="canon",
                            name=f"Canon Name {idx}",
                            description="Imported from docs.",
                            tags=[],
                            last_session_id=None,
                            updated_at="2026-03-15T00:00:00+00:00",
                        )
                    )
                rows.append(
                    WorldEntityRecord(
                        id=999,
                        campaign_id="kng",
                        entity_kind="location",
                        name="Port Peril",
                        description="Shackles port.",
                        tags=["places"],
                        last_session_id=None,
                        updated_at="2026-03-15T00:00:00+00:00",
                    )
                )
                return rows[:limit]

        service = WorldModelService(world_model_store=ManyEntitiesStore())

        hits = service.search("Port Peril", limit=10)

        self.assertTrue(any(hit.title == "Port Peril" for hit in hits))


if __name__ == "__main__":
    unittest.main()
