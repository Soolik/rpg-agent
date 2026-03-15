import unittest

from app.api_models import WorldModelSearchItem
from app.models_v2 import WorldDocInfo
from app.retrieval_service import HybridRetrievalService


class HybridRetrievalServiceTest(unittest.TestCase):
    def test_structured_search_supports_keyword_only_limit(self):
        docs = [
            WorldDocInfo(
                folder="01 Locations",
                title="Port Peril",
                doc_id="doc-port-peril",
                path_hint="/world/locations/port-peril",
            )
        ]
        structured_calls: list[tuple[str, int]] = []

        def structured_search(query: str, *, limit: int = 20):
            structured_calls.append((query, limit))
            return [
                WorldModelSearchItem(
                    record_type="entity",
                    record_id=1,
                    title="Port Peril",
                    snippet="Major city in the Shackles.",
                    score=100,
                )
            ]

        service = HybridRetrievalService(
            vector_search_fn=lambda question, limit: [],
            vector_search_in_docs_fn=lambda question, doc_ids, limit: [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-port-peril",
                    "title": "Port Peril",
                    "path_hint": "/world/locations/port-peril",
                    "chunk_text": "Port Peril is the largest port in the Shackles.",
                    "distance": 0.15,
                }
            ],
            lexical_search_fn=lambda question, limit: [],
            lexical_search_in_docs_fn=lambda question, doc_ids, limit: [],
            list_docs_fn=lambda: docs,
            structured_search_fn=structured_search,
        )

        hits = service.retrieve("Co to Port Peril?", top_k=3)

        self.assertEqual(structured_calls, [("Co to Port Peril?", 10)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["doc_id"], "doc-port-peril")


if __name__ == "__main__":
    unittest.main()
