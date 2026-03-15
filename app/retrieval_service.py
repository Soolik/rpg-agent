from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import math
import re

from .api_models import WorldModelSearchItem
from .models_v2 import WorldDocInfo


def normalize_key(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def query_terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9_-]{2,}", normalize_key(value)) if term]


@dataclass
class HybridRetrievalService:
    vector_search_fn: Callable[[str, int], list[dict]]
    vector_search_in_docs_fn: Callable[[str, list[str], int], list[dict]]
    lexical_search_fn: Callable[[str, int], list[dict]]
    lexical_search_in_docs_fn: Callable[[str, list[str], int], list[dict]]
    list_docs_fn: Callable[[], list[WorldDocInfo]]
    structured_search_fn: Optional[Callable[..., list[WorldModelSearchItem]]] = None

    def retrieve(self, question: str, *, top_k: int = 6) -> list[dict]:
        if not question.strip():
            return []

        seed_limit = min(max(top_k * 3, 10), 24)
        doc_candidates = self._doc_candidates(question, limit=seed_limit)
        seeded_doc_ids = [doc.doc_id for doc in doc_candidates if doc.doc_id]
        seeded_doc_ids.extend(self._structured_doc_ids(question, limit=seed_limit))
        seeded_doc_ids = list(dict.fromkeys(seeded_doc_ids))

        lists: list[list[dict]] = []
        vector_hits = self.vector_search_fn(question, seed_limit)
        lexical_hits = self.lexical_search_fn(question, seed_limit)
        if vector_hits:
            lists.append(vector_hits)
        if lexical_hits:
            lists.append(lexical_hits)

        if seeded_doc_ids:
            scoped_vector = self.vector_search_in_docs_fn(question, seeded_doc_ids, seed_limit)
            scoped_lexical = self.lexical_search_in_docs_fn(question, seeded_doc_ids, seed_limit)
            if scoped_vector:
                lists.append(scoped_vector)
            if scoped_lexical:
                lists.append(scoped_lexical)

        ranked = self._rerank(question, lists, doc_candidates)
        per_doc_limit = 2
        results: list[dict] = []
        per_doc_counts: dict[str, int] = {}
        for hit in ranked:
            doc_id = str(hit.get("doc_id") or "")
            if doc_id and per_doc_counts.get(doc_id, 0) >= per_doc_limit:
                continue
            if doc_id:
                per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
            results.append(hit)
            if len(results) >= max(top_k + 4, 8):
                break
        return results

    def _structured_doc_ids(self, question: str, *, limit: int) -> list[str]:
        if self.structured_search_fn is None:
            return []
        doc_by_title = {
            normalize_key(doc.title): doc.doc_id
            for doc in self.list_docs_fn()
            if doc.doc_id and doc.title
        }
        doc_ids: list[str] = []
        for item in self.structured_search_fn(question, limit=limit):
            doc_id = doc_by_title.get(normalize_key(item.title))
            if doc_id:
                doc_ids.append(doc_id)
        return doc_ids

    def _doc_candidates(self, question: str, *, limit: int) -> list[WorldDocInfo]:
        terms = query_terms(question)
        if not terms:
            return []
        scored: list[tuple[int, WorldDocInfo]] = []
        normalized_question = normalize_key(question)
        for doc in self.list_docs_fn():
            haystack = normalize_key(" ".join(filter(None, [doc.folder, doc.title, doc.path_hint or ""])))
            if not haystack:
                continue
            overlap = sum(1 for term in terms if term in haystack)
            if overlap == 0:
                continue
            score = overlap * 20
            if normalize_key(doc.title) in normalized_question or normalized_question in normalize_key(doc.title):
                score += 40
            if normalize_key(doc.folder) in normalized_question:
                score += 10
            scored.append((score, doc))
        scored.sort(key=lambda item: (item[0], normalize_key(item[1].title)), reverse=True)
        return [doc for _, doc in scored[:limit]]

    def _rerank(self, question: str, lists: list[list[dict]], doc_candidates: list[WorldDocInfo]) -> list[dict]:
        candidate_doc_scores = {
            doc.doc_id: 1.0 - (index / max(len(doc_candidates), 1)) * 0.4
            for index, doc in enumerate(doc_candidates)
            if doc.doc_id
        }
        normalized_question = normalize_key(question)
        term_set = set(query_terms(question))
        merged: dict[tuple[object, object, object], dict] = {}
        scores: dict[tuple[object, object, object], float] = {}
        for result_list in lists:
            for rank, hit in enumerate(result_list, start=1):
                key = (hit.get("chunk_id"), hit.get("doc_id"), hit.get("chunk_text"))
                merged[key] = {**hit}
                scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank)
        ranked_hits: list[dict] = []
        for key, hit in merged.items():
            title = normalize_key(str(hit.get("title") or ""))
            path_hint = normalize_key(str(hit.get("path_hint") or ""))
            lexical_bonus = 0.0
            if title and (title in normalized_question or normalized_question in title):
                lexical_bonus += 0.2
            lexical_bonus += 0.03 * sum(1 for term in term_set if term in title or term in path_hint)
            doc_bonus = candidate_doc_scores.get(str(hit.get("doc_id") or ""), 0.0)
            distance = float(hit.get("distance") or 1.0)
            similarity_bonus = max(0.0, 1.0 - min(distance, 1.0)) * 0.15
            hit["hybrid_score"] = round(scores[key] + lexical_bonus + doc_bonus + similarity_bonus, 6)
            ranked_hits.append(hit)
        ranked_hits.sort(
            key=lambda hit: (
                -float(hit.get("hybrid_score") or 0.0),
                float(hit.get("distance") or math.inf),
                normalize_key(str(hit.get("title") or "")),
            )
        )
        return ranked_hits
