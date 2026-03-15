from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .models_v2 import EntityRelationRecord, WorldFactRecord


@dataclass(frozen=True)
class ProjectionFact:
    subject_type: str
    subject_name: str
    predicate: str
    object_value: str
    source_type: str
    source_ref: Optional[str] = None
    confidence: float = 1.0
    metadata: dict | None = None


@dataclass(frozen=True)
class ProjectionRelation:
    source_name: str
    relation_type: str
    target_name: str
    evidence: str
    source_type: str
    source_ref: Optional[str] = None
    confidence: float = 1.0
    metadata: dict | None = None


def normalize_key(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


@dataclass
class WorldFactStore:
    campaign_id: str
    connection_factory: Callable[[], object]

    def replace_projection(
        self,
        *,
        facts: Iterable[ProjectionFact],
        relations: Iterable[ProjectionRelation],
        source_type_prefix: str = "world_model",
    ) -> None:
        fact_rows = list(facts)
        relation_rows = list(relations)
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    delete from entity_relations
                    where campaign_id = %s and source_type like %s
                    """,
                    (self.campaign_id, f"{source_type_prefix}%"),
                )
                cur.execute(
                    """
                    delete from world_facts
                    where campaign_id = %s and source_type like %s
                    """,
                    (self.campaign_id, f"{source_type_prefix}%"),
                )

                for fact in fact_rows:
                    cur.execute(
                        """
                        insert into world_facts (
                            campaign_id,
                            subject_type,
                            subject_name,
                            normalized_subject,
                            predicate,
                            object_value,
                            normalized_object,
                            source_type,
                            source_ref,
                            confidence,
                            metadata
                        )
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            self.campaign_id,
                            fact.subject_type,
                            fact.subject_name,
                            normalize_key(fact.subject_name),
                            fact.predicate,
                            fact.object_value,
                            normalize_key(fact.object_value),
                            fact.source_type,
                            fact.source_ref,
                            fact.confidence,
                            json.dumps(fact.metadata or {}),
                        ),
                    )

                for relation in relation_rows:
                    cur.execute(
                        """
                        insert into entity_relations (
                            campaign_id,
                            source_name,
                            normalized_source,
                            relation_type,
                            target_name,
                            normalized_target,
                            evidence,
                            source_type,
                            source_ref,
                            confidence,
                            metadata
                        )
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            self.campaign_id,
                            relation.source_name,
                            normalize_key(relation.source_name),
                            relation.relation_type,
                            relation.target_name,
                            normalize_key(relation.target_name),
                            relation.evidence,
                            relation.source_type,
                            relation.source_ref,
                            relation.confidence,
                            json.dumps(relation.metadata or {}),
                        ),
                    )
            conn.commit()

    def list_facts(
        self,
        *,
        limit: int = 50,
        subject_name: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> list[WorldFactRecord]:
        query = """
            select id, campaign_id, subject_type, subject_name, predicate, object_value, source_type, source_ref, confidence, updated_at
            from world_facts
            where campaign_id = %s
        """
        params: list[object] = [self.campaign_id]
        if subject_name:
            query += " and normalized_subject = %s"
            params.append(normalize_key(subject_name))
        if predicate:
            query += " and predicate = %s"
            params.append(predicate)
        query += " order by updated_at desc, id desc limit %s"
        params.append(limit)

        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        return [
            WorldFactRecord(
                id=row[0],
                campaign_id=row[1],
                subject_type=row[2],
                subject_name=row[3],
                predicate=row[4],
                object_value=row[5],
                source_type=row[6],
                source_ref=row[7],
                confidence=float(row[8]),
                updated_at=row[9].isoformat(),
            )
            for row in rows
        ]

    def list_relations(
        self,
        *,
        limit: int = 50,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> list[EntityRelationRecord]:
        query = """
            select id, campaign_id, source_name, relation_type, target_name, evidence, source_type, source_ref, confidence, updated_at
            from entity_relations
            where campaign_id = %s
        """
        params: list[object] = [self.campaign_id]
        if source_name:
            query += " and normalized_source = %s"
            params.append(normalize_key(source_name))
        if target_name:
            query += " and normalized_target = %s"
            params.append(normalize_key(target_name))
        if relation_type:
            query += " and relation_type = %s"
            params.append(relation_type)
        query += " order by updated_at desc, id desc limit %s"
        params.append(limit)
        with self.connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        return [
            EntityRelationRecord(
                id=row[0],
                campaign_id=row[1],
                source_name=row[2],
                relation_type=row[3],
                target_name=row[4],
                evidence=row[5],
                source_type=row[6],
                source_ref=row[7],
                confidence=float(row[8]),
                updated_at=row[9].isoformat(),
            )
            for row in rows
        ]


class NullWorldFactStore:
    def replace_projection(
        self,
        *,
        facts: Iterable[ProjectionFact],
        relations: Iterable[ProjectionRelation],
        source_type_prefix: str = "world_model",
    ) -> None:
        return None

    def list_facts(
        self,
        *,
        limit: int = 50,
        subject_name: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> list[WorldFactRecord]:
        return []

    def list_relations(
        self,
        *,
        limit: int = 50,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> list[EntityRelationRecord]:
        return []
