create table if not exists world_facts (
    id bigserial primary key,
    campaign_id text not null,
    subject_type text not null,
    subject_name text not null,
    normalized_subject text not null,
    predicate text not null,
    object_value text not null,
    normalized_object text not null,
    source_type text not null,
    source_ref text,
    confidence double precision not null default 1.0,
    metadata jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create index if not exists world_facts_campaign_subject_idx
    on world_facts (campaign_id, normalized_subject, predicate, updated_at desc);

create index if not exists world_facts_campaign_predicate_idx
    on world_facts (campaign_id, predicate, updated_at desc);

create index if not exists world_facts_search_idx
    on world_facts
    using gin (
        to_tsvector(
            'simple',
            coalesce(subject_name, '') || ' ' || coalesce(predicate, '') || ' ' || coalesce(object_value, '')
        )
    );

create table if not exists entity_relations (
    id bigserial primary key,
    campaign_id text not null,
    source_name text not null,
    normalized_source text not null,
    relation_type text not null,
    target_name text not null,
    normalized_target text not null,
    evidence text not null,
    source_type text not null,
    source_ref text,
    confidence double precision not null default 1.0,
    metadata jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create index if not exists entity_relations_campaign_source_idx
    on entity_relations (campaign_id, normalized_source, relation_type, updated_at desc);

create index if not exists entity_relations_campaign_target_idx
    on entity_relations (campaign_id, normalized_target, relation_type, updated_at desc);

create index if not exists entity_relations_search_idx
    on entity_relations
    using gin (
        to_tsvector(
            'simple',
            coalesce(source_name, '') || ' ' || coalesce(relation_type, '') || ' ' || coalesce(target_name, '') || ' ' || coalesce(evidence, '')
        )
    );

create index if not exists chunks_lexical_search_idx
    on chunks
    using gin (
        to_tsvector(
            'simple',
            coalesce(metadata->>'title', '') || ' ' ||
            coalesce(metadata->>'folder', '') || ' ' ||
            coalesce(metadata->>'path_hint', '') || ' ' ||
            coalesce(metadata->>'section_title', '') || ' ' ||
            coalesce(chunk_text, '')
        )
    );
