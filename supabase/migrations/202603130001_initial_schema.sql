create extension if not exists vector;

create table if not exists chunks (
    id uuid primary key,
    campaign_id text not null,
    doc_id text not null,
    doc_type text not null,
    chunk_text text not null,
    embedding vector(3072) not null,
    metadata jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create index if not exists chunks_campaign_doc_idx
    on chunks (campaign_id, doc_id);

create index if not exists chunks_campaign_doc_type_idx
    on chunks (campaign_id, doc_type);

create table if not exists world_docs (
    id bigserial primary key,
    campaign_id text not null,
    doc_id text not null,
    folder text not null,
    title text not null,
    entity_type text not null,
    content_hash text,
    last_synced_at timestamptz,
    metadata jsonb not null default '{}'::jsonb,
    unique (campaign_id, doc_id)
);

create table if not exists doc_snapshots (
    id bigserial primary key,
    campaign_id text not null,
    doc_id text not null,
    content_hash text not null,
    content_text text not null,
    created_at timestamptz not null default now()
);

create index if not exists doc_snapshots_campaign_doc_idx
    on doc_snapshots (campaign_id, doc_id, created_at desc);

create table if not exists proposals (
    id bigserial primary key,
    campaign_id text not null,
    summary text not null,
    user_goal text not null,
    proposal_json jsonb not null,
    approved boolean not null default false,
    approved_by text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

