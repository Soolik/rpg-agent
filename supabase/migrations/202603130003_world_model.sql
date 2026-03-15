create table if not exists world_sessions (
    id bigserial primary key,
    campaign_id text not null,
    session_summary text not null,
    raw_notes text,
    patch_json jsonb not null,
    rag_additions jsonb not null default '[]'::jsonb,
    entity_count integer not null default 0,
    thread_count integer not null default 0,
    source_doc_id text,
    source_title text,
    created_at timestamptz not null default now()
);

create index if not exists world_sessions_campaign_created_idx
    on world_sessions (campaign_id, created_at desc);

create table if not exists world_entities (
    id bigserial primary key,
    campaign_id text not null,
    entity_kind text not null,
    name text not null,
    normalized_name text not null,
    description text not null default '',
    tags jsonb not null default '[]'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    last_session_id bigint references world_sessions(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (campaign_id, entity_kind, normalized_name)
);

create index if not exists world_entities_campaign_kind_idx
    on world_entities (campaign_id, entity_kind, updated_at desc);

create table if not exists world_threads (
    id bigserial primary key,
    campaign_id text not null,
    thread_key text not null,
    thread_id text,
    title text not null,
    normalized_title text not null,
    status text,
    last_change text not null default '',
    metadata jsonb not null default '{}'::jsonb,
    last_session_id bigint references world_sessions(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (campaign_id, thread_key)
);

create index if not exists world_threads_campaign_status_idx
    on world_threads (campaign_id, status, updated_at desc);
