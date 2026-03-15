create table if not exists conversations (
    id text primary key,
    campaign_id text not null,
    title text not null default '',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    last_message_at timestamptz
);

create index if not exists conversations_campaign_updated_idx
    on conversations (campaign_id, updated_at desc);

create table if not exists conversation_messages (
    id bigserial primary key,
    conversation_id text not null references conversations(id) on delete cascade,
    campaign_id text not null,
    role text not null,
    kind text,
    artifact_type text,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists conversation_messages_conversation_created_idx
    on conversation_messages (conversation_id, created_at asc);

create index if not exists conversation_messages_campaign_created_idx
    on conversation_messages (campaign_id, created_at desc);
