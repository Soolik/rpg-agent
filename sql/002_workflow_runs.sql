alter table proposals
    add column if not exists request_json jsonb not null default '{}'::jsonb;

create table if not exists apply_runs (
    id bigserial primary key,
    campaign_id text not null,
    proposal_id bigint references proposals(id) on delete set null,
    approved boolean not null,
    approved_by text,
    ok boolean not null,
    request_json jsonb not null,
    response_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists apply_runs_campaign_created_idx
    on apply_runs (campaign_id, created_at desc);

create index if not exists apply_runs_proposal_idx
    on apply_runs (proposal_id);
