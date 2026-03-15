create table if not exists google_drive_oauth_connections (
    campaign_id text not null,
    provider text not null default 'google_drive',
    subject_email text,
    refresh_token_encrypted text not null,
    scopes jsonb not null default '[]'::jsonb,
    token_uri text not null default 'https://oauth2.googleapis.com/token',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (campaign_id, provider)
);
