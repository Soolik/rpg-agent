do $$
declare
    table_name text;
    sequence_name text;
    protected_tables text[] := array[
        'chunks',
        'world_docs',
        'doc_snapshots',
        'proposals',
        'apply_runs',
        'world_sessions',
        'world_entities',
        'world_threads',
        'conversations',
        'conversation_messages',
        'google_drive_oauth_connections',
        'world_facts',
        'entity_relations'
    ];
begin
    foreach table_name in array protected_tables loop
        execute format('alter table public.%I enable row level security', table_name);
        execute format('alter table public.%I force row level security', table_name);
        execute format('revoke all on table public.%I from anon', table_name);
        execute format('revoke all on table public.%I from authenticated', table_name);
    end loop;

    foreach sequence_name in array array[
        'world_docs_id_seq',
        'doc_snapshots_id_seq',
        'proposals_id_seq',
        'apply_runs_id_seq',
        'world_sessions_id_seq',
        'world_entities_id_seq',
        'world_threads_id_seq',
        'conversation_messages_id_seq',
        'world_facts_id_seq',
        'entity_relations_id_seq'
    ] loop
        execute format('revoke all on sequence public.%I from anon', sequence_name);
        execute format('revoke all on sequence public.%I from authenticated', sequence_name);
    end loop;
end
$$;

alter default privileges for role postgres in schema public
    revoke all on tables from anon, authenticated;

alter default privileges for role postgres in schema public
    revoke all on sequences from anon, authenticated;
