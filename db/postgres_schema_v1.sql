-- Memory Lab PostgreSQL MVP schema.
-- Scope:
--   CP SQLite -> turns -> three-layer evidence pool -> memories.
-- Current MVP uses user_id='local-default-user' but keeps user_id on every
-- table so later account/email/UUID mapping does not require a schema rewrite.

create table if not exists users (
  user_id text primary key,
  email text unique,
  display_name text,
  created_at timestamptz not null default now()
);

create table if not exists sources (
  source_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  source_kind text not null,
  database_path text,
  database_fingerprint text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists turns (
  turn_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  source_id text not null references sources(source_id) on delete cascade,
  session_id text not null,
  project_key text,
  state text not null,
  rule_phase text,
  final_phase text,
  occurred_at timestamptz,
  source_hash text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists evidence_items (
  evidence_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  turn_id text not null references turns(turn_id) on delete cascade,
  layer text not null check (layer in ('core', 'candidate', 'pointer', 'supplemental')),
  source_type text not null,
  kind text not null,
  text_or_summary text not null,
  raw_pointer_id text,
  message_id text,
  part_id text,
  ordinal integer,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- Lightweight Claude Code-style current-state memory tables.
-- Formal memory is intentionally not revisioned and does not maintain a
-- per-memory evidence graph in this MVP.  ``memory_agent_runs`` is optional
-- debug/audit metadata for the Writer call, not the source of truth.
drop table if exists memory_evidence_links;
drop table if exists memory_revisions;

create table if not exists memory_topics (
  topic_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  topic_key text not null,
  title text not null,
  summary text not null,
  topic_type text not null,
  scope jsonb not null default '{}'::jsonb,
  status text not null default 'active',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, topic_key)
);

create table if not exists memories (
  memory_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  topic_id text not null references memory_topics(topic_id) on delete cascade,
  stable_key text not null,
  memory_type text not null,
  scope jsonb not null default '{}'::jsonb,
  summary text not null,
  content text not null,
  status text not null default 'active',
  confidence numeric(4,3),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, stable_key)
);

create table if not exists memory_agent_runs (
  run_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  turn_id text references turns(turn_id) on delete set null,
  agent_role text not null,
  input_json jsonb not null default '{}'::jsonb,
  output_json jsonb not null default '{}'::jsonb,
  status text not null,
  error text,
  created_at timestamptz not null default now()
);

create table if not exists memory_conflict_records (
  conflict_id text primary key,
  user_id text not null references users(user_id) on delete cascade,
  memory_id text not null references memories(memory_id) on delete cascade,
  topic_id text references memory_topics(topic_id) on delete set null,
  turn_id text references turns(turn_id) on delete set null,
  conflict_type text not null,
  old_claim text not null,
  new_claim text not null,
  decision text not null,
  reason text not null,
  evidence_refs jsonb not null default '[]'::jsonb,
  status text not null default 'recorded',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (user_id, memory_id, turn_id, conflict_type, old_claim, new_claim, decision)
);

create index if not exists idx_sources_user_id on sources(user_id);
create index if not exists idx_turns_user_source on turns(user_id, source_id);
create index if not exists idx_turns_user_project on turns(user_id, project_key);
create index if not exists idx_turns_phase on turns(rule_phase, final_phase);
create index if not exists idx_evidence_turn_layer on evidence_items(turn_id, layer);
create index if not exists idx_evidence_user_layer on evidence_items(user_id, layer);
create index if not exists idx_evidence_raw_pointer on evidence_items(raw_pointer_id);
create index if not exists idx_memory_topics_user_type on memory_topics(user_id, topic_type);
create index if not exists idx_memory_topics_scope_gin on memory_topics using gin(scope);
create index if not exists idx_memories_user_type on memories(user_id, memory_type);
create index if not exists idx_memories_topic on memories(topic_id);
create index if not exists idx_memories_scope_gin on memories using gin(scope);
create index if not exists idx_memory_agent_runs_user_turn on memory_agent_runs(user_id, turn_id);
create index if not exists idx_memory_conflict_records_memory on memory_conflict_records(user_id, memory_id);
create index if not exists idx_memory_conflict_records_turn on memory_conflict_records(user_id, turn_id);
create unique index if not exists idx_memory_conflict_records_unique_slot
  on memory_conflict_records(user_id, memory_id, turn_id, conflict_type);

insert into users(user_id, email, display_name)
values ('local-default-user', null, 'Local Default User')
on conflict (user_id) do nothing;
