"""Local SQLite store for ingestion state and normalized events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
    source_id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_success_at TEXT,
    baseline_cutoff_ms INTEGER
);

CREATE TABLE IF NOT EXISTS ingestion_run (
    run_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    scanned_parts INTEGER NOT NULL DEFAULT 0,
    new_parts INTEGER NOT NULL DEFAULT 0,
    user_events INTEGER NOT NULL DEFAULT 0,
    isolated_user_events INTEGER NOT NULL DEFAULT 0,
    reclassified_late_events INTEGER NOT NULL DEFAULT 0,
    reclassified_child_events INTEGER NOT NULL DEFAULT 0,
    filtered_parts INTEGER NOT NULL DEFAULT 0,
    duplicate_parts INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    details_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source_id) REFERENCES source(source_id)
);

CREATE TABLE IF NOT EXISTS processed_part (
    source_id TEXT NOT NULL,
    part_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    disposition TEXT NOT NULL,
    reason TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (source_id, part_id),
    FOREIGN KEY (run_id) REFERENCES ingestion_run(run_id)
);

CREATE TABLE IF NOT EXISTS event (
    event_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    parent_session_id TEXT,
    message_id TEXT NOT NULL,
    part_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    directory TEXT,
    session_title TEXT,
    project_id TEXT,
    session_version TEXT,
    agent TEXT,
    timestamp INTEGER,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    eligibility_status TEXT NOT NULL DEFAULT 'eligible',
    ingested_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE (source_id, part_id)
);

CREATE INDEX IF NOT EXISTS event_timestamp_idx ON event(timestamp);
CREATE INDEX IF NOT EXISTS event_session_idx ON event(session_id, timestamp);
CREATE INDEX IF NOT EXISTS event_content_hash_idx ON event(content_hash);

CREATE TABLE IF NOT EXISTS project_context (
    project_key TEXT PRIMARY KEY,
    directory TEXT NOT NULL,
    display_name TEXT NOT NULL,
    identity_source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_context (
    source_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    project_key TEXT NOT NULL,
    directory TEXT NOT NULL,
    context_source TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (source_id, session_id),
    FOREIGN KEY (project_key) REFERENCES project_context(project_key)
);

CREATE TABLE IF NOT EXISTS event_annotation (
    event_id TEXT PRIMARY KEY,
    experiment_case_id TEXT,
    semantic_content TEXT NOT NULL,
    project_key TEXT NOT NULL,
    annotation_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES event(event_id),
    FOREIGN KEY (project_key) REFERENCES project_context(project_key)
);

CREATE TABLE IF NOT EXISTS instruction_coverage_run (
    coverage_run_id TEXT PRIMARY KEY,
    resolver_version TEXT NOT NULL,
    assessment_level TEXT NOT NULL,
    started_at TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    completed_at TEXT,
    completed_at_ms INTEGER,
    status TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS instruction_scan (
    scan_id TEXT PRIMARY KEY,
    coverage_run_id TEXT,
    project_key TEXT NOT NULL,
    session_directory TEXT NOT NULL,
    resolver_version TEXT NOT NULL,
    resolver_confidence TEXT NOT NULL,
    resolver_config_json TEXT NOT NULL,
    resolver_context_hash TEXT,
    scanned_at TEXT NOT NULL,
    started_at_ms INTEGER,
    completed_at_ms INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    UNIQUE(project_key, resolver_version, scanned_at),
    FOREIGN KEY (coverage_run_id) REFERENCES instruction_coverage_run(coverage_run_id),
    FOREIGN KEY (project_key) REFERENCES project_context(project_key)
);

CREATE TABLE IF NOT EXISTS instruction_scan_source (
    scan_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    session_version TEXT NOT NULL,
    PRIMARY KEY (scan_id, source_id, session_version),
    FOREIGN KEY (scan_id) REFERENCES instruction_scan(scan_id),
    FOREIGN KEY (source_id) REFERENCES source(source_id)
);

CREATE TABLE IF NOT EXISTS instruction_document_revision (
    revision_id TEXT PRIMARY KEY,
    canonical_path TEXT NOT NULL,
    origin TEXT NOT NULL,
    effective_scope_kind TEXT NOT NULL,
    scope_root TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_ns INTEGER,
    modified_ns INTEGER,
    captured_at TEXT NOT NULL,
    read_status TEXT NOT NULL,
    read_error TEXT,
    UNIQUE(canonical_path, content_hash)
);

CREATE TABLE IF NOT EXISTS instruction_scan_item (
    scan_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    discovered_by TEXT NOT NULL,
    cp_would_load INTEGER NOT NULL,
    memory_applicability TEXT NOT NULL,
    applicability_reason TEXT NOT NULL,
    observed_at_ms INTEGER,
    observed_size_bytes INTEGER,
    observed_created_ns INTEGER,
    observed_modified_ns INTEGER,
    observed_read_status TEXT,
    observed_read_error TEXT,
    PRIMARY KEY (scan_id, revision_id, discovered_by),
    FOREIGN KEY (scan_id) REFERENCES instruction_scan(scan_id),
    FOREIGN KEY (revision_id) REFERENCES instruction_document_revision(revision_id)
);

CREATE TABLE IF NOT EXISTS instruction_clause (
    clause_id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    heading_path TEXT,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    exact_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    normalization_version TEXT,
    clause_fingerprint TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(revision_id, ordinal),
    FOREIGN KEY (revision_id) REFERENCES instruction_document_revision(revision_id)
);

CREATE TABLE IF NOT EXISTS event_instruction_hint (
    event_id TEXT PRIMARY KEY,
    assessment_level TEXT NOT NULL,
    covered_now TEXT NOT NULL,
    covered_at_event TEXT NOT NULL,
    event_scope_hint TEXT NOT NULL,
    best_clause_id TEXT,
    content_score REAL NOT NULL,
    scope_relation TEXT NOT NULL,
    reason TEXT NOT NULL,
    analyzed_at TEXT NOT NULL,
    latest_coverage_run_id TEXT,
    temporal_evidence TEXT,
    FOREIGN KEY (event_id) REFERENCES event(event_id),
    FOREIGN KEY (best_clause_id) REFERENCES instruction_clause(clause_id),
    FOREIGN KEY (latest_coverage_run_id) REFERENCES instruction_coverage_run(coverage_run_id)
);

CREATE TABLE IF NOT EXISTS event_instruction_assessment (
    assessment_id TEXT PRIMARY KEY,
    coverage_run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    assessment_level TEXT NOT NULL,
    covered_now TEXT NOT NULL,
    temporal_evidence TEXT NOT NULL,
    event_scope_hint TEXT NOT NULL,
    best_clause_id TEXT,
    content_score REAL NOT NULL,
    scope_relation TEXT NOT NULL,
    reason TEXT NOT NULL,
    analyzed_at TEXT NOT NULL,
    UNIQUE (coverage_run_id, event_id, assessment_level),
    FOREIGN KEY (coverage_run_id) REFERENCES instruction_coverage_run(coverage_run_id),
    FOREIGN KEY (event_id) REFERENCES event(event_id),
    FOREIGN KEY (best_clause_id) REFERENCES instruction_clause(clause_id)
);

CREATE TABLE IF NOT EXISTS admission_run (
    run_id TEXT PRIMARY KEY,
    extractor_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    decision_count INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    error TEXT
);

CREATE TABLE IF NOT EXISTS admission_decision (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    event_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    exact_quote TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    family TEXT NOT NULL,
    memory_type TEXT,
    stable_key_suggestion TEXT,
    scope_json TEXT NOT NULL,
    activation_triggers_json TEXT NOT NULL,
    disposition TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    instruction_coverage_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    UNIQUE (run_id, candidate_key),
    FOREIGN KEY (run_id) REFERENCES admission_run(run_id),
    FOREIGN KEY (event_id) REFERENCES event(event_id)
);

CREATE TABLE IF NOT EXISTS memory_proposal (
    proposal_id TEXT PRIMARY KEY,
    proposal_fingerprint TEXT NOT NULL UNIQUE,
    admission_run_id TEXT NOT NULL,
    admission_snapshot_hash TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    stable_key TEXT NOT NULL,
    semantic_value TEXT NOT NULL,
    raw_scope_json TEXT NOT NULL,
    canonical_scope_json TEXT NOT NULL,
    scope_schema_version TEXT NOT NULL,
    scope_fingerprint TEXT NOT NULL,
    relation TEXT NOT NULL,
    disposition TEXT NOT NULL,
    is_correction INTEGER NOT NULL,
    expected_memory_id TEXT,
    expected_current_revision_id TEXT,
    expected_current_semantic_value TEXT,
    expected_memory_state_version INTEGER,
    expected_open_conflicts_hash TEXT NOT NULL,
    application_payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (expected_memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (expected_current_revision_id) REFERENCES memory_revision(revision_id)
);

CREATE TABLE IF NOT EXISTS candidate_review (
    review_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('approve', 'reject')),
    reviewer TEXT NOT NULL,
    note TEXT,
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY (proposal_id) REFERENCES memory_proposal(proposal_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id)
);

CREATE TABLE IF NOT EXISTS auto_memory_run (
    run_id TEXT PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    admission_run_id TEXT,
    admission_snapshot_hash TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    writer_prompt_version TEXT NOT NULL,
    reviewer_prompt_version TEXT NOT NULL,
    writer_schema_version TEXT NOT NULL,
    reviewer_schema_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'complete', 'complete_with_rejections', 'failed')
    ),
    candidate_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    revised_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    replay_count INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    formal_state_before_json TEXT NOT NULL,
    formal_state_after_json TEXT,
    error_code TEXT,
    error_message TEXT,
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id)
);

CREATE TABLE IF NOT EXISTS memory_writer_result (
    writer_result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    admission_run_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    plan_hash TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('valid', 'invalid', 'replayed')),
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN ('writer_call', 'cache_replay', 'local_block')
    ),
    plan_json TEXT,
    error_code TEXT,
    error_message TEXT,
    replayed_from_writer_result_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, decision_id),
    FOREIGN KEY (run_id) REFERENCES auto_memory_run(run_id),
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (replayed_from_writer_result_id)
        REFERENCES memory_writer_result(writer_result_id)
);

CREATE TABLE IF NOT EXISTS memory_reviewer_result (
    reviewer_result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    writer_result_id TEXT NOT NULL,
    admission_run_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    reviewed_plan_hash TEXT NOT NULL,
    final_plan_hash TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('valid', 'invalid', 'replayed')),
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN ('reviewer_call', 'cache_replay', 'local_block')
    ),
    outcome TEXT CHECK (outcome IN ('accept', 'revise', 'reject')),
    decision_json TEXT,
    final_plan_json TEXT,
    error_code TEXT,
    error_message TEXT,
    replayed_from_reviewer_result_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, decision_id),
    FOREIGN KEY (run_id) REFERENCES auto_memory_run(run_id),
    FOREIGN KEY (writer_result_id) REFERENCES memory_writer_result(writer_result_id),
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (replayed_from_reviewer_result_id)
        REFERENCES memory_reviewer_result(reviewer_result_id)
);

CREATE TABLE IF NOT EXISTS memory_rollback (
    rollback_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    from_revision_id TEXT NOT NULL,
    target_revision_id TEXT NOT NULL,
    new_revision_id TEXT NOT NULL UNIQUE,
    expected_state_version INTEGER NOT NULL,
    requested_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (from_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (target_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (new_revision_id) REFERENCES memory_revision(revision_id)
);

CREATE TABLE IF NOT EXISTS llm_shadow_run (
    shadow_run_id TEXT PRIMARY KEY,
    shadow_version TEXT NOT NULL,
    admission_run_id TEXT NOT NULL,
    admission_snapshot_hash TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_schema_version TEXT NOT NULL,
    output_schema_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    replay_count INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    formal_state_before_json TEXT NOT NULL,
    formal_state_after_json TEXT,
    error TEXT,
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id)
);

CREATE TABLE IF NOT EXISTS llm_judgment (
    judgment_id TEXT PRIMARY KEY,
    shadow_run_id TEXT NOT NULL,
    admission_run_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_schema_version TEXT NOT NULL,
    output_schema_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('valid', 'invalid', 'blocked', 'replayed')),
    execution_mode TEXT CHECK (execution_mode IN ('judge_call', 'cache_replay', 'local_block')),
    result_json TEXT,
    should_store INTEGER,
    memory_type TEXT,
    stable_key TEXT,
    scope_json TEXT,
    semantic_value TEXT,
    relation TEXT,
    target_kind TEXT,
    target_id TEXT,
    confidence REAL,
    error_code TEXT,
    error_message TEXT,
    replayed_from_judgment_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (shadow_run_id, decision_id),
    FOREIGN KEY (shadow_run_id) REFERENCES llm_shadow_run(shadow_run_id),
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (replayed_from_judgment_id) REFERENCES llm_judgment(judgment_id)
);

CREATE TABLE IF NOT EXISTS memory (
    memory_id TEXT PRIMARY KEY,
    stable_key TEXT NOT NULL,
    scope_fingerprint TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    family TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'retired')),
    current_revision_id TEXT,
    state_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (stable_key, scope_fingerprint),
    FOREIGN KEY (current_revision_id) REFERENCES memory_revision(revision_id)
);

CREATE TABLE IF NOT EXISTS memory_revision (
    revision_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    semantic_value TEXT NOT NULL,
    statement TEXT NOT NULL,
    activation_triggers_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'superseded')),
    supersedes_revision_id TEXT,
    source_decision_id TEXT NOT NULL,
    source_proposal_id TEXT,
    review_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (memory_id, revision_number),
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (supersedes_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (source_decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (source_proposal_id) REFERENCES memory_proposal(proposal_id),
    FOREIGN KEY (review_id) REFERENCES candidate_review(review_id)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    revision_id TEXT,
    conflict_id TEXT,
    candidate_key TEXT NOT NULL UNIQUE,
    decision_id TEXT NOT NULL,
    proposal_id TEXT,
    review_id TEXT,
    reviewer TEXT,
    event_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    part_id TEXT NOT NULL,
    project_key TEXT NOT NULL,
    source_project_id TEXT,
    exact_quote TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    occurred_at INTEGER,
    evidence_role TEXT NOT NULL CHECK (evidence_role IN ('supporting', 'conflicting', 'correction')),
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (conflict_id) REFERENCES conflict(conflict_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (proposal_id) REFERENCES memory_proposal(proposal_id),
    FOREIGN KEY (review_id) REFERENCES candidate_review(review_id),
    FOREIGN KEY (event_id) REFERENCES event(event_id)
);

CREATE TABLE IF NOT EXISTS conflict (
    conflict_id TEXT PRIMARY KEY,
    conflict_fingerprint TEXT NOT NULL UNIQUE,
    memory_id TEXT NOT NULL,
    current_revision_id TEXT NOT NULL,
    incoming_candidate_key TEXT NOT NULL UNIQUE,
    incoming_decision_id TEXT NOT NULL,
    incoming_proposal_id TEXT,
    review_id TEXT,
    reviewer TEXT,
    incoming_semantic_value TEXT NOT NULL,
    incoming_statement TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    resolution_revision_id TEXT,
    resolution_proposal_id TEXT,
    resolution_review_id TEXT,
    resolution_reviewer TEXT,
    resolution_reason TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (current_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (incoming_decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (incoming_proposal_id) REFERENCES memory_proposal(proposal_id),
    FOREIGN KEY (review_id) REFERENCES candidate_review(review_id),
    FOREIGN KEY (resolution_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (resolution_proposal_id) REFERENCES memory_proposal(proposal_id),
    FOREIGN KEY (resolution_review_id) REFERENCES candidate_review(review_id)
);

CREATE TABLE IF NOT EXISTS consolidation_run (
    run_id TEXT PRIMARY KEY,
    consolidator_version TEXT NOT NULL,
    admission_run_id TEXT,
    admission_snapshot_hash TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    FOREIGN KEY (admission_run_id) REFERENCES admission_run(run_id)
);

CREATE TABLE IF NOT EXISTS consolidation_action (
    action_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    proposal_id TEXT,
    review_id TEXT,
    reviewer TEXT,
    action TEXT NOT NULL,
    memory_id TEXT,
    before_revision_id TEXT,
    after_revision_id TEXT,
    conflict_id TEXT,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (run_id, candidate_key),
    FOREIGN KEY (run_id) REFERENCES consolidation_run(run_id),
    FOREIGN KEY (decision_id) REFERENCES admission_decision(decision_id),
    FOREIGN KEY (proposal_id) REFERENCES memory_proposal(proposal_id),
    FOREIGN KEY (review_id) REFERENCES candidate_review(review_id),
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (before_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (after_revision_id) REFERENCES memory_revision(revision_id),
    FOREIGN KEY (conflict_id) REFERENCES conflict(conflict_id)
);

CREATE TRIGGER IF NOT EXISTS memory_proposal_no_update
BEFORE UPDATE ON memory_proposal
BEGIN
    SELECT RAISE(ABORT, 'memory_proposal is immutable');
END;

CREATE TRIGGER IF NOT EXISTS memory_proposal_no_delete
BEFORE DELETE ON memory_proposal
BEGIN
    SELECT RAISE(ABORT, 'memory_proposal is immutable');
END;

CREATE TRIGGER IF NOT EXISTS candidate_review_no_update
BEFORE UPDATE ON candidate_review
BEGIN
    SELECT RAISE(ABORT, 'candidate_review is append-only');
END;

CREATE TRIGGER IF NOT EXISTS candidate_review_no_delete
BEFORE DELETE ON candidate_review
BEGIN
    SELECT RAISE(ABORT, 'candidate_review is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_writer_result_no_update
BEFORE UPDATE ON memory_writer_result
BEGIN
    SELECT RAISE(ABORT, 'memory_writer_result is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_writer_result_no_delete
BEFORE DELETE ON memory_writer_result
BEGIN
    SELECT RAISE(ABORT, 'memory_writer_result is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_reviewer_result_no_update
BEFORE UPDATE ON memory_reviewer_result
BEGIN
    SELECT RAISE(ABORT, 'memory_reviewer_result is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_reviewer_result_no_delete
BEFORE DELETE ON memory_reviewer_result
BEGIN
    SELECT RAISE(ABORT, 'memory_reviewer_result is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_rollback_no_update
BEFORE UPDATE ON memory_rollback
BEGIN
    SELECT RAISE(ABORT, 'memory_rollback is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_rollback_no_delete
BEFORE DELETE ON memory_rollback
BEGIN
    SELECT RAISE(ABORT, 'memory_rollback is append-only');
END;

CREATE TRIGGER IF NOT EXISTS llm_judgment_no_update
BEFORE UPDATE ON llm_judgment
BEGIN
    SELECT RAISE(ABORT, 'llm_judgment is append-only');
END;

CREATE TRIGGER IF NOT EXISTS llm_judgment_no_delete
BEFORE DELETE ON llm_judgment
BEGIN
    SELECT RAISE(ABORT, 'llm_judgment is append-only');
END;

CREATE TABLE IF NOT EXISTS turn_detection_run (
    run_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    settle_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    candidate_turns INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    FOREIGN KEY (source_id) REFERENCES source(source_id)
);

CREATE TABLE IF NOT EXISTS conversation_turn (
    turn_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_message_id TEXT NOT NULL,
    primary_event_id TEXT NOT NULL,
    user_timestamp INTEGER,
    current_state TEXT NOT NULL,
    first_detected_at TEXT NOT NULL,
    last_detected_at TEXT NOT NULL,
    last_detection_run_id TEXT NOT NULL,
    last_structural_signature TEXT NOT NULL,
    assistant_count INTEGER NOT NULL DEFAULT 0,
    part_count INTEGER NOT NULL DEFAULT 0,
    active_tool_count INTEGER NOT NULL DEFAULT 0,
    terminal_assistant_id TEXT,
    terminal_finish TEXT,
    terminal_error INTEGER NOT NULL DEFAULT 0,
    next_user_message_id TEXT,
    UNIQUE (source_id, session_id, user_message_id),
    FOREIGN KEY (source_id) REFERENCES source(source_id),
    FOREIGN KEY (primary_event_id) REFERENCES event(event_id),
    FOREIGN KEY (last_detection_run_id) REFERENCES turn_detection_run(run_id)
);

CREATE TABLE IF NOT EXISTS turn_event (
    turn_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    PRIMARY KEY (turn_id, event_id),
    FOREIGN KEY (turn_id) REFERENCES conversation_turn(turn_id),
    FOREIGN KEY (event_id) REFERENCES event(event_id)
);

CREATE TABLE IF NOT EXISTS turn_observation (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    sample_index INTEGER NOT NULL CHECK (sample_index IN (1, 2)),
    observed_at TEXT NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    structural_signature TEXT NOT NULL,
    source_user_present INTEGER NOT NULL,
    assistant_count INTEGER NOT NULL,
    part_count INTEGER NOT NULL,
    active_tool_count INTEGER NOT NULL,
    terminal_assistant_id TEXT,
    terminal_finish TEXT,
    terminal_completed INTEGER NOT NULL,
    terminal_error INTEGER NOT NULL,
    next_user_message_id TEXT,
    preliminary_state TEXT NOT NULL,
    UNIQUE (run_id, turn_id, sample_index),
    FOREIGN KEY (run_id) REFERENCES turn_detection_run(run_id),
    FOREIGN KEY (turn_id) REFERENCES conversation_turn(turn_id)
);

CREATE TABLE IF NOT EXISTS turn_detection_result (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    state TEXT NOT NULL,
    signature_stable INTEGER NOT NULL,
    sample_1_signature TEXT NOT NULL,
    sample_2_signature TEXT NOT NULL,
    reason TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    UNIQUE (run_id, turn_id),
    FOREIGN KEY (run_id) REFERENCES turn_detection_run(run_id),
    FOREIGN KEY (turn_id) REFERENCES conversation_turn(turn_id)
);

CREATE TABLE IF NOT EXISTS episode_extraction_run (
    run_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    settle_ms INTEGER NOT NULL DEFAULT 1000,
    eligible_turns INTEGER NOT NULL DEFAULT 0,
    created_episodes INTEGER NOT NULL DEFAULT 0,
    existing_episodes INTEGER NOT NULL DEFAULT 0,
    unstable_turns INTEGER NOT NULL DEFAULT 0,
    skipped_turns INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    FOREIGN KEY (source_id) REFERENCES source(source_id)
);

CREATE TABLE IF NOT EXISTS trace_episode (
    episode_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    extraction_run_id TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    source_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    primary_event_id TEXT NOT NULL,
    project_key TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    trace_signature TEXT NOT NULL,
    turn_state TEXT NOT NULL CHECK (turn_state IN ('completed', 'failed')),
    evidence_class TEXT NOT NULL
        CHECK (evidence_class IN ('success', 'failure', 'recovered', 'unverified')),
    domain_verification_outcome TEXT NOT NULL DEFAULT 'unverified'
        CHECK (domain_verification_outcome IN (
            'verified_success', 'verified_failure', 'ambiguous', 'unverified'
        )),
    semantic_status TEXT NOT NULL CHECK (semantic_status = 'not_evaluated'),
    assistant_count INTEGER NOT NULL,
    tool_event_count INTEGER NOT NULL,
    positive_event_count INTEGER NOT NULL,
    negative_event_count INTEGER NOT NULL,
    unverified_event_count INTEGER NOT NULL,
    recovered_tool_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (extraction_run_id) REFERENCES episode_extraction_run(run_id),
    FOREIGN KEY (source_id) REFERENCES source(source_id),
    FOREIGN KEY (turn_id) REFERENCES conversation_turn(turn_id),
    FOREIGN KEY (primary_event_id) REFERENCES event(event_id)
);

CREATE TABLE IF NOT EXISTS trace_episode_event (
    episode_event_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    source_assistant_message_id TEXT NOT NULL,
    source_part_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    state_status TEXT NOT NULL,
    started_at_ms INTEGER,
    ended_at_ms INTEGER,
    has_error_payload INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    input_hash TEXT,
    input_length INTEGER,
    output_hash TEXT,
    output_length INTEGER,
    error_hash TEXT,
    error_length INTEGER,
    evidence_verdict TEXT NOT NULL
        CHECK (evidence_verdict IN ('success', 'failure', 'unverified')),
    event_signature TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (episode_id, ordinal),
    UNIQUE (episode_id, source_part_id),
    FOREIGN KEY (episode_id) REFERENCES trace_episode(episode_id)
);

CREATE TABLE IF NOT EXISTS run_episode_result (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    episode_id TEXT,
    disposition TEXT NOT NULL
        CHECK (disposition IN ('created', 'existing', 'unstable', 'skipped')),
    sample_1_trace_signature TEXT NOT NULL,
    sample_2_trace_signature TEXT NOT NULL,
    domain_verification_outcome TEXT NOT NULL
        CHECK (domain_verification_outcome IN (
            'verified_success', 'verified_failure', 'ambiguous', 'unverified'
        )),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, turn_id),
    FOREIGN KEY (run_id) REFERENCES episode_extraction_run(run_id),
    FOREIGN KEY (turn_id) REFERENCES conversation_turn(turn_id),
    FOREIGN KEY (episode_id) REFERENCES trace_episode(episode_id)
);

CREATE TRIGGER IF NOT EXISTS trace_episode_no_update
BEFORE UPDATE ON trace_episode
BEGIN
    SELECT RAISE(ABORT, 'trace_episode is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trace_episode_no_delete
BEFORE DELETE ON trace_episode
BEGIN
    SELECT RAISE(ABORT, 'trace_episode is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trace_episode_event_no_update
BEFORE UPDATE ON trace_episode_event
BEGIN
    SELECT RAISE(ABORT, 'trace_episode_event is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trace_episode_event_no_delete
BEFORE DELETE ON trace_episode_event
BEGIN
    SELECT RAISE(ABORT, 'trace_episode_event is immutable');
END;

CREATE TRIGGER IF NOT EXISTS run_episode_result_no_update
BEFORE UPDATE ON run_episode_result
BEGIN
    SELECT RAISE(ABORT, 'run_episode_result is immutable');
END;

CREATE TRIGGER IF NOT EXISTS run_episode_result_no_delete
BEFORE DELETE ON run_episode_result
BEGIN
    SELECT RAISE(ABORT, 'run_episode_result is immutable');
END;

CREATE INDEX IF NOT EXISTS instruction_scan_project_idx
    ON instruction_scan(project_key, scanned_at);
CREATE INDEX IF NOT EXISTS instruction_revision_path_idx
    ON instruction_document_revision(canonical_path, captured_at);
CREATE INDEX IF NOT EXISTS instruction_clause_revision_idx
    ON instruction_clause(revision_id, ordinal);
CREATE INDEX IF NOT EXISTS instruction_assessment_event_idx
    ON event_instruction_assessment(event_id, analyzed_at);
CREATE INDEX IF NOT EXISTS admission_decision_candidate_idx
    ON admission_decision(candidate_key, decided_at);
CREATE INDEX IF NOT EXISTS candidate_review_decision_idx
    ON candidate_review(decision_id, reviewed_at);
CREATE INDEX IF NOT EXISTS llm_shadow_admission_idx
    ON llm_shadow_run(admission_run_id, started_at);
CREATE INDEX IF NOT EXISTS llm_judgment_request_idx
    ON llm_judgment(request_hash, model, prompt_version, created_at);
CREATE INDEX IF NOT EXISTS llm_judgment_candidate_idx
    ON llm_judgment(candidate_key, created_at);
CREATE INDEX IF NOT EXISTS memory_proposal_decision_idx
    ON memory_proposal(decision_id, created_at);
CREATE INDEX IF NOT EXISTS memory_proposal_candidate_idx
    ON memory_proposal(candidate_key, created_at);
CREATE INDEX IF NOT EXISTS memory_identity_idx
    ON memory(stable_key, scope_fingerprint);
CREATE INDEX IF NOT EXISTS memory_revision_memory_idx
    ON memory_revision(memory_id, revision_number);
CREATE UNIQUE INDEX IF NOT EXISTS memory_revision_one_current_idx
    ON memory_revision(memory_id) WHERE status = 'current';
CREATE INDEX IF NOT EXISTS evidence_memory_idx
    ON evidence(memory_id, revision_id);
CREATE INDEX IF NOT EXISTS conflict_memory_status_idx
    ON conflict(memory_id, status);
CREATE INDEX IF NOT EXISTS consolidation_action_candidate_idx
    ON consolidation_action(candidate_key, created_at);
CREATE INDEX IF NOT EXISTS conversation_turn_session_idx
    ON conversation_turn(source_id, session_id, user_timestamp);
CREATE INDEX IF NOT EXISTS turn_observation_turn_idx
    ON turn_observation(turn_id, observed_at_ms);
CREATE INDEX IF NOT EXISTS turn_detection_result_turn_idx
    ON turn_detection_result(turn_id, decided_at);
CREATE INDEX IF NOT EXISTS episode_extraction_run_source_idx
    ON episode_extraction_run(source_id, started_at);
CREATE INDEX IF NOT EXISTS trace_episode_turn_idx
    ON trace_episode(source_id, turn_id, created_at);
CREATE INDEX IF NOT EXISTS trace_episode_project_class_idx
    ON trace_episode(project_key, evidence_class, created_at);
CREATE INDEX IF NOT EXISTS trace_episode_event_episode_idx
    ON trace_episode_event(episode_id, ordinal);
CREATE INDEX IF NOT EXISTS trace_episode_event_tool_idx
    ON trace_episode_event(tool_name, evidence_verdict);
CREATE UNIQUE INDEX IF NOT EXISTS trace_episode_v2_current_idx
    ON trace_episode(source_id, turn_id, extractor_version)
    WHERE extractor_version = 'cp-trace-episode-v2';
CREATE UNIQUE INDEX IF NOT EXISTS trace_episode_v3_current_idx
    ON trace_episode(source_id, turn_id, extractor_version)
    WHERE extractor_version = 'cp-trace-episode-v3';
CREATE INDEX IF NOT EXISTS run_episode_result_run_idx
    ON run_episode_result(run_id, disposition);
CREATE INDEX IF NOT EXISTS run_episode_result_episode_idx
    ON run_episode_result(episode_id, created_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    event_table_existed = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'event'"
    ).fetchone() is not None
    event_columns_before = {
        row[1] for row in connection.execute("PRAGMA table_info(event)")
    }
    migrate_legacy_event_eligibility = (
        event_table_existed and "eligibility_status" not in event_columns_before
    )
    connection.executescript(SCHEMA)
    # The lab database is intentionally append-only across MVP iterations. These
    # lightweight migrations preserve previous scans while adding audit fields.
    for table, column, declaration in (
        ("source", "baseline_cutoff_ms", "INTEGER"),
        ("ingestion_run", "isolated_user_events", "INTEGER NOT NULL DEFAULT 0"),
        ("ingestion_run", "reclassified_late_events", "INTEGER NOT NULL DEFAULT 0"),
        ("ingestion_run", "reclassified_child_events", "INTEGER NOT NULL DEFAULT 0"),
        ("event", "eligibility_status", "TEXT NOT NULL DEFAULT 'eligible'"),
        ("instruction_scan", "coverage_run_id", "TEXT"),
        ("instruction_scan", "resolver_context_hash", "TEXT"),
        ("instruction_scan", "started_at_ms", "INTEGER"),
        ("instruction_scan", "completed_at_ms", "INTEGER"),
        ("instruction_scan_item", "observed_at_ms", "INTEGER"),
        ("instruction_scan_item", "observed_size_bytes", "INTEGER"),
        ("instruction_scan_item", "observed_created_ns", "INTEGER"),
        ("instruction_scan_item", "observed_modified_ns", "INTEGER"),
        ("instruction_scan_item", "observed_read_status", "TEXT"),
        ("instruction_scan_item", "observed_read_error", "TEXT"),
        ("instruction_clause", "normalization_version", "TEXT"),
        ("instruction_clause", "clause_fingerprint", "TEXT"),
        ("event_instruction_hint", "latest_coverage_run_id", "TEXT"),
        ("event_instruction_hint", "temporal_evidence", "TEXT"),
        ("evidence", "conflict_id", "TEXT"),
        ("conflict", "conflict_fingerprint", "TEXT"),
        ("memory", "family", "TEXT"),
        ("evidence", "ordinal", "INTEGER"),
        ("evidence", "start_offset", "INTEGER"),
        ("evidence", "end_offset", "INTEGER"),
        ("evidence", "occurred_at", "INTEGER"),
        ("candidate_review", "proposal_id", "TEXT"),
        ("memory_revision", "source_proposal_id", "TEXT"),
        ("memory_revision", "review_id", "TEXT"),
        ("evidence", "proposal_id", "TEXT"),
        ("evidence", "review_id", "TEXT"),
        ("evidence", "reviewer", "TEXT"),
        ("consolidation_action", "proposal_id", "TEXT"),
        ("consolidation_action", "review_id", "TEXT"),
        ("consolidation_action", "reviewer", "TEXT"),
        ("memory_proposal", "raw_scope_json", "TEXT"),
        ("memory_proposal", "scope_schema_version", "TEXT"),
        ("memory_proposal", "admission_snapshot_hash", "TEXT"),
        ("memory_proposal", "expected_memory_state_version", "INTEGER"),
        ("memory_proposal", "expected_open_conflicts_hash", "TEXT"),
        ("memory_proposal", "application_payload_json", "TEXT"),
        ("memory", "state_version", "INTEGER NOT NULL DEFAULT 0"),
        ("conflict", "incoming_proposal_id", "TEXT"),
        ("conflict", "review_id", "TEXT"),
        ("conflict", "reviewer", "TEXT"),
        ("conflict", "resolution_proposal_id", "TEXT"),
        ("conflict", "resolution_review_id", "TEXT"),
        ("conflict", "resolution_reviewer", "TEXT"),
        ("consolidation_run", "admission_run_id", "TEXT"),
        ("consolidation_run", "admission_snapshot_hash", "TEXT"),
        ("llm_shadow_run", "admission_snapshot_hash", "TEXT"),
        ("llm_judgment", "execution_mode", "TEXT"),
        ("episode_extraction_run", "settle_ms", "INTEGER NOT NULL DEFAULT 1000"),
        ("episode_extraction_run", "unstable_turns", "INTEGER NOT NULL DEFAULT 0"),
        ("episode_extraction_run", "skipped_turns", "INTEGER NOT NULL DEFAULT 0"),
        (
            "trace_episode",
            "domain_verification_outcome",
            "TEXT NOT NULL DEFAULT 'unverified'",
        ),
        (
            "trace_episode_event",
            "has_error_payload",
            "INTEGER NOT NULL DEFAULT 0",
        ),
    ):
        _ensure_column(connection, table, column, declaration)
    legacy_fingerprints_exist = connection.execute(
        """
        SELECT 1
        FROM trace_episode_event AS ee
        JOIN trace_episode AS ep ON ep.episode_id = ee.episode_id
        WHERE ep.extractor_version = 'cp-trace-episode-v1'
          AND (
              ee.input_hash IS NOT NULL OR ee.input_length IS NOT NULL
              OR ee.output_hash IS NOT NULL OR ee.output_length IS NOT NULL
              OR ee.error_hash IS NOT NULL OR ee.error_length IS NOT NULL
          )
        LIMIT 1
        """
    ).fetchone()
    if legacy_fingerprints_exist is not None:
        # v1 briefly persisted unsalted payload hashes and exact lengths. Clear
        # those sensitive derivatives in one controlled migration while
        # preserving episode/event IDs and restoring the immutability guard in
        # the same transaction.
        connection.execute("DROP TRIGGER IF EXISTS trace_episode_event_no_update")
        try:
            connection.execute(
                """
                UPDATE trace_episode_event
                SET input_hash = NULL, input_length = NULL,
                    output_hash = NULL, output_length = NULL,
                    error_hash = NULL, error_length = NULL
                WHERE episode_id IN (
                    SELECT episode_id FROM trace_episode
                    WHERE extractor_version = 'cp-trace-episode-v1'
                )
                """
            )
        finally:
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trace_episode_event_no_update
                BEFORE UPDATE ON trace_episode_event
                BEGIN
                    SELECT RAISE(ABORT, 'trace_episode_event is immutable');
                END
                """
            )
    if migrate_legacy_event_eligibility:
        connection.execute(
            """
            UPDATE event
            SET eligibility_status = 'child_or_fork_session_user_message'
            WHERE parent_session_id IS NOT NULL AND TRIM(parent_session_id) <> ''
            """
        )
        connection.execute(
            """
            UPDATE event
            SET eligibility_status = 'late_arrival_pre_baseline'
            WHERE eligibility_status = 'eligible'
              AND timestamp IS NOT NULL
              AND EXISTS(
                  SELECT 1 FROM source AS s
                  WHERE s.source_id = event.source_id
                    AND s.baseline_cutoff_ms IS NOT NULL
                    AND event.timestamp < s.baseline_cutoff_ms
              )
            """
        )
        connection.execute(
            """
            UPDATE event
            SET eligibility_status = 'legacy_unverified_eligibility'
            WHERE eligibility_status = 'eligible'
              AND NOT EXISTS(
                  SELECT 1 FROM source AS s
                  WHERE s.source_id = event.source_id
                    AND s.baseline_cutoff_ms IS NOT NULL
                    AND event.timestamp IS NOT NULL
                    AND event.timestamp >= s.baseline_cutoff_ms
              )
            """
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS instruction_scan_context_idx "
        "ON instruction_scan(project_key, resolver_context_hash, completed_at_ms)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS conflict_fingerprint_idx "
        "ON conflict(conflict_fingerprint) WHERE conflict_fingerprint IS NOT NULL"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS memory_revision_one_current_idx "
        "ON memory_revision(memory_id) WHERE status = 'current'"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS trace_episode_domain_outcome_idx "
        "ON trace_episode(project_key, domain_verification_outcome, created_at)"
    )
    return connection


def register_source(connection: sqlite3.Connection, identifier: str, path: Path) -> None:
    connection.execute(
        """
        INSERT INTO source(source_id, source_kind, source_path, created_at)
        VALUES (?, 'sqlite', ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET source_path = excluded.source_path
        """,
        (identifier, str(path.resolve()), now_iso()),
    )
    # Databases created before baseline_cutoff_ms was introduced already have a
    # durable baseline run. Recover its original boundary instead of using the
    # migration time, otherwise old rows written late could become eligible.
    row = connection.execute(
        """
        SELECT MIN(started_at)
        FROM ingestion_run
        WHERE source_id = ? AND mode = 'baseline'
        """,
        (identifier,),
    ).fetchone()
    if row and row[0]:
        connection.execute(
            """
            UPDATE source
            SET baseline_cutoff_ms = COALESCE(baseline_cutoff_ms, ?)
            WHERE source_id = ?
            """,
            (_iso_to_epoch_ms(row[0]), identifier),
        )


def _iso_to_epoch_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def begin_run(connection: sqlite3.Connection, source_identifier: str, mode: str) -> str:
    run_id = uuid4().hex
    started_at = now_iso()
    connection.execute(
        "INSERT INTO ingestion_run(run_id, source_id, mode, started_at) VALUES (?, ?, ?, ?)",
        (run_id, source_identifier, mode, started_at),
    )
    if mode == "baseline":
        connection.execute(
            """
            UPDATE source
            SET baseline_cutoff_ms = COALESCE(baseline_cutoff_ms, ?)
            WHERE source_id = ?
            """,
            (_iso_to_epoch_ms(started_at), source_identifier),
        )
    return run_id


def baseline_cutoff_ms(
    connection: sqlite3.Connection, source_identifier: str
) -> int | None:
    row = connection.execute(
        "SELECT baseline_cutoff_ms FROM source WHERE source_id = ?",
        (source_identifier,),
    ).fetchone()
    return row[0] if row else None


def mark_prebaseline_events(
    connection: sqlite3.Connection, source_identifier: str, cutoff_ms: int | None
) -> int:
    """Quarantine legacy events that should never have entered the eligible set."""
    if cutoff_ms is None:
        return 0
    cursor = connection.execute(
        """
        UPDATE event
        SET eligibility_status = 'late_arrival_pre_baseline'
        WHERE source_id = ?
          AND eligibility_status = 'eligible'
          AND timestamp IS NOT NULL
          AND timestamp < ?
        """,
        (source_identifier, cutoff_ms),
    )
    return cursor.rowcount


def mark_child_session_events(
    connection: sqlite3.Connection, source_identifier: str
) -> int:
    """Quarantine child/fork session user events from the U1 memory channel."""
    cursor = connection.execute(
        """
        UPDATE event
        SET eligibility_status = 'child_or_fork_session_user_message'
        WHERE source_id = ?
          AND eligibility_status = 'eligible'
          AND parent_session_id IS NOT NULL
          AND TRIM(parent_session_id) <> ''
        """,
        (source_identifier,),
    )
    return cursor.rowcount


def already_processed(connection: sqlite3.Connection, source_identifier: str, part_id: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM processed_part WHERE source_id = ? AND part_id = ?",
        (source_identifier, part_id),
    ).fetchone() is not None


def record_processed(
    connection: sqlite3.Connection,
    source_identifier: str,
    part_id: str,
    run_id: str,
    disposition: str,
    reason: str,
) -> None:
    connection.execute(
        """
        INSERT INTO processed_part(
            source_id, part_id, run_id, disposition, reason, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_identifier, part_id, run_id, disposition, reason, now_iso()),
    )


def record_event(
    connection: sqlite3.Connection,
    event: dict,
    eligibility_status: str = "eligible",
) -> None:
    connection.execute(
        """
        INSERT INTO event(
            event_id, source_id, session_id, parent_session_id, message_id,
            part_id, event_type, directory, session_title, project_id,
            session_version, agent, timestamp, content, content_hash,
            source_kind, eligibility_status, ingested_at, raw_json
        ) VALUES (
            :event_id, :source_id, :session_id, :parent_session_id, :message_id,
            :part_id, :event_type, :directory, :session_title, :project_id,
            :session_version, :agent, :timestamp, :content, :content_hash,
            :source_kind, :eligibility_status, :ingested_at, :raw_json
        )
        """,
        {
            **event,
            "eligibility_status": eligibility_status,
            "ingested_at": now_iso(),
            "raw_json": json.dumps(event, ensure_ascii=False, sort_keys=True),
        },
    )


def finish_run(connection: sqlite3.Connection, run_id: str, report: dict) -> None:
    connection.execute(
        """
        UPDATE ingestion_run
        SET completed_at = ?, scanned_parts = ?, new_parts = ?, user_events = ?,
            isolated_user_events = ?, reclassified_late_events = ?,
            reclassified_child_events = ?,
            filtered_parts = ?, duplicate_parts = ?, error_count = ?, details_json = ?
        WHERE run_id = ?
        """,
        (
            now_iso(),
            report["scanned_parts"],
            report["new_parts"],
            report["user_events"],
            report["isolated_user_events"],
            report["reclassified_late_events"],
            report["reclassified_child_events"],
            report["filtered_parts"],
            report["duplicate_parts"],
            report["error_count"],
            json.dumps(report, ensure_ascii=False, sort_keys=True),
            run_id,
        ),
    )
    connection.execute(
        "UPDATE source SET last_success_at = ? WHERE source_id = ?",
        (now_iso(), report["source_id"]),
    )
