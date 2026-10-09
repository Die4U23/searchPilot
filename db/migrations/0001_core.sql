CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_migrations (
    filename text PRIMARY KEY,
    sha256 text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS items (
    item_id text PRIMARY KEY,
    title text NOT NULL,
    abstract text NOT NULL,
    category text NOT NULL,
    subcategory text NOT NULL,
    url text NOT NULL,
    first_seen_at timestamptz NULL,
    source text NOT NULL DEFAULT 'mind'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec'))
);

CREATE TABLE IF NOT EXISTS feedback_events (
    event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key uuid NOT NULL UNIQUE,
    request_id text NOT NULL,
    item_id text NOT NULL,
    kind text NOT NULL
        CHECK (kind IN ('impression', 'click', 'like', 'hide')),
    event_at timestamptz NOT NULL,
    user_id text NULL,
    position integer NULL CHECK (position IS NULL OR position >= 0),
    content_hash text NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL DEFAULT 'searchpilot'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec'))
);

CREATE TABLE IF NOT EXISTS model_versions (
    version text PRIMARY KEY,
    kind text NOT NULL,
    artifact_uri text NOT NULL,
    feature_version text NULL,
    sha256 text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    source text NOT NULL DEFAULT 'searchpilot'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec')),
    source_ref jsonb NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id text PRIMARY KEY,
    kind text NOT NULL,
    config_json jsonb NOT NULL,
    code_commit text NULL,
    data_version text NOT NULL,
    protocol_version text NOT NULL,
    source text NOT NULL
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec')),
    source_ref jsonb NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    name text NOT NULL,
    split text NOT NULL,
    segment text NULL,
    value double precision NOT NULL,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    source text NOT NULL
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS metrics_experiment_idx
    ON metrics (experiment_id, name, split);
