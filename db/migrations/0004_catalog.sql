CREATE TABLE IF NOT EXISTS queries (
    query_id text PRIMARY KEY,
    query_text text NOT NULL,
    normalized_text text NOT NULL,
    query_type text NOT NULL
        CHECK (query_type IN (
            'exact_entity',
            'synonym',
            'multi_condition',
            'misspelling_or_abbrev',
            'no_answer',
            'long_tail_popular_distractor'
        )),
    split text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    source text NOT NULL DEFAULT 'searchpilot'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec'))
);

CREATE TABLE IF NOT EXISTS relevance_labels (
    query_id text NOT NULL REFERENCES queries(query_id),
    item_id text NOT NULL,
    grade smallint NOT NULL CHECK (grade BETWEEN 0 AND 3),
    annotator text NOT NULL,
    labeled_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL DEFAULT 'searchpilot'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec')),
    PRIMARY KEY (query_id, item_id, annotator)
);

CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id text PRIMARY KEY,
    question text NOT NULL,
    kind text NOT NULL,
    expected_tools jsonb NOT NULL DEFAULT '[]'::jsonb,
    source text NOT NULL DEFAULT 'searchpilot'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec'))
);

CREATE TABLE IF NOT EXISTS bids (
    item_id text PRIMARY KEY,
    bid double precision NOT NULL CHECK (bid > 0),
    distribution_version text NOT NULL,
    source text NOT NULL DEFAULT 'synthetic'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec'))
);
