CREATE TABLE IF NOT EXISTS users (
    user_id text PRIMARY KEY,
    created_at timestamptz NULL,
    segment text NOT NULL
        CHECK (segment IN ('history_len_0', 'history_len_gt_0')),
    source text NOT NULL DEFAULT 'mind'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec'))
);

CREATE TABLE IF NOT EXISTS impressions (
    impression_id text NOT NULL,
    user_id text NOT NULL,
    item_id text NOT NULL,
    position integer NULL CHECK (position IS NULL OR position >= 0),
    clicked smallint NOT NULL CHECK (clicked IN (0, 1)),
    model_version text NULL,
    shown_at timestamptz NOT NULL,
    split text NOT NULL CHECK (split IN ('train', 'dev')),
    source text NOT NULL DEFAULT 'mind'
        CHECK (source IN ('searchpilot', 'mind', 'synthetic', 'evorec')),
    PRIMARY KEY (impression_id, item_id)
);

CREATE INDEX IF NOT EXISTS impressions_user_shown_idx
    ON impressions (user_id, shown_at);
