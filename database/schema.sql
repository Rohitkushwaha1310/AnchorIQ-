-- Supabase / PostgreSQL schema for AnchorIQ
-- Paste into the Supabase SQL editor.

CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target TEXT,
    problem_type TEXT,
    model_card JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS charts (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    cloudinary_url TEXT,
    cloudinary_public_id TEXT,
    fact_card JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS insights (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'rules'
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    inputs JSONB DEFAULT '{}'::jsonb,
    output JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_charts_run_id ON charts(run_id);
CREATE INDEX IF NOT EXISTS idx_insights_run_id ON insights(run_id);
CREATE INDEX IF NOT EXISTS idx_chat_run_id ON chat_messages(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);
