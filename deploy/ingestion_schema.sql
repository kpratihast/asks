-- PostgreSQL schema for the ingestion registry (iisc_registry), RECONSTRUCTED from the columns that
-- Smit's pipeline.py and cleanup_url.py read and write (2026-10-24). The shared files contain no
-- table-creation script; if the ingestion team has its own, use theirs instead of this one.
--
--   psql -h 127.0.0.1 -U iisc_admin -d iisc_registry -f deploy/ingestion_schema.sql
--
-- Safe to run twice (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS document_versions (
    version_id          UUID PRIMARY KEY,
    source_url          TEXT        NOT NULL,
    content_hash        TEXT        NOT NULL,
    source_type         TEXT        NOT NULL DEFAULT 'html',      -- html | pdf
    raw_snapshot_s3_key TEXT,                                     -- MinIO object key
    department          TEXT,
    programme           TEXT,
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    is_superseded       BOOLEAN     NOT NULL DEFAULT FALSE,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS document_versions_url ON document_versions (source_url);
CREATE INDEX IF NOT EXISTS document_versions_active ON document_versions (is_active) WHERE is_active;

-- cleanup_url.py deletes document_versions rows and relies on the chunks going with them.
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        UUID PRIMARY KEY,
    version_id      UUID    NOT NULL REFERENCES document_versions (version_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    text_content    TEXT    NOT NULL,
    token_count     INTEGER,
    qdrant_point_id UUID    NOT NULL
);
CREATE INDEX IF NOT EXISTS document_chunks_version ON document_chunks (version_id);

-- One row per completed ingestion run (FR-4 manifest).
CREATE TABLE IF NOT EXISTS active_snapshots (
    snapshot_id        SERIAL PRIMARY KEY,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    graph_version      TEXT,
    active_version_ids TEXT[]      NOT NULL,
    total_chunks       INTEGER,
    notes              TEXT
);
