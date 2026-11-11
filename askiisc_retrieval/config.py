"""Paths and tunables. Everything lives under the project folder so the setup is portable."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path


def _load_env_file(path: str | None) -> None:
    """KEY=VALUE lines (the ingestion team's .env) -> os.environ, never overriding a variable that is
    already set. Lets retrieval/serving use the same credentials file as ingestion without copying
    passwords anywhere: set ASKIISC_ENV_FILE=<path to ingestion .env>."""
    if not path:
        return
    f = Path(path)
    if not f.is_file():
        raise SystemExit(f"ASKIISC_ENV_FILE={path} does not exist")
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file(os.environ.get("ASKIISC_ENV_FILE"))

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ASKIISC_DATA_DIR", ROOT / "data")).expanduser().resolve()
RAW_DIR = DATA / "raw"
QDRANT_PATH = DATA / "qdrant"            # Qdrant local mode: a folder, no server
QDRANT_URL = os.environ.get("QDRANT_URL")   # e.g. http://localhost:6333 -> use the Docker server instead
GRAPH_PATH = DATA / "graph.json"
SNAPSHOT_PATH = DATA / "snapshot.json"
CHUNKS_PATH = DATA / "chunks.jsonl"      # human-readable copy of everything indexed
COLLECTION = "askiisc_chunks"

EMBED_MODEL = os.environ.get("ASKIISC_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.environ.get("ASKIISC_EMBED_DIM", "1024"))
EMBED_MAX_SEQ = 512                      # tokens; caps RAM/time on CPU. Chunks are well under this.
EMBED_BATCH = 8

CHUNK_WORDS = 250
CHUNK_OVERLAP_WORDS = 40

# Reranking (FR-13) and the abstention threshold (FR-11)
# Laptop default: bge-reranker-base (~1.1 GB). Deployment on GPU: BAAI/bge-reranker-v2-m3.
RERANK_MODEL = os.environ.get("ASKIISC_RERANK_MODEL", "BAAI/bge-reranker-base")
# Cost grows linearly with the pool and roughly linearly with length. On a laptop CPU, try
# ASKIISC_RERANK_POOL=6 / ASKIISC_RERANK_MAX_LEN=256 if it is too slow; keep the defaults for reported numbers.
RERANK_POOL = int(os.environ.get("ASKIISC_RERANK_POOL", "10"))
RERANK_MAX_LEN = int(os.environ.get("ASKIISC_RERANK_MAX_LEN", "384"))
# Set a hard floor only after calibrating on the deployed corpus and reranker.
# Without one, reranking changes the order and serving decides whether the evidence answers the question.
_RMIN = os.environ.get("ASKIISC_RERANK_MIN", "none")
RERANK_MIN = None if _RMIN.lower() == "none" else float(_RMIN)
DEVICE = os.environ.get("ASKIISC_DEVICE")    # "cuda" / "cpu"; default: cuda if available

# Backend: "stub" = the laptop stand-in (fetch/ingest in this repo); "ingestion" = the ingestion team's
# stores (Qdrant iisc_policy_passages + TEI + Postgres registry + Neo4j). Credentials come ONLY from env.
BACKEND = os.environ.get("ASKIISC_BACKEND", "stub").lower()
# Each setting also accepts the ingestion .env's own name (QDRANT_COLLECTION, TEI_ENDPOINT, PG_*).
INGEST_COLLECTION = os.environ.get("ASKIISC_INGEST_COLLECTION") or os.environ.get("QDRANT_COLLECTION") \
    or "iisc_policy_passages"
INGEST_DEPARTMENTS = tuple(d.strip() for d in os.environ.get("ASKIISC_DEPARTMENTS", "CDS").split(",") if d.strip())
INGEST_QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
TEI_URL = (os.environ.get("TEI_URL") or os.environ.get("TEI_ENDPOINT", "http://127.0.0.1:8080")
           ).removesuffix("/").removesuffix("/embed").replace("://localhost", "://127.0.0.1")   # Windows: avoid ::1


def _pg_dsn() -> str | None:
    if os.environ.get("ASKIISC_PG_DSN"):
        return os.environ["ASKIISC_PG_DSN"]   # postgresql://user:pass@127.0.0.1:5432/iisc_registry
    e = os.environ
    if e.get("PG_USER") and e.get("PG_PASSWORD"):
        from urllib.parse import quote
        host = e.get("PG_HOST", "127.0.0.1").replace("localhost", "127.0.0.1")
        return (f"postgresql://{quote(e['PG_USER'])}:{quote(e['PG_PASSWORD'])}@{host}:{e.get('PG_PORT', '5432')}"
                f"/{e.get('PG_DB', 'iisc_registry')}")
    return None


PG_DSN = _pg_dsn()
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")       # no graph if unset (vector-only, reported)
GRAPH_TTL_S = float(os.environ.get("ASKIISC_GRAPH_TTL_S", "300"))

# Retrieval
CANDIDATES = 30          # vector candidates fetched before term/dedup logic
TOP_K = 6                # vector/course-code passages returned
PEOPLE_K = int(os.environ.get("ASKIISC_PEOPLE_K", "5"))   # people-pass size; 0 turns the pass off
PEOPLE_DOC_TYPES = ("people",)                              # doc_type of profile / research-interest pages
GRAPH_EXTRA = 3          # max extra passages graph expansion may add (E5 tunes this)
GRAPH_HOPS = 2           # walks only through Course/Person nodes, so 2 hops stays small
GRAPH_TIMEOUT_S = 0.5


def current_term(today: date | None = None) -> str:
    """IISc terms: Aug-Dec -> 'YYYY-AUG', Jan-Apr -> 'YYYY-JAN', May-Jul -> 'YYYY-SUM' (summer)."""
    d = today or date.today()
    return f"{d.year}-{'AUG' if d.month >= 8 else 'SUM' if d.month >= 5 else 'JAN'}"
