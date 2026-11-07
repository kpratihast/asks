"""Run public IISc chat against an existing OpenAI-compatible model server, without Docker."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def configure():
    # The local index is deliberately separate from the BGE-M3/ingestion index.
    defaults = {
        "ASKIISC_BACKEND": "stub", "ASKIISC_DATA_DIR": str(ROOT / "data" / "local"),
        "ASKIISC_EMBED_MODEL": "BAAI/bge-small-en-v1.5", "ASKIISC_EMBED_DIM": "384",
        "ASKIISC_RERANKER": "none", "ASKIISC_RERANK_MIN": "none",
        "VLLM_URL": "http://127.0.0.1:8000", "SERVED_MODEL": "auto", "MODEL_PROVIDER_PREFERENCES": {},
        "CHAT_TEMPLATE_KWARGS": {},
        "AUTH_COOKIE_SECURE": "0",
        "EMAIL_AUTH_ENABLED": "1", "AUTH_PROFESSOR_TEST_EMAILS": "",
        "MODEL_REASONING": {},
        "MAX_MODEL_LEN": "4096", "MAX_OUTPUT_TOKENS": "512", "PUBLIC_CHAT": "1",
        "GLOBAL_CONCURRENCY": "1", "TENANT_CONCURRENCY": "1", "TENANT_QUEUE": "4",
        "QUEUE_WAIT_S": "120", "GEN_TIMEOUT_S": "180", "TOKENIZERS_PARALLELISM": "false",
    }
    # The checked-in local profile selects the current server; explicit env overrides win.
    config_path = Path(os.environ.get("ASKIISC_LOCAL_CONFIG", str(ROOT / "config" / "local.json"))).expanduser()
    if config_path.is_file():
        profile = json.loads(config_path.read_text())
        if not isinstance(profile, dict):
            raise SystemExit(f"Local config must be a JSON object: {config_path}")
        unknown = set(profile) - set(defaults)
        if unknown:
            raise SystemExit(f"Unknown local config keys: {', '.join(sorted(unknown))}")
        defaults.update(profile)
    elif "ASKIISC_LOCAL_CONFIG" in os.environ:
        raise SystemExit(f"Local config not found: {config_path}")
    # Optional private machine configuration; never stored in the project.
    email_config = Path(os.environ.get("ASKIISC_EMAIL_CONFIG", str(Path.home() / ".config" / "askiisc" / "email.json"))).expanduser()
    email_enabled = str(os.environ.get("EMAIL_AUTH_ENABLED", defaults["EMAIL_AUTH_ENABLED"])).strip().lower() in {"1", "true", "yes", "on"}
    if email_enabled and email_config.is_file():
        email_settings = json.loads(email_config.read_text())
        allowed = {"SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM"}
        if not isinstance(email_settings, dict) or set(email_settings) - allowed:
            raise SystemExit("Invalid private email configuration")
        for k, v in email_settings.items():
            os.environ.setdefault(k, str(v))
    for k, v in defaults.items():
        os.environ.setdefault(k, v if isinstance(v, str) else json.dumps(v))


def prepare(refresh=False, rebuild=False):
    import httpx
    from askiisc_retrieval import config
    from askiisc_retrieval.cli import make_embedder, make_store
    from askiisc_retrieval.fetch import UA, load_sources, raw_name
    from askiisc_retrieval.ingest_stub import PageInput, build_index

    if config.BACKEND != "stub" or config.QDRANT_URL:
        raise SystemExit("Local prepare requires ASKIISC_BACKEND=stub and QDRANT_URL unset. "
                         "Use the ingestion pipeline to populate server-backed stores.")
    if config.SNAPSHOT_PATH.exists() and not refresh and not rebuild:
        print("A local index exists. Use --refresh to fetch and rebuild it while serving is stopped.")
        return
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages, manifest = [], {}
    # Fixed approved source list only: no synthetic fixtures, no unbounded crawl.
    with httpx.Client(timeout=45, follow_redirects=True, headers={"User-Agent": UA}, trust_env=False) as client:
        for pg in load_sources()["pages"]:
            target = config.RAW_DIR / raw_name(pg["url"])
            try:
                if refresh or not target.exists():
                    response = client.get(pg["url"])
                    response.raise_for_status()
                    target.write_bytes(response.content)
                    manifest[pg["url"]] = {"file": target.name, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    time.sleep(0.5)
                pages.append(PageInput(pg["url"], target.read_bytes(), pg["doc_type"], pg["scope"]))
                print("Source:", pg["url"], flush=True)
            except httpx.HTTPError as exc:
                print(f"Unavailable: {pg['url']} ({type(exc).__name__})", file=sys.stderr)
    if not pages:
        raise SystemExit("No configured source pages could be fetched; check network access and retry.")
    old_manifest = config.RAW_DIR / "index.json"
    previous = json.loads(old_manifest.read_text()) if old_manifest.exists() else {}
    old_manifest.write_text(json.dumps({**previous, **manifest}, indent=2))
    # Include the full cached crawl and bounded admissions snapshot on rebuild;
    # rebuilding only the small seed list would silently discard most local knowledge.
    from askiisc_retrieval.ingest_stub import load_page_inputs
    pages = load_page_inputs(include_fixtures=False)
    if (config.DATA / 'admissions' / 'sources.json').is_file():
        print('Admissions sources: using the verified cache. Use tools/import_admissions.py --fetch for a fresh bounded snapshot.')
    emb = make_embedder()
    store = make_store(reset=False)
    try:
        if store.count():
            # Acquire Qdrant's local lock before replacing anything; a running server owns it.
            store.reset()
        report = build_index(pages, emb, store, config.GRAPH_PATH, config.SNAPSHOT_PATH, config.CHUNKS_PATH)
    finally:
        store.close()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Fetch configured source pages and build the local index")
    parser.add_argument("--refresh", action="store_true", help="Refetch/rebuild (stop serving first)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild cached pages and classified private documents (stop serving first)")
    parser.add_argument("--auth", choices=("on", "off"), help="Enable OTP/roles, or disable sign-in and serve public content only")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if args.auth is not None:
        os.environ["EMAIL_AUTH_ENABLED"] = "1" if args.auth == "on" else "0"
    configure()
    from askiisc_retrieval import config
    if args.prepare or args.refresh or args.rebuild:
        prepare(args.refresh, args.rebuild)
        return
    if not config.SNAPSHOT_PATH.exists() and os.environ["ASKIISC_BACKEND"] == "stub":
        raise SystemExit("No local index yet. Run ./run-local.sh --prepare first.")
    from askiisc_serving.app import create_app
    import uvicorn
    print(f"Ask IISc: http://127.0.0.1:{args.port} | model: {os.environ['VLLM_URL']}", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
