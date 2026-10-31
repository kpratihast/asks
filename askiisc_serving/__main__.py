"""python -m askiisc_serving [--host 0.0.0.0] [--port 8100]  (8080 = ingestion TEI, 8000 = vLLM, 11434 = Ollama)"""
import argparse
import logging
import sys

import uvicorn

from .app import create_app


def main():
    ap = argparse.ArgumentParser(prog="askiisc_serving")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)       # one log line per request is ours
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(create_app(), host=a.host, port=a.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
