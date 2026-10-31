#!/usr/bin/env python3
"""Clone the existing public index and add reviewed, classified documents without touching the live index."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-data-dir', required=True, type=Path)
    parser.add_argument('--output-data-dir', required=True, type=Path, help='Must be a fresh directory')
    args = parser.parse_args()
    base, target = args.base_data_dir.resolve(), args.output_data_dir.resolve()
    if target.exists():
        raise SystemExit('Output directory already exists; choose a fresh path to preserve the current index.')
    from tools.local import configure
    os.environ['ASKIISC_DATA_DIR'] = str(target)
    configure()
    # Validate all inputs before creating an output directory.
    from askiisc_retrieval.private_documents import DocumentCatalog
    pages = DocumentCatalog().page_inputs()
    from tools.compare_graph import clone_index
    # Imports must follow configuration: config paths and model dimensions are read at import time.
    from askiisc_retrieval.contract import ChunkPayload
    from askiisc_retrieval.snapshot import Snapshot, JsonSnapshotProvider
    from askiisc_retrieval.graph_rules import build_edges, GRAPH_RULES_VERSION
    from askiisc_retrieval.stores import JsonGraphStore
    from askiisc_retrieval.ingest_stub import build_index
    from askiisc_retrieval.chunking import h
    from askiisc_retrieval.cli import make_embedder, make_store
    old = JsonSnapshotProvider(base / 'snapshot.json').active()
    public = [ChunkPayload(**json.loads(line)) for line in (base / 'chunks.jsonl').read_text().splitlines()
              if json.loads(line)['source_url'] not in {'/v1/documents/' + d['id'] for d in DocumentCatalog().documents}]
    target.mkdir(parents=True, mode=0o700)
    clone_index(base / 'qdrant', target / 'qdrant')
    if (base / 'raw').exists():
        shutil.copytree(base / 'raw', target / 'raw')
    emb = make_embedder()
    if emb.model_id != old.embedding_model:
        raise SystemExit('The embedder must match the existing public index.')
    store = make_store()
    try:
        private_report = build_index(pages, emb, store, target / 'graph.json', target / 'snapshot.json', target / 'chunks.jsonl')
        private = [ChunkPayload(**json.loads(line)) for line in (target / 'chunks.jsonl').read_text().splitlines()]
        all_chunks = public + private
        active = frozenset(c.version_id for c in all_chunks if c.version_id in old.active_version_ids or c in private)
        graph = JsonGraphStore(build_edges(all_chunks)); graph.save(target / 'graph.json')
        snapshot = Snapshot('snap-' + h('|'.join(sorted(active)) + emb.model_id,10), emb.model_id,
                            GRAPH_RULES_VERSION, active)
        JsonSnapshotProvider.write(target / 'snapshot.json', snapshot)
        (target / 'chunks.jsonl').write_text(''.join(json.dumps(c.to_dict(),ensure_ascii=False)+'\n' for c in all_chunks))
        report = {'snapshot':snapshot.snapshot_id, 'preserved_chunks':len(public),'private_chunks':len(private),
                  'total_chunks':len(all_chunks),'scopes':{s:sum(c.access_scope==s for c in all_chunks) for s in ['public','student','faculty']},
                  'documents':private_report['pages']}
        (target / 'import-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        store.close()


if __name__ == '__main__': main()
