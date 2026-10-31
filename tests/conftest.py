import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from askiisc_retrieval.ingest_stub import PageInput, build_index      # noqa: E402
from askiisc_retrieval.retriever import Retriever                      # noqa: E402
from askiisc_retrieval.snapshot import JsonSnapshotProvider            # noqa: E402
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore, QdrantStore   # noqa: E402

PAGES = Path(__file__).parent / "pages"
SCHEDULE = "https://cds.iisc.ac.in/courses/schedule/"
DESCRIPTIONS = "https://cds.iisc.ac.in/courses/descriptions/"
MTECH = "https://cds.iisc.ac.in/courses/m-tech-course-requirements/"


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("index")
    pages = [
        PageInput(SCHEDULE, (PAGES / "schedule.html").read_text(), "schedule", "public"),
        PageInput(DESCRIPTIONS, (PAGES / "descriptions.html").read_text(), "catalogue", "public"),
        PageInput(MTECH, (PAGES / "mtech.html").read_text(), "policy", "public"),
    ]
    fx = ROOT / "fixtures"
    for f in yaml.safe_load((fx / "fixtures.yaml").read_text())["fixtures"]:
        pages.append(PageInput(f["url"], (fx / f["file"]).read_text(), f["doc_type"], f["scope"],
                               active=f.get("active", True), version_tag=f.get("version_tag", "")))
    emb = HashEmbedder()
    store = QdrantStore("test", emb.dim, path=d / "qdrant")
    report = build_index(pages, emb, store, d / "graph.json", d / "snapshot.json", d / "chunks.jsonl")
    yield {"dir": d, "emb": emb, "store": store, "report": report}
    store.close()


@pytest.fixture
def retriever(built):
    graph = JsonGraphStore.load(built["dir"] / "graph.json")
    return Retriever(built["emb"], built["store"], JsonSnapshotProvider(built["dir"] / "snapshot.json"),
                     graph, today_term="2026-AUG")
