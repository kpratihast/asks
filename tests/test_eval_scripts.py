"""#44: the E4, E6 and stale-version evaluation scripts keep working (plumbing check with the test-only hash
embedder; the reported numbers come from runs with the configured embedder)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
ROOT = Path(__file__).resolve().parents[1]


def run(script, out, *extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ASKIISC_", "QDRANT_URL", "VLLM_URL"))}
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / script), "--hash", "--out", str(out), *extra],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


def test_stale_eval_returns_no_superseded_passage_and_is_not_vacuous(tmp_path):
    run("eval_stale.py", tmp_path)
    s = json.loads((tmp_path / "stale_summary.json").read_text())
    assert s["vector"]["questions"] == 12 and s["config"]["superseded_versions"] == 3
    assert s["vector"]["superseded_passages_returned"] == 0 and s["full"]["superseded_passages_returned"] == 0
    assert s["control_unfiltered"]["superseded_passages_in_top_k"] > 0      # the old answers really are indexed


def test_e6_generation_and_graph_outage(tmp_path):
    run("eval_e6.py", tmp_path)
    s = json.loads((tmp_path / "e6_summary.json").read_text())
    gen = s["generation_outage"]
    assert gen["requests"] == gen["http_200"] == 24 and gen["fabricated_answers"] == 0 and gen["expected_fallback"] > 0
    assert gen["expected_fallback"] + gen["abstained_before_model_call"] == 24
    g = s["graph_outage"]
    assert g["retriever"]["degraded_with_passages"] == g["retriever"]["queries"] == 12
    assert g["gateway"]["http_200_degraded_with_excerpts"] == g["gateway"]["requests"] == 12


def test_e4_modified_document_becomes_queryable_without_duplicate_requests(tmp_path):
    run("eval_e4.py", tmp_path)
    s = json.loads((tmp_path / "e4_summary.json").read_text())
    assert s["2_unchanged_refresh"]["new_registry_rows"] == 0
    assert s["2_unchanged_refresh"]["stats"].get("new", 0) == s["2_unchanged_refresh"]["stats"].get("changed", 0) == 0
    m = s["3_modified_refresh"]
    assert [(r["version"], r["superseded"]) for r in m["registry"]["mtech_versions"]] == [(1, True), (2, False)]
    assert m["registry"]["mtech_v1_object_kept_byte_for_byte"] and m["snapshot_after"] != m["snapshot_before"]
    assert all(step["site"]["duplicate_requests"] == 0
               for step in (s["1_initial"], s["2_unchanged_refresh"], m))
    assert s["4_queryable"]["passed"]
