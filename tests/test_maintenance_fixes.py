"""Regression tests for issues #51-#54 (maintenance tools, serving metrics, date parsing, config)."""
import asyncio
import importlib
import json
import math
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# ---------------------------------------------------------------------------------------- #51
from tools.rebuild_graph import rebuild  # noqa: E402


def test_rebuild_graph_completes_without_nameerror(built, tmp_path):
    """#51: rebuild_graph.py:26 referenced the undefined name `chunks` instead of `active`,
    so every invocation raised NameError after BUILD_INCOMPLETE/qdrant/graph.json were already
    written, leaving a half-built output directory stuck at BUILD_INCOMPLETE."""
    out = tmp_path / "rebuilt"
    report = rebuild(built["dir"], out)
    assert not (out / "BUILD_INCOMPLETE").exists()
    assert (out / "snapshot.json").exists()
    assert (out / "graph.json").exists()
    assert report["chunks"] > 0
    assert report["base_snapshot"]
    assert report["snapshot"] and report["snapshot"] != report["base_snapshot"]


# ---------------------------------------------------------------------------------------- #52
from askiisc_serving.retrieval_client import RemoteRetrieval, SnapshotMismatch  # noqa: E402
from test_serving import FakeModel, FakeRetrieval, make, ask  # noqa: E402


def _health_transport(snapshot_id):
    def handler(request: httpx.Request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"snapshot": {"id": snapshot_id}})
        return httpx.Response(500)
    return httpx.MockTransport(handler)


def test_remote_retrieval_mode_none_rejects_a_stale_pin():
    """#52a: RemoteRetrieval.run returned early on mode=='none' before any pin check, so a
    Baseline-B0 benchmark run with a stale/invalid pinned_snapshot silently ignored the pin
    instead of raising SnapshotMismatch (unlike LocalRetrieval.run, which validates first)."""
    async def run():
        async with httpx.AsyncClient(transport=_health_transport("snap-current")) as client:
            ret = RemoteRetrieval("http://retrieval", client=client)
            with pytest.raises(SnapshotMismatch):
                await ret.run("q", "snap-stale", "none", None, None, scopes=["public"])
            # a matching pin still takes the fast path with no retrieval call, now carrying the
            # real snapshot id instead of None
            hits, trace = await ret.run("q", "snap-current", "none", None, None, scopes=["public"])
            assert hits == [] and trace["snapshot_id"] == "snap-current"
            # no pin at all: unchanged behaviour, no extra round trip needed
            hits, trace = await ret.run("q", None, "none", None, None, scopes=["public"])
            assert hits == [] and trace["snapshot_id"] is None
    asyncio.run(run())


def test_rejected_422_and_403_ask_responses_are_counted_in_metrics():
    """#52b: the mode/empty-question/mode=='none'-without-auth/oversized-question early returns
    in /v1/ask returned HTTP 422/403 without calling metrics.reject, so /metrics undercounted
    those rejections (unlike the 401 path, which principal() already records)."""
    c, _ = make(FakeModel("x [S1]"), FakeRetrieval([]))
    with c:
        assert ask(c, mode="bogus").status_code == 422
        assert ask(c, q="   ").status_code == 422
        assert ask(c, q="x" * 2000).status_code == 422
        body = c.app.state.metrics.prometheus()
    # mode="none" without email-auth enabled: a separate app, since this is a startup-time setting.
    c2, _ = make(FakeModel("x [S1]"), FakeRetrieval([]), email_auth_enabled=False)
    with c2:
        assert ask(c2, mode="none").status_code == 403
        body += c2.app.state.metrics.prometheus()
    for code in ("422", "403"):
        assert f'code="{code}"' in body
    assert "rejected:invalid_mode" in body
    assert "rejected:empty_question" in body
    assert "rejected:mode_none_requires_auth" in body
    assert "rejected:question_too_long" in body


# ---------------------------------------------------------------------------------------- #53
from askiisc_retrieval.scheduling import exam_table_html  # noqa: E402


def test_year_crossing_exam_term_assigns_the_correct_calendar_year():
    """#53: an AUG-season term (e.g. "Oct 2020 -Jan 2021") was keyed to its start year, and that
    single year was then applied to every exam date including January ones, which actually fall
    in the following calendar year. January 4, 2021 is a Monday; January 4, 2020 is a Saturday,
    so the old code rejected this correct PDF with 'PDF exam date and weekday disagree'."""
    layout_pages = [
        "Forenoon: 09:00 AM - 12:00 PM",
        "1. MWF 8:00 Monday January 4 Forenoon",
    ]
    html = exam_table_html(layout_pages, "Exam schedule", "Oct 2020 - Jan 2021", "2020-AUG")
    assert html is not None
    assert "2021-01-04" in html


def test_same_year_aug_season_exam_date_is_unaffected():
    """A December exam in an AUG-season term stays in the term's own start year (no regression
    for the common, non-year-crossing case)."""
    layout_pages = [
        "Forenoon: 09:00 AM - 12:00 PM",
        "1. MWF 8:00 Tuesday December 1 Forenoon",
    ]
    html = exam_table_html(layout_pages, "Exam schedule", "Aug - Dec 2026", "2026-AUG")
    assert html is not None
    assert "2026-12-01" in html


# ---------------------------------------------------------------------------------------- #54
def test_non_finite_rerank_min_is_rejected_at_config_load(monkeypatch):
    """#54: ASKIISC_RERANK_MIN=nan/inf passed float() with no finiteness check; `top >= nan` is
    False for every query, silently abstaining on all answerable questions. Must fail fast."""
    from askiisc_retrieval import config
    for bad in ("nan", "inf", "-inf"):
        monkeypatch.setenv("ASKIISC_RERANK_MIN", bad)
        with pytest.raises(ValueError, match="finite"):
            importlib.reload(config)
    monkeypatch.setenv("ASKIISC_RERANK_MIN", "0.69")
    assert importlib.reload(config).RERANK_MIN == 0.69
    monkeypatch.delenv("ASKIISC_RERANK_MIN")
    importlib.reload(config)
