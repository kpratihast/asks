"""Serving tier: auth, admission control, prompt/budget, vLLM streaming, abstention, fallback, SSE.
The model endpoint is a fake OpenAI-compatible stream (httpx.MockTransport); retrieval is a stub."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from askiisc_serving.admission import Admission, Rejected
from askiisc_serving.app import create_app
from askiisc_serving.conversation import HELP_REPLY
from askiisc_serving.auth import KeyStore
from askiisc_serving.generator import ABSTAIN, SYSTEM, Generator, verify
from askiisc_serving.retrieval_client import SnapshotMismatch, academic_year, hits_from_result
from askiisc_serving.settings import Settings

GUEST, STUDENT = "guest-key-0123456789abcdef", "student-key-0123456789abcdef"
KEYS = KeyStore({GUEST: {"tenant": "applicants", "role": "guest"},
                 STUDENT: {"tenant": "students", "role": "student"}})
H = lambda k=GUEST: {"Authorization": f"Bearer {k}"}

HIT = {"access_scope": "public", "marker": "S1", "title": "Course Schedule", "section": "Aug - Dec 2026 | DS 252",
       "url": "https://cds.iisc.ac.in/courses/schedule/", "version_id": "v1", "chunk_ids": ["c1"],
       "academic_term": "2026-AUG", "academic_year": "2026-2027", "programme": "not specified",
       "location": "Aug - Dec 2026 | DS 252 (https://cds.iisc.ac.in/courses/schedule/)",
       "freshness": {"status": "current", "snapshot": "snap-1"}, "instructor": "Yogesh Simmhan",
       "text": "Course ID: DS 252; Credits: 3:1; Instructor: Yogesh Simmhan", "excerpt": "Course ID: DS 252 ...",
       "rerank_score": 0.99, "via": "vector"}


class FakeRetrieval:
    def __init__(self, hits=None, sufficient=True, raise_=None):
        self.hits, self.sufficient, self.raise_ = hits if hits is not None else [HIT], sufficient, raise_
        self.calls = []

    async def run(self, question, pinned, mode, programme, year, *, scopes):
        self.calls.append({"q": question, "mode": mode, "scopes": scopes, "pinned": pinned})
        if self.raise_:
            raise self.raise_
        if mode == "none":
            return [], {"mode": mode, "snapshot_id": "snap-1", "evidence_sufficient": None, "timings_ms": {}}
        return list(self.hits), {"mode": mode, "snapshot_id": "snap-1", "evidence_sufficient": self.sufficient,
                                 "rerank_top": 0.99, "degraded": False, "timings_ms": {"total": 12.0}}

    def source(self, chunk_id, scopes):
        return {"chunk_id": chunk_id, "text": "full text", "access_scope": "public"} if chunk_id == "c1" and "public" in scopes else None

    async def ready(self):
        return True


def sse_body(answer: str, usage=(120, 14)):
    out = []
    for piece in answer.split(" "):
        out.append("data: " + json.dumps({"choices": [{"delta": {"content": piece + " "}, "finish_reason": None}]}))
    out.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    out.append("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}))
    out.append("data: [DONE]")
    return ("\n\n".join(out) + "\n\n").encode()


class FakeModel:
    """Answers in order from `answers`; an Exception instance means the endpoint fails."""

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def handler(self, request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        self.requests.append(json.loads(request.content))
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        return httpx.Response(200, content=sse_body(a), headers={"content-type": "text/event-stream"})


def make(model: FakeModel, retrieval=None, **kw):
    s = Settings(vllm_url="http://model", served_model="test-model", **kw)
    gen = Generator(s, client=httpx.AsyncClient(transport=httpx.MockTransport(model.handler)))
    retrieval = retrieval or FakeRetrieval()
    return TestClient(create_app(s, retrieval=retrieval, generator=gen, keys=KEYS)), retrieval


def ask(c, q="Who teaches DS 252 this semester?", key=GUEST, **body):
    return c.post("/v1/ask", json={"question": q, **body}, headers=H(key))


# ---------------------------------------------------------------- auth & scope
def test_requests_without_a_valid_key_get_401():
    with make(FakeModel("x [S1]"))[0] as c:
        assert c.post("/v1/ask", json={"question": "hi"}).status_code == 401
        r = c.post("/v1/ask", json={"question": "hi"}, headers={"Authorization": "Bearer wrong-key-000000000000"})
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_scope_comes_from_the_key_not_the_request():
    c, ret = make(FakeModel("DS 252 is taught by Yogesh Simmhan [S1]."))
    with c:
        ask(c, key=GUEST)
        ask(c, key=STUDENT)
    assert ret.calls[0]["scopes"] == ["public"] and ret.calls[1]["scopes"] == ["public", "student"]


# ---------------------------------------------------------------- answering
def test_grounded_answer_with_citations_usage_and_timings():
    model = FakeModel("DS 252 is taught by Yogesh Simmhan [S1].")
    c, _ = make(model)
    with c:
        out = ask(c).json()
    assert out["status"] == "answered" and "[S1]" in out["answer"] and out["grounded"]
    assert out["citations"][0]["marker"] == "S1" and out["citations"][0]["used"]
    assert out["usage"] == {"prompt_tokens": 120, "completion_tokens": 14}
    t = out["trace"]["timings_ms"]
    assert t["ttft_ms"] is not None and t["total_ms"] >= t["retrieval_ms"]
    req = model.requests[0]
    assert req["stream"] and req["temperature"] == 0 and req["stream_options"] == {"include_usage": True}
    assert req["messages"][0] == {"role": "system", "content": SYSTEM}
    assert "[S1] Course Schedule\nProgramme: not specified; year: 2026-2027" in req["messages"][1]["content"]
    assert "instructor: Yogesh Simmhan" in req["messages"][1]["content"]


def test_insufficient_evidence_abstains_without_calling_the_model():
    model = FakeModel("should not be called [S1]")
    c, _ = make(model, retrieval=FakeRetrieval(sufficient=False))
    with c:
        out = ask(c, q="What is the GATE cutoff?").json()
    assert out["status"] == "abstained" and out["reason"] == "insufficient_evidence" and out["answer"] == ABSTAIN + "\n\n" + HELP_REPLY
    assert model.requests == []


def test_model_abstention_is_reported_as_abstained():
    c, _ = make(FakeModel(ABSTAIN))
    with c:
        out = ask(c, q="What is the exact stipend?").json()
    assert out["status"] == "abstained" and out["reason"] == "model_abstained" and out["citations"] == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("position", ["after", "same_paragraph"])
def test_cited_answer_survives_refusal_boilerplate(stream, position):
    partial = "The source lists the semester venue as CDS 202 [S1]. A room number for the tutorial is unavailable."
    answer = {"after": partial + "\n\n" + ABSTAIN,
              "same_paragraph": partial + " " + ABSTAIN}[position]
    c, _ = make(FakeModel(answer))
    with c:
        response = ask(c, stream=stream)
    out = json.loads(response.text.strip().split("\n\n")[-1].split("\n")[1][6:]) if stream else response.json()
    assert out["status"] == "answered" and out["answer"] == partial
    assert out["citations"][0]["used"]


@pytest.mark.parametrize("stream", [False, True])
def test_primary_refusal_with_cited_background_remains_abstained(stream):
    c, _ = make(FakeModel(ABSTAIN + "\n\nThe evidence discusses other course details [S1]."))
    with c:
        response = ask(c, stream=stream)
    out = json.loads(response.text.strip().split("\n\n")[-1].split("\n")[1][6:]) if stream else response.json()
    assert out["status"] == "abstained" and out["citations"] == []


@pytest.mark.parametrize("stream", [False, True])
def test_refusal_with_only_a_marker_does_not_become_an_answer(stream):
    c, _ = make(FakeModel(ABSTAIN + " [S1]"))
    with c:
        response = ask(c, stream=stream)
    out = json.loads(response.text.strip().split("\n\n")[-1].split("\n")[1][6:]) if stream else response.json()
    assert out["status"] != "answered"


def test_bad_citations_regenerate_once_then_answer():
    model = FakeModel("Taught by Simmhan [S7].", "Taught by Yogesh Simmhan [S1].")
    c, _ = make(model)
    with c:
        out = ask(c).json()
    assert out["status"] == "answered" and out["attempts"] == 2 and len(model.requests) == 2
    assert model.requests[1]["messages"][:2] == model.requests[0]["messages"]     # same prefix, then a reminder


def test_answer_without_valid_citations_falls_back_to_excerpts():
    c, _ = make(FakeModel("Taught by Simmhan, no citation at all."))
    with c:
        out = ask(c).json()
    assert out["status"] == "fallback" and out["reason"] == "citation_check_failed" and out["answer"] is None
    assert out["excerpts"][0]["url"] == HIT["url"]


def test_generation_outage_returns_excerpts_not_an_error():
    c, _ = make(FakeModel(httpx.ConnectError("refused")))
    with c:
        r = ask(c)
    out = r.json()
    assert r.status_code == 200 and out["status"] == "fallback" and out["reason"] == "generation_unavailable"
    assert out["excerpts"] and out["answer"] is None


def test_no_retrieval_baseline_b0():
    model = FakeModel("Probably Prof. X.")
    c, ret = make(model)
    with c:
        out = ask(c, mode="none").json()
    assert out["status"] == "answered" and out["grounded"] is False and out["reason"] == "no_retrieval_baseline"
    assert "Evidence" not in model.requests[0]["messages"][1]["content"]


def test_vector_mode_is_passed_to_retrieval_and_unknown_modes_rejected():
    c, ret = make(FakeModel("Answer [S1]."))
    with c:
        ask(c, mode="vector")
        assert ask(c, mode="magic").status_code == 422
    assert ret.calls[0]["mode"] == "vector"


def test_pinned_snapshot_mismatch_is_409():
    c, _ = make(FakeModel("x"), retrieval=FakeRetrieval(raise_=SnapshotMismatch("pinned snap-0 is not active")))
    with c:
        assert ask(c, pinned_snapshot="snap-0").status_code == 409


# ---------------------------------------------------------------- streaming
def test_stream_sends_meta_tokens_and_a_checked_done_event():
    c, _ = make(FakeModel("DS 252 is taught by Yogesh Simmhan [S1]."))
    with c:
        r = ask(c, stream=True)
    events = [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:])) for b in r.text.strip().split("\n\n")]
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and kinds.count("token") >= 3
    assert events[0][1]["citations"][0]["url"] == HIT["url"]
    assert events[-1][1]["status"] == "answered"


def test_stream_with_bad_citations_tells_client_to_replace_text():
    c, _ = make(FakeModel("Simmhan teaches it."))
    with c:
        r = ask(c, stream=True)
    done = json.loads(r.text.strip().split("\n\n")[-1].split("\n")[1][6:])
    assert done["status"] == "fallback" and done["replace_streamed_text"] and done["excerpts"]


# ---------------------------------------------------------------- admission control
def test_rate_limit_returns_429_with_retry_after():
    c, _ = make(FakeModel("Answer [S1]."), rate_per_min=2)
    with c:
        codes = [ask(c).status_code for _ in range(3)]
        r = ask(c)
    assert codes == [200, 200, 429] and int(r.headers["retry-after"]) >= 1 and r.json()["reason"] == "rate_limited"


def test_tenant_concurrency_and_bounded_queue():
    async def scenario():
        adm = Admission(rate_per_min=100, tenant_concurrency=1, tenant_queue=1, queue_wait_s=0.5, global_concurrency=10)
        s1 = await adm.acquire("k1", "t")                       # in flight
        waiter = asyncio.create_task(adm.acquire("k2", "t"))   # queued
        await asyncio.sleep(0.01)
        with pytest.raises(Rejected) as full:
            await adm.acquire("k3", "t")                       # queue full -> immediate 429
        assert full.value.reason == "tenant_queue_full"
        other = await adm.acquire("k4", "other")               # other tenants are unaffected
        await s1.release()
        s2 = await waiter                                       # queued request gets the freed slot
        with pytest.raises(Rejected) as timeout:
            await adm.acquire("k5", "t")                       # waits 0.5 s, then gives up
        assert timeout.value.reason == "queue_timeout"
        for s in (s2, other):
            await s.release()
        assert adm.snapshot()["global_active"] == 0
    asyncio.run(scenario())


# ---------------------------------------------------------------- prompt, budget, mapping
def test_prompt_prefix_is_identical_across_requests():
    g = Generator(Settings())
    m1, _, _ = g.prepare("Who teaches DS 252?", [HIT])
    m2, _, _ = g.prepare("What are the credits for DS 221?", [{**HIT, "title": "Other"}])
    assert m1[0] == m2[0]
    assert m1[1]["content"].startswith("Evidence:\n[S1] ") and m2[1]["content"].startswith("Evidence:\n[S1] ")


def test_budget_drops_lowest_ranked_evidence_first():
    hits = [{**HIT, "title": f"T{i}", "text": "word " * 400} for i in range(3)]
    s = Settings(max_model_len=2400, max_output_tokens=200, prompt_safety_tokens=50)
    msgs, selected, budget = Generator(s).prepare("q?", hits)
    assert [h["title"] for h in selected] == ["T0", "T1"] and budget["dropped"] == 1
    assert budget["prompt_tokens"] <= budget["limit"] and "[S3]" not in msgs[1]["content"]


def test_verify_checks_markers():
    assert verify("A [S1] and B [S2].", 2)["ok"]
    v = verify("A [S1] and B [S5].", 2)
    assert not v["ok"] and v["unknown"] == ["S5"]
    assert not verify("No citations.", 2)["ok"] and verify(ABSTAIN, 2)["ok"]


def test_hits_from_retrieval_result():
    d = {"snapshot_id": "snap-1",
         "citations": [{"marker": "S1", "source_url": "u", "title": "Course Schedule", "section_heading": "DS 252",
                        "version_id": "v", "academic_term": "2025-AUG", "chunk_ids": ["c"], "excerpt": "e",
                        "rerank_score": 0.9, "via": "vector"}],
         "passages": [{"marker": "S1", "text": "Course ID: DS 252; Instructor: Yogesh Simmhan; Venue: CDS 202"}]}
    h = hits_from_result(d, now_term="2026-AUG")[0]
    assert h["academic_year"] == "2025-2026" and h["instructor"] == "Yogesh Simmhan"
    assert h["freshness"]["status"] == "past term (2025-AUG)"
    assert academic_year("2027-JAN") == "2026-2027" and academic_year(None) is None


# ---------------------------------------------------------------- other endpoints
def test_sources_health_ready_metrics():
    c, _ = make(FakeModel("Answer [S1]."))
    with c:
        ask(c)
        assert c.get("/v1/sources/c1", headers=H()).json()["text"] == "full text"
        assert c.get("/v1/sources/zzz", headers=H()).status_code == 404
        assert c.get("/v1/sources/c1").status_code == 401
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").status_code == 200
        m = c.get("/metrics", headers=H()).text
    assert 'askiisc_requests_total{mode="full",status="answered"} 1' in m and "askiisc_ttft_ms" in m


def test_four_roles_get_the_right_scopes():
    from askiisc_serving.auth import KeyStore
    ks = KeyStore({"g" * 20: {"tenant": "applicants", "role": "guest"},
                   "s" * 20: {"tenant": "students", "role": "student"},
                   "f" * 20: {"tenant": "faculty", "role": "faculty"},
                   "a" * 20: {"tenant": "office", "role": "admin"},
                   "o" * 20: {"tenant": "old", "role": "staff"}})
    see = lambda k: set(ks.lookup("Bearer " + k * 20).scopes)
    assert see("g") == {"public"}
    assert see("s") == {"public", "student"}
    assert see("f") == {"public", "student", "faculty"}
    assert "admin" not in see("f") and "faculty" not in see("s")
    assert see("a") == {"public", "student", "faculty", "admin", "staff"}
    assert see("o") == see("a")                      # older "staff" key files keep working


def test_unknown_role_is_rejected_at_startup():
    import pytest
    from askiisc_serving.auth import KeyStore
    with pytest.raises(ValueError, match="unknown role"):
        KeyStore({"x" * 20: {"tenant": "t", "role": "superuser"}})


def test_retry_merges_nested_mlx_usage_details():
    from askiisc_serving.app import _merge_meta
    a={'usage':{'prompt_tokens':120,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':80}},'generation_ms':100,'ttft_ms':20}
    b={'usage':{'prompt_tokens':140,'completion_tokens':12,'prompt_tokens_details':{'cached_tokens':110},'completion_tokens_details':{'reasoning_tokens':0}},'generation_ms':150}
    m=_merge_meta(a,b)
    assert m['usage']['prompt_tokens']==260
    assert m['usage']['completion_tokens']==22
    assert m['usage']['prompt_tokens_details']=={'cached_tokens':190}
    assert m['usage']['completion_tokens_details']=={'reasoning_tokens':0}
    assert m['generation_ms']==250 and m['ttft_ms']==20
    assert _merge_meta({'usage':None},{'usage':None})['usage'] is None
