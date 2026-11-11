"""The retriever: question + user scopes -> ranked, citable passages from ONE active snapshot.

Pipeline (each step is timed in RetrievalResult.timings_ms):
  1. analyse      course codes and an explicit term from the question
  2. embed        same pinned model as the index (FR-5; mismatch is an error, not a warning)
  3. search       Qdrant, filtered to active versions + user scopes (+ a course-code pass, and a people
                  pass over profile chunks for "who / which professor / which lab" questions)
  4. term logic   schedule-type chunks: explicit term -> only that term; otherwise prefer the current
                  term and keep one chunk per course (the latest one not after the current term)
  5. rank         chunks whose heading names the asked course first, then by similarity
  6. graph        bounded expansion; adds supporting chunks of facts, never the facts themselves.
                  Any graph error or timeout -> vector-only result with degraded=True (FR-9)
  7. audit        every returned passage re-checked against snapshot + scope; violators are dropped
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout

from . import config
from .connections import vector_exam_evidence
from .contract import ChunkPayload, Passage, RetrievalResult
from .departments import DEFAULT, SHARED, course_department
from .parse import course_codes
from .query_analysis import QueryInfo, analyse, institute_admissions, term_key, requested_relations


def _tkey(term: str | None) -> tuple[int, int]:
    return term_key(term) if term else (-1, -1)
from .citations import build_citations
from .rerank_text import rerank_input
from .snapshot import SnapshotProvider
from .stores import Embedder, GraphStore, QdrantStore, Reranker

log = logging.getLogger("askiisc.retrieval")
_graph_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="graph")


class EmbeddingModelMismatch(RuntimeError):
    pass


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore, snapshots: SnapshotProvider,
                 graph: GraphStore | None = None, *, top_k: int = config.TOP_K,
                 candidates: int = config.CANDIDATES, graph_extra: int = config.GRAPH_EXTRA,
                 graph_hops: int = config.GRAPH_HOPS, graph_timeout_s: float = config.GRAPH_TIMEOUT_S,
                 today_term: str | None = None, reranker: Reranker | None = None,
                 rerank_pool: int = config.RERANK_POOL, rerank_min: float | None = config.RERANK_MIN,
                 people_k: int = config.PEOPLE_K):
        self.embedder, self.store, self.snapshots, self.graph = embedder, store, snapshots, graph
        self.reranker, self.rerank_pool, self.rerank_min = reranker, rerank_pool, rerank_min
        self.top_k, self.candidates = top_k, candidates
        self.graph_extra, self.graph_hops, self.graph_timeout_s = graph_extra, graph_hops, graph_timeout_s
        self.today_term = today_term
        self.people_k = people_k
        from .private_documents import DocumentCatalog
        self.document_catalog = DocumentCatalog()
        self.audit_violations = 0                       # should stay 0 forever; surfaced in metrics

    # ------------------------------------------------------------------ public
    def retrieve(self, query: str, scopes: list[str], use_graph: bool = True,
                 use_rerank: bool = True) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000, 1)
            return now

        snap = self.snapshots.active()
        if snap.embedding_model != self.embedder.model_id:
            raise EmbeddingModelMismatch(
                f"query embedder {self.embedder.model_id!r} != snapshot model {snap.embedding_model!r}")
        info = analyse(query, self.today_term)
        t = lap("analyse", t0)

        departments = info.departments
        campus = set(snap.departments) - {SHARED}
        if not departments and len(campus) > 1 and (owner := course_department(info.course_codes)):
            departments = (owner,)
        shared_admissions = institute_admissions(info) and not self.document_catalog.matching(query)
        allowed_departments = (SHARED,) if shared_admissions else (*departments, SHARED) if departments else ()
        graph_departments = None
        graph_route = None
        clarification = False
        if shared_admissions:
            graph_departments = (SHARED,)
            graph_route = "institute_admissions"
        elif len(departments) == 1:
            graph_departments = departments
            if departments == (DEFAULT,) and not campus - {DEFAULT}:
                graph_departments = (*departments, None)  # graphs from existing CDS-only releases
        elif len(departments) > 1 or len(campus) > 1:
            graph_route = "comparison" if departments else "campus_search"
            use_graph = False
            if not departments:
                from .policy_graph import programmes, requested_facets
                clarification = bool(programmes(query) and set(requested_facets(query)) & {
                    "ELIGIBILITY", "INTAKE", "SELECTION", "TEST_SYLLABUS", "COURSEWORK",
                    "DISSERTATION", "LAB_PREFERENCES",
                })

        if len(departments) > min(self.top_k, 3):
            return RetrievalResult(query, [], snap.snapshot_id, evidence_sufficient=False,
                                   department_status="narrow_comparison")
        if clarification:
            return RetrievalResult(query, [], snap.snapshot_id, evidence_sufficient=False,
                                   department_status="clarify")

        qvec = self.embedder.embed_query(query)
        t = lap("embed", t)

        # Search each named department independently so a large department cannot
        # use up the candidate budget before smaller departments are searched.
        routes = [(dept,) for dept in allowed_departments] or [None]

        def search(k, **filters):
            found = {}
            for route in routes:
                if route:
                    filters["departments"] = route
                for chunk, score in self.store.search(qvec, k, snap.active_version_ids, scopes,
                                                      snap.embedding_model, **filters):
                    found.setdefault(chunk.chunk_id, (chunk, score))
            return list(found.values())

        initial_filters = {"doc_types": ["admissions", "policy"]} if shared_admissions else {}
        hits = search(self.candidates, **initial_filters)
        named_urls = self._named_source_urls(query, hits) if departments else []
        catalogue_leads = {}

        def section_key(chunk):
            return chunk.version_id, chunk.department, chunk.access_scope, chunk.section_heading

        if named_urls:
            # Shared front matter may name the document more clearly than its department sections.
            # Search the same source within each authorized route before the top-k cut.
            named_hits = search(self.candidates, source_urls=named_urls, **initial_filters)
            hits = named_hits + hits
            if not info.course_codes and "TAUGHT_BY" in requested_relations(query):
                # The first chunk identifies the course. Continuations often start with
                # a description or prerequisites, which do not make good catalogue entries.
                for chunk, _ in named_hits:
                    if chunk.doc_type == "catalogue" and course_codes(chunk.section_heading):
                        key = section_key(chunk)
                        catalogue_leads[key] = min(chunk.ordinal, catalogue_leads.get(key, chunk.ordinal))
        via = {c.chunk_id: "vector" for c, _ in hits}

        def source_priority(chunk):
            if chunk.source_url not in named_urls:
                return 2, False, False
            key = section_key(chunk)
            front_matter = bool(catalogue_leads) and key not in catalogue_leads
            continuation = chunk.ordinal > catalogue_leads.get(key, chunk.ordinal)
            return (0 if chunk.department in departments else 1), front_matter, continuation

        if info.course_codes:
            for c, score in search(self.candidates, course_codes=info.course_codes):
                if c.chunk_id not in via:
                    hits.append((c, score))
                    via[c.chunk_id] = "course_code"
        people_hits = []
        if info.wants_people and self.people_k and not shared_admissions:
            people_hits = search(self.people_k, doc_types=list(config.PEOPLE_DOC_TYPES))
        if departments:
            from .policy_graph import programmes

            requested_programmes = set(programmes(query))
            cfti = bool(re.search(r"\bCFTI\b", query, re.I))

            def applicable(hit):
                chunk, _ = hit
                if chunk.doc_type not in {"admissions", "policy"}:
                    return True
                if not cfti and re.search(r"\bCFTI\b", chunk.title, re.I):
                    return False
                source_programmes = set(programmes(chunk.section_heading) or programmes(chunk.title))
                return not (requested_programmes and source_programmes
                            and requested_programmes.isdisjoint(source_programmes))

            hits = [hit for hit in hits if applicable(hit)]
        t = lap("search", t)

        rerank = use_rerank and self.reranker is not None
        pool = self.rerank_pool if rerank else self.top_k
        ranked = self._rank(self._apply_term_logic(hits, info), info)
        if named_urls:
            ranked.sort(key=lambda item: source_priority(item[0]))
        link_anchors = [Passage(c, score, via[c.chunk_id]) for c, score in ranked if c.links or (c.doc_type == "schedule" and c.course_codes)]
        ranked = self._cover_departments(ranked, departments, lambda item: item[0].department)[:max(pool, self.top_k)]
        if rerank:                                      # people-pass chunks always reach the reranker, which
            have = {c.chunk_id for c, _ in ranked}      # decides; a short profile chunk is otherwise cut by
            for c, s in people_hits:                    # long general pages before it is ever scored (cf. q14)
                if c.chunk_id not in have:
                    ranked.append((c, s))
                    have.add(c.chunk_id)
                    via.setdefault(c.chunk_id, "people")
        passages = [Passage(c, s, via[c.chunk_id]) for c, s in ranked]
        t = lap("rank", t)

        rerank_error = None
        if rerank:                                      # before the graph, so graph seeds come from the best passages
            rerank_error = self._rerank(query, passages)
            passages = self._order(passages, info)
            if named_urls:
                passages.sort(key=lambda passage: source_priority(passage.chunk))
            passages = self._cover_departments(passages, departments,
                                               lambda passage: passage.chunk.department)[:self.top_k]
            t = lap("rerank", t)

        degraded, reason = False, None
        if use_graph and self.graph is not None:
            try:
                extra = _graph_pool.submit(self._graph_passages, passages, info, snap, scopes, graph_departments) \
                    .result(timeout=self.graph_timeout_s)
                t = lap("graph", t)
                if extra and rerank and rerank_error is None:
                    rerank_error = self._rerank(query, extra)   # graph additions compete on the same scale
                    t = lap("rerank_graph", t)
                passages = self._order(passages + extra, info) if rerank else passages + extra
                if named_urls:
                    passages.sort(key=lambda passage: source_priority(passage.chunk))
            except FutTimeout:
                degraded, reason = True, f"graph timeout after {self.graph_timeout_s}s"
            except Exception as e:                      # any graph failure -> vector-only (FR-9 fallback)
                degraded, reason = True, f"graph error: {type(e).__name__}: {e}"
            if degraded:
                log.warning("degraded retrieval: %s", reason)
                t = lap("graph", t)

        proof_passages, graph_proofs, connection_status = [], [], None
        if use_graph:
            from .connections import plan_for, resolve_connections
            try:
                plan = plan_for(info, self.graph, departments=graph_departments)
                if plan is not None:
                    connection_status = "missing_path"
                    if self.graph is not None:
                        proof_passages, graph_proofs, connection_status = _graph_pool.submit(
                            resolve_connections, self.graph, self.store, snap, scopes, plan, departments=graph_departments
                        ).result(timeout=self.graph_timeout_s)
            except Exception as exc:
                # A partial or failed join is never treated as a proved connected answer.
                connection_status = "unavailable"
                degraded, reason = True, f"connection error: {type(exc).__name__}"
        policy_trace = {}
        if use_graph and self.graph is not None and connection_status is None and not info.course_codes:
            from .policy_graph import retrieve_policy
            try:
                policy_passages, policy_proofs, policy_trace = _graph_pool.submit(
                    retrieve_policy, query, self.graph, self.store, snap, scopes, departments=graph_departments
                ).result(timeout=self.graph_timeout_s)
                if policy_passages:
                    proof_passages = policy_passages
                    graph_proofs.extend(policy_proofs)
            except Exception as exc:
                degraded, reason = True, f"policy graph error: {type(exc).__name__}"
                policy_trace = {"status": "unavailable"}
        if use_graph and self.graph is not None and not degraded and connection_status is None and not proof_passages:
            from .link_graph import linked_evidence
            try:
                proof_passages, link_proofs = _graph_pool.submit(
                    linked_evidence, passages, link_anchors, info, self.graph, self.store, snap, scopes, qvec
                ).result(timeout=self.graph_timeout_s)
                graph_proofs.extend(link_proofs)
            except Exception as exc:
                degraded, reason = True, f"link graph error: {type(exc).__name__}"
        # Explicit document questions must retain the authorized document, even when exact
        # course matches otherwise crowd the identifier-free exam lookup out of top-k.
        targeted = [] if shared_admissions else self._document_passages(info, snap, scopes)
        have = {p.chunk.chunk_id for p in targeted}
        passages = targeted + [p for p in passages if p.chunk.chunk_id not in have]
        if proof_passages:
            proved = {p.chunk.chunk_id for p in proof_passages}
            passages = proof_passages + [p for p in passages if p.chunk.chunk_id not in proved]
        passages = self._audit(passages, snap.active_version_ids, scopes, snap.embedding_model, allowed_departments)
        passages = self._cover_departments(passages, departments if len(departments) > 1 else (),
                                           lambda passage: passage.chunk.department)
        if not use_graph:
            passages, connection_status = vector_exam_evidence(info, passages)
        missing_departments = set(departments) - {p.chunk.department for p in passages}
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        scores = [p.rerank_score for p in passages if p.rerank_score is not None]
        top = max(scores) if (rerank and rerank_error is None and scores) else None
        sufficient = None if (top is None or self.rerank_min is None) else top >= self.rerank_min
        if missing_departments or (connection_status is not None and connection_status != "complete"):
            sufficient = False
        from .link_graph import EMAIL_QUESTION
        if info.course_codes and EMAIL_QUESTION.search(query):
            # Refuse to borrow another course's contact or a different term's instructor.
            exact = set(info.course_codes)
            if not any(exact.intersection(p.chunk.course_codes)
                       and (not info.term_is_explicit or p.chunk.academic_term == info.target_term)
                       for p in passages):
                sufficient = False
        citations = build_citations(passages)            # only audited passages can be cited
        return RetrievalResult(query, passages, snap.snapshot_id, degraded, reason,
                               info.target_term, timings, top, sufficient, rerank_error, citations,
                               graph_proofs=graph_proofs, connection_status=connection_status, policy_trace=policy_trace,
                               departments=departments,
                               department_status="missing_evidence" if missing_departments else graph_route)

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _cover_departments(items, departments, owner):
        first = {}
        for index, item in enumerate(items):
            if owner(item) in departments:
                first.setdefault(owner(item), index)
        indices = set(first.values())
        return [items[first[d]] for d in departments if d in first] + [
            item for index, item in enumerate(items) if index not in indices]

    def _apply_term_logic(self, hits: list[tuple[ChunkPayload, float]], info: QueryInfo):
        """Term-bound chunks (schedule rows) are versioned by term, not by document: an older term's
        schedule is a valid page, so the version filter cannot remove it. Handle it here."""
        timeless, by_key = [], {}
        for c, s in hits:
            if c.academic_term is None:
                timeless.append((c, s))
                continue
            if info.term_is_explicit:
                if c.academic_term == info.target_term:
                    timeless.append((c, s))
                continue
            if term_key(c.academic_term) > term_key(info.target_term):
                continue                                 # future term: not asked for
            key = (c.document_id, c.course_codes[0] if c.course_codes else c.section_heading.split(" | ")[-1])
            best = by_key.get(key)
            if best is None or term_key(c.academic_term) > term_key(best[0].academic_term):
                by_key[key] = (c, s)                     # keep the latest term <= current
        out, seen_text = [], set()
        for c, s in timeless + list(by_key.values()):
            if (c.department, c.text) not in seen_text:                  # drop exact duplicates
                seen_text.add((c.department, c.text))
                out.append((c, s))
        return out

    @staticmethod
    def _tier(c: ChunkPayload, info: QueryInfo) -> int:
        if not info.course_codes:
            return 0
        heading = c.section_heading.upper().replace(" ", "")
        if any(code in heading for code in info.course_codes):
            return 0                                      # the asked course's own section/row
        if any(code in c.course_codes for code in info.course_codes):
            return 1                                      # mentions it
        return 2

    @classmethod
    def _rank(cls, hits, info: QueryInfo):
        return sorted(hits, key=lambda cs: (cls._tier(cs[0], info), -cs[1]))

    @classmethod
    def _order(cls, passages: list[Passage], info: QueryInfo) -> list[Passage]:
        """Exact course match first (a deterministic signal), then cross-encoder relevance."""
        return sorted(passages, key=lambda p: (cls._tier(p.chunk, info),
                                               -(p.rerank_score if p.rerank_score is not None else -1.0)))

    def _rerank(self, query: str, passages: list[Passage]) -> str | None:
        """Sets rerank_score in place. A reranker failure keeps vector order (never fails the request).
        Schedule rows are scored in BOTH forms (raw key-value and sentence) and keep the higher score:
        the sentence lifted q18 from rank 8 to 1, but dropped q04 ('Who is the instructor for ...') from
        0.839 to 0.134, below the abstention threshold. Max of both can only match or beat either form."""
        texts, owners = [], []
        for i, p in enumerate(passages):
            raw = p.chunk.embedding_input()
            texts.append(raw)
            owners.append(i)
            alt = rerank_input(p.chunk)
            if alt != raw:
                texts.append(alt)
                owners.append(i)
        try:
            scores = self.reranker.score(query, texts)
        except Exception as e:
            log.warning("rerank failed, keeping vector order: %s", e)
            return f"{type(e).__name__}: {e}"
        best = [float("-inf")] * len(passages)
        for i, sc in zip(owners, scores):
            best[i] = max(best[i], float(sc))
        for p, sc in zip(passages, best):
            p.rerank_score = sc
        return None

    def _graph_passages(self, passages: list[Passage], info: QueryInfo, snap, scopes, departments=None) -> list[Passage]:
        """Seeds: courses and people named in the question (these define relevance), plus courses/people
        in the top-3 passages. Only evidence edges contribute chunks; chunks connected to more of the
        question's own entities rank first (q18: a course linked to BOTH named instructors)."""
        graph_filter = {"departments": departments} if departments else {}
        query_seeds = [f"Course:{c}" for c in info.course_codes] + self.graph.find_people(info.query, **graph_filter)
        seeds = list(query_seeds)
        for e in self.graph.entities_of_chunks([p.chunk.chunk_id for p in passages[:3]]):
            if e.split(":", 1)[0] in ("Course", "Person") and e not in seeds:
                seeds.append(e)
        if not seeds:
            return []
        # No limit here: truncating before ranking by QUESTION-seed coverage can drop the one edge that
        # links both people named in the question (seen on real data once reranking added more seeds).
        hits = self.graph.expand(seeds, snap.active_version_ids, scopes, self.graph_hops, limit=None,
                                 target_term=info.target_term, explicit=info.term_is_explicit,
                                 now_term=self.today_term or config.current_term(), **graph_filter)
        have = {p.chunk.chunk_id for p in passages}
        qs = set(query_seeds)
        requested = requested_relations(info.query)
        # Anchor an indirect lab/research question to the instructor of the named
        # course, rather than people found in unrelated vector seed passages.
        focus_people = {q for q in qs if q.startswith("Person:")}
        for course in (q for q in qs if q.startswith("Course:")):
            teaching = [h.edge for h in hits if h.edge.src == course and h.edge.rel == "TAUGHT_BY"]
            asked = [e for e in teaching if info.term_is_explicit and e.academic_term == info.target_term]
            if asked:
                teaching = asked
            elif teaching:
                latest = max((_tkey(e.academic_term) for e in teaching))
                teaching = [e for e in teaching if _tkey(e.academic_term) == latest]
            focus_people.update(e.dst for e in teaching)

        # Group evidence by (course, term). Coverage = question entities DIRECTLY on the group's edges
        # (2-hop reach would let every term of a course "cover" everything).
        groups: dict[tuple, dict] = {}
        for h in hits:
            if not h.evidence:
                continue                                     # bridges connect hops; never cited
            e = h.edge
            g = groups.setdefault((e.src, e.academic_term), {"cov": set(), "hop": h.hop, "chunks": [], "edges": [], "reach": set(), "focus": set(), "requested": False})
            direct = {q for q in qs if q in (e.src, e.dst)}
            g["cov"] |= direct
            g["reach"] |= set(h.seeds) & qs
            focused = focus_people & {e.src, e.dst}
            g["focus"] |= focused
            if e.rel in requested and (focused or not focus_people):
                g["requested"] = True
            g["hop"] = min(g["hop"], h.hop)
            g["edges"].append((len(direct), e))
            g["chunks"].extend(c for c in e.chunk_ids if c not in g["chunks"])
        if not info.term_is_explicit:                        # one term per course: most coverage, then latest
            best_term: dict[str, tuple] = {}
            for (src, term), g in groups.items():
                if term is None:
                    continue
                k = (len(g["cov"]), _tkey(term))
                if src not in best_term or k > best_term[src][0]:
                    best_term[src] = (k, term)
            groups = {key: g for key, g in groups.items() if key[1] is None or best_term[key[0]][1] == key[1]}
        # For "who / which lab" questions without a course code, a person's own facts (lab, research areas)
        # come before the courses they taught.
        person_first = info.wants_people and not info.course_codes
        ordered = sorted(groups.items(), key=lambda kv: (
            0 if kv[1]["requested"] else 1, -len(kv[1]["reach"]),
            -len(kv[1]["cov"]), 0 if (person_first and kv[0][0].startswith("Person:")) else 1,
            kv[1]["hop"], tuple(-x for x in _tkey(kv[0][1]))))
        best: dict[str, tuple[int, int, str, dict]] = {}
        for (src, term), g in ordered:
            n = len(g["cov"])
            why = f"links {n} entities named in the question" if n else f"{g['hop'] + 1} hops from the question"
            for cid in g["chunks"]:
                if cid not in have and cid not in best:
                    # explain with the edge that this chunk supports (best question coverage first)
                    e = max((ce for ce in g["edges"] if cid in ce[1].chunk_ids),
                            key=lambda ce: (ce[1].rel in requested, ce[0]))[1]
                    path = f"{e.src} -{e.rel}-> {e.dst} ({term + ', ' if term else ''}{why})"
                    support = {"relation": e.rel, "entities": [e.src, e.dst],
                               "coverage": n, "focus": len(g["focus"]),
                               "question_seeds": sorted(g["reach"]), "hop": g["hop"] + 1}
                    best[cid] = (n, g["hop"], path, support)
        wanted = list(best.items())[: self.graph_extra]                 # already in priority order
        got = {c.chunk_id: c for c in self.store.get([cid for cid, _ in wanted])}
        floor = min((p.score for p in passages), default=0.0)
        return [Passage(got[cid], floor, "graph", meta[2], graph_support=meta[3])
                for cid, meta in wanted if cid in got]

    @staticmethod
    def _named_source_urls(query: str, hits: list[tuple[ChunkPayload, float]]) -> list[str]:
        """Identify explicit references to a public source title among the authorized candidates."""
        query_words = ' '.join(re.findall(r'[a-z0-9]+', query.lower()))
        urls = {}
        for chunk, _ in hits:
            if chunk.access_scope != 'public' or chunk.source_url in urls:
                continue
            title = re.sub(r'\s+\([^()]+\)$', '', chunk.title)
            words = re.findall(r'[a-z0-9]+', title.lower())
            if len(words) >= 3 and re.search(r'\b(?:in|from|according to|per) (?:the )?'
                                             + ' '.join(words) + r'\b', query_words):
                urls[chunk.source_url] = len(words)
        # The longer title wins when one document's name is the prefix of another.
        longest = max(urls.values(), default=0)
        return [url for url, length in urls.items() if length == longest][:3]

    def _document_passages(self, info, snap, scopes):
        if not hasattr(self.store, 'document_chunks'):
            return []
        passages = []
        words = set(re.findall(r'[a-z0-9]+', info.query.lower()))
        docs = sorted(self.document_catalog.matching(info.query),
                      key=lambda d: 0 if d['access_scope'] == 'faculty' else 1)
        for doc in docs:
            if doc['access_scope'] not in scopes:
                continue
            chunks = self.store.document_chunks('/v1/documents/' + doc['id'], snap.active_version_ids,
                                                scopes, snap.embedding_model)
            if info.term_is_explicit:
                chunks = [c for c in chunks if not c.academic_term or c.academic_term == info.target_term]
            chunks.sort(key=lambda c: (self._tier(c, info),
                                       -len(words & set(re.findall(r'[a-z0-9]+', c.text.lower()))), c.ordinal))
            for c in chunks[:3]:
                passages.append(Passage(c, 1.0, 'document'))
        return passages

    def _audit(self, passages: list[Passage], version_ids, scopes, model, departments=()) -> list[Passage]:
        ok = []
        for p in passages:
            c = p.chunk
            if (c.version_id in version_ids and c.access_scope in scopes and c.embedding_model == model
                    and (not departments or c.department in departments)):
                ok.append(p)
            else:
                self.audit_violations += 1
                log.error("AUDIT: dropped chunk %s (version=%s scope=%s)", c.chunk_id, c.version_id, c.access_scope)
        return ok
