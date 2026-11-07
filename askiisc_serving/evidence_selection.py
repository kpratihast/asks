"""Select source-backed relationship evidence before enforcing a prompt source cap.

Graph metadata steers selection; verified connection paths accompany the source text.
A course/person anchor and its requested endpoint form a bundle. Unrelated neighbours
are not added merely to fill the source cap. No model calls or benchmark labels.
"""
from __future__ import annotations

import re
from askiisc_retrieval.query_analysis import analyse, requested_relations
from askiisc_retrieval.parse import course_codes


def evidence_bundle(question: str, hits: list[dict]) -> tuple[list[dict], set[str]]:
    """Return a ranked candidate pool plus source markers required by the relationship.

    An empty required set preserves ordinary ranked retrieval. Structured graph
    support must describe a requested relation; graph provenance alone is not a boost.
    """
    # Ordered graph proofs carry explicit dependencies across source documents.
    dependencies = {cid for h in hits for support in h.get('graph_support', [])
                    if support.get('proof_id') or support.get('policy_bundle_id') for cid in support.get('required_chunk_ids', [])}
    # A single policy passage can improve ranking without crowding out other
    # sources. Only a real multi-source dependency needs exclusive packing.
    if len(dependencies) == 1 and not any(
            support.get("proof_id") for hit in hits for support in hit.get("graph_support", [])):
        return list(hits), set()
    if dependencies:
        core = [h for h in hits if dependencies.intersection(h.get('chunk_ids', []))]
        available = {cid for h in core for cid in h.get('chunk_ids', [])}
        required = {h['marker'] for h in core} | {'missing-proof:' + cid for cid in dependencies - available}
        return core + [h for h in hits if h not in core], required
    info = analyse(question)
    requested = requested_relations(question)
    # A direct instructor/venue question is answered by its own course row.
    joins = bool(re.search(r'\blabs?\b|\bresearch\s+groups?\b|\b(other|which|what)\s+courses?\b|\btogether\b|\bsame\s+instructor\b', question, re.I))
    if not joins or not requested:
        return list(hits), set()
    endpoints = []
    for rank, h in enumerate(hits):
        support = [s for s in h.get('graph_support', []) if s.get('relation') in requested]
        support = [s for s in support if s.get('focus', 0) or s.get('question_seeds')]
        if support:
            coverage = max(s.get('coverage', 0) for s in support)
            focus = max(s.get('focus', 0) for s in support)
            endpoints.append((h, coverage, focus, rank))
    if not endpoints:
        return list(hits), set()
    if "HEADS" in requested and not info.course_codes and not any(
            s.get('question_seeds') for h, *_ in endpoints for s in h.get('graph_support', [])):
        # A direct named-lab question already has a vector answer. Neighbours of
        # other seed passages must not displace it without an explicit entity join.
        return list(hits), set()
    # With multiple named people, prefer the shared fact over facts about one person.
    best = max((x[1], x[2]) for x in endpoints)
    endpoints = [x for x in endpoints if (x[1], x[2]) == best]
    endpoints.sort(key=lambda x: x[3])
    anchors = []
    for code in info.course_codes:
        candidates = [h for h in hits if code in course_codes(h.get('section', ''))]
        # Prefer the requested term; an exact course row is cheaper and more precise
        # than its full multi-chunk syllabus. A timeless catalogue can anchor a cross-term hop.
        candidates.sort(key=lambda h: (
            0 if info.explicit_term and h.get('academic_term') == info.explicit_term else 1,
            0 if h.get('instructor') else 1,
            len(h.get('text', ''))))
        if candidates:
            anchors.append(candidates[0])
    core = []
    seen = set()
    for h in anchors + [x[0] for x in endpoints]:
        marker = h['marker']
        if marker not in seen:
            core.append(h)
            seen.add(marker)
    # The remainder is retained as a fallback if a required block cannot fit.
    return core + [h for h in hits if h['marker'] not in seen], seen
