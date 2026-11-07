"""Build and follow edges backed by actual hyperlinks in ingested passages."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque

from .contract import Passage
from .stores import Edge
from .web_links import EMAIL, page_key

EMAIL_QUESTION = re.compile(r"\b(?:e[- ]?mail|mail address)\b", re.I)


def build_link_edges(chunks):
    targets = defaultdict(set)
    for chunk in chunks:
        if key := page_key(chunk.source_url):
            targets[key].add((chunk.department, chunk.source_url))
    edges = []
    for chunk in chunks:
        for link in dict.fromkeys(chunk.links[:16]):
            for department, url in sorted(targets.get(link, ())):
                if department == chunk.department and page_key(chunk.source_url) != link:
                    edges.append(Edge(f"Chunk:{chunk.chunk_id}", "LINKS_TO", f"Page:{url}",
                                      (chunk.chunk_id,), chunk.version_id, chunk.access_scope,
                                      chunk.academic_term, department=department))
    edges.extend(_instructor_contact_edges(chunks))
    return edges


def _instructors(chunk):
    from .graph_rules import _fields

    names = next((value for key, value in _fields(chunk.text).items()
                  if key in {"instructor", "instructors", "instructor(s)", "faculty"}), "")
    names = re.split(r",|&|/|\band\b", names.split("(")[0], flags=re.I)
    names = [tuple(re.findall(r"[a-z]+", name.lower())) for name in names]
    return [name for name in names if 2 <= len(name) <= 6 and len(name[0]) > 1]


def _instructor_contact_edges(chunks):
    """Join a course row to a directory contact only through an exact full name."""
    names_by_department = defaultdict(set)
    edges = []
    for chunk in chunks:
        if chunk.doc_type != "schedule" or not chunk.course_codes:
            continue
        for name in _instructors(chunk):
            node = f"Contact:{chunk.department}/{' '.join(name)}"
            names_by_department[chunk.department].add(name)
            edges.append(Edge(f"Chunk:{chunk.chunk_id}", "INSTRUCTOR", node, (chunk.chunk_id,),
                              chunk.version_id, chunk.access_scope, chunk.academic_term,
                              department=chunk.department))
    for chunk in chunks:
        if chunk.doc_type != "people":
            continue
        names = names_by_department[chunk.department]
        for line in chunk.contacts:
            emails = set(address.lower() for address in EMAIL.findall(line))
            if len(emails) != 1:
                continue
            words = re.findall(r"[a-z]+", f"{chunk.section_heading} {line}".lower())
            matches = {tuple(words[i:i+size]) for i in range(len(words)) for size in range(2, 7)
                       if tuple(words[i:i+size]) in names and i + size <= len(words)}
            if len(matches) == 1:
                name = next(iter(matches))
                node = f"Contact:{chunk.department}/{' '.join(name)}"
                edges.append(Edge(node, "PUBLISHED_EMAIL", f"Chunk:{chunk.chunk_id}", (chunk.chunk_id,),
                                  chunk.version_id, chunk.access_scope, qualifiers=(("email", next(iter(emails))),),
                                  department=chunk.department))
    return edges


def _directory_contact(anchors, graph, store, info, snapshot, scopes):
    exact = set(info.course_codes)
    query_words = " " + " ".join(re.findall(r"[a-z]+", info.query.lower())) + " "
    for anchor in anchors:
        chunk = anchor.chunk
        if chunk.doc_type != "schedule" or not exact.intersection(chunk.course_codes):
            continue
        evidence = graph.outgoing(f"Chunk:{chunk.chunk_id}", "INSTRUCTOR")
        chosen = [edge for edge in evidence if f" {edge.dst.split('/', 1)[-1]} " in query_words] or evidence
        found = []
        if len(chosen) != 1:
            continue
        first = chosen[0]
        if first.version_id != chunk.version_id or first.scope != chunk.access_scope:
            continue
        for second in graph.outgoing(first.dst, "PUBLISHED_EMAIL")[:16]:
            if (second.department == chunk.department and second.version_id in snapshot.active_version_ids
                    and second.scope in scopes):
                found.append(second)
        addresses = {dict(edge.qualifiers)["email"] for edge in found}
        if len(addresses) != 1:
            continue
        address = next(iter(addresses))
        chunks = {c.chunk_id: c for c in store.get([cid for edge in found for cid in edge.chunk_ids])}
        for edge in found:
            target = chunks.get(edge.chunk_ids[0])
            if (target and target.access_scope == edge.scope and target.version_id == edge.version_id
                    and target.department == chunk.department and target.embedding_model == snapshot.embedding_model
                    and address in {mail.lower() for mail in EMAIL.findall(target.text)}):
                return 1.0, (chunk, target), (first, edge), [address]
    return None


def _contact_addresses(anchor, target, course_codes):
    contacts = target.contacts or [line for line in target.text.splitlines() if EMAIL.search(line)]
    if not course_codes:
        return {address.lower() for line in contacts for address in EMAIL.findall(line)}
    names = _instructors(anchor)
    addresses = set()
    for line in contacts:
        words = re.findall(r"[a-z]+", f"{target.section_heading} {line}".lower())
        person_matches = any(any(tuple(words[i:i+len(name)]) == name for i in range(len(words)))
                             for name in names)
        same_course = bool(set(course_codes) & set(target.course_codes))
        labelled = re.match(r"^\s*[-•]?\s*instructors?(?:\(s\))?\s*:", line, re.I)
        if person_matches or (same_course and labelled and not names):
            addresses.update(address.lower() for address in EMAIL.findall(line))
    return addresses


def linked_evidence(passages, anchors, info, graph, store, snapshot, scopes, query_vector):
    """Find a useful linked passage with at most two hops and eight page lookups."""
    outgoing = getattr(graph, "outgoing", None)
    if outgoing is None or not passages:
        return [], []
    exact = set(info.course_codes)
    seeds = [p for p in (anchors if exact else passages)
             if p.chunk.links and (not exact or exact.intersection(p.chunk.course_codes))][:3]
    contact = bool(EMAIL_QUESTION.search(info.query))
    if contact and exact and any(_contact_addresses(p.chunk, p.chunk, info.course_codes) for p in passages):
        return [], []  # a direct instructor contact is already in the evidence
    floor = max(0.2, max(p.score for p in passages) - 0.12)
    queue = deque((p.chunk, (p.chunk,), ()) for p in seeds)
    lookups = {}
    visible_ids = {p.chunk.chunk_id for p in passages}
    candidates = []
    while queue and len(lookups) < 8:
        anchor, chain, steps = queue.popleft()
        current = chain[-1]
        for edge in outgoing(f"Chunk:{current.chunk_id}", "LINKS_TO"):
            if (edge.chunk_ids != (current.chunk_id,) or edge.version_id != current.version_id
                    or edge.scope != current.access_scope or edge.department != current.department):
                continue
            url = edge.dst.removeprefix("Page:")
            if url in {chunk.source_url for chunk in chain}:
                continue
            key = (url, current.department)
            if key not in lookups:
                if len(lookups) >= 8:
                    break
                lookups[key] = store.search(query_vector, 3, snapshot.active_version_ids, scopes,
                                            snapshot.embedding_model, departments=(current.department,),
                                            source_urls=(url,))
            for target, score in lookups[key]:
                if (target.department != current.department or target.source_url != url
                        or target.version_id not in snapshot.active_version_ids or target.access_scope not in scopes
                        or target.embedding_model != snapshot.embedding_model):
                    continue
                if (target.academic_term and (info.term_is_explicit or current.academic_term)
                        and target.academic_term != (info.target_term if info.term_is_explicit else current.academic_term)):
                    continue
                path = (*steps, edge, Edge(edge.dst, "CONTAINS", f"Chunk:{target.chunk_id}",
                                           (target.chunk_id,), target.version_id, target.access_scope,
                                           target.academic_term, department=target.department))
                next_chain = (*chain, target)
                if contact:
                    if addresses := _contact_addresses(anchor, target, info.course_codes):
                        candidates.append((score, next_chain, path, sorted(addresses)))
                elif score >= floor and target.chunk_id not in visible_ids:
                    candidates.append((score, next_chain, path, []))
                if not steps and target.links:
                    queue.append((anchor, next_chain, path))
    if not candidates and contact and exact:
        if fallback := _directory_contact(anchors[:16], graph, store, info, snapshot, scopes):
            candidates.append(fallback)
    if not candidates:
        return [], []
    _, chain, path, addresses = max(candidates, key=lambda candidate: (candidate[0], -len(candidate[1])))
    ids = [c.chunk_id for c in chain]
    proof_id = hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]
    proof = {"id": proof_id, "kind": "hyperlink", "required_chunk_ids": ids,
             "steps": [{"from": e.src, "relation": e.rel, "to": e.dst, "chunk_ids": list(e.chunk_ids),
                        "version_id": e.version_id, "access_scope": e.scope} for e in path]}
    # Endpoints are ordinary source passages. Keep every page in the selected path.
    support = {"proof_id": proof_id, "required_chunk_ids": ids, "relation": "LINKS_TO"}
    if addresses:
        support["contact_addresses"] = addresses
    connection = "The same instructor is named in the course and faculty directory" if path[0].rel == "INSTRUCTOR" else "The ingested pages link in this order"
    relationship = connection + ": " + " → ".join(c.title for c in chain)
    return [Passage(c, 1.0, "graph", relationship, graph_support=support) for c in chain], [proof]
