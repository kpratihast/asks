"""Rule-based edges from schedule rows and catalogue sections. A stand-in for the ingestion team's
rule + SLM extractor, producing the same Edge format. Every edge keeps its supporting chunk ID."""
from __future__ import annotations

import re

from .contract import ChunkPayload
from .parse import course_codes
from .stores import Edge

GRAPH_RULES_VERSION = "rules-v6-hyperlinks"

PREREQ_RE = re.compile(r"pre-?\s?requisites?\s*:?\s*(.+?)(?:\.\s|$)", re.I | re.S)
TITLES = re.compile(r"^(prof(essor)?\.?|dr\.?|mr\.?|ms\.?|mrs\.?)\s+", re.I)
NOT_PEOPLE = {"faculty", "tba", "tbd", "staff", "guest", "various", "multiple", "instructors", "all"}


def person(name: str) -> str | None:
    """Alias resolution: 'Yogesh Simmhan', 'Y. Simmhan' and 'Prof. Y Simmhan' -> 'Person:y simmhan'.
    Key = first initial + surname. Collisions are possible in principle but rare within one department;
    the ingestion team can replace this with a lookup against the faculty page."""
    n = TITLES.sub("", name.strip()).strip(" .")
    parts = [p for p in re.split(r"[.\s]+", n) if p]
    if len(parts) >= 2 and len(parts[-1]) == 1:     # trailing initial = the initial written first elsewhere:
        parts = [parts[-1]] + parts[:-1]            # 'Venkatesh Babu R' == 'R. Venkatesh Babu', 'Sekar K' == 'K. Sekar'
    if not parts or any(ch.isdigit() for ch in n) or len(parts) > 5:
        return None
    if any(p.lower() in NOT_PEOPLE for p in parts):
        return None
    surname = parts[-1].lower()
    if len(surname) < 3:
        return None
    return f"Person:{parts[0][0].lower()} {surname}" if len(parts) > 1 else f"Person:{surname}"


def _fields(row_text: str) -> dict[str, str]:
    """'Course ID: DS 252; Instructor: X' (stand-in) or 'Course ID: DS 252 | Instructor: X' (ingestion)."""
    out = {}
    for part in re.split(r";\s|\s\|\s", row_text):
        if ": " in part:
            k, v = part.split(": ", 1)
            out[k.strip().lower()] = v.strip()
    return out


INTERESTS_HEADING = re.compile(r"(?:^|>\s*)([^>]+?)\s*>\s*Research Interests\s*$", re.I)
LAB_FACULTY = re.compile(r"Faculty\s*:\s*(.+?)\s*(?:Website\s*:|Lab Description\s*:|E-?mail\s*:|\n|$)", re.I)
LAB_NAME = re.compile(r"^\s*\d+\s*[.)]\s*")


def split_topics(text: str) -> list[str]:
    """'A, B (C, D), and E. F' -> ['a', 'b (c, d)', 'e', 'f']: commas, semicolons and sentence ends
    outside brackets. Long fragments (> 80 chars) are prose, not topics, and are skipped."""
    out, depth, cur = [], 0, []
    for i, ch in enumerate(text):
        depth += 1 if ch in "([" else -1 if ch in ")]" else 0
        sentence_end = ch == "." and (i + 1 == len(text) or text[i + 1].isspace())
        if depth <= 0 and (ch in ",;\n" or sentence_end):
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    topics = []
    for t in out:
        t = re.sub(r"^\s*(and|&)\s+", "", t.strip(" .-\u2013\u2014\t")).strip().lower()
        if 2 < len(t) <= 80 and t not in topics:
            topics.append(t)
    return topics


def _names(text: str) -> list[str]:
    return [n for n in (person(x) for x in re.split(r",|&|/|\band\b", text)) if n]


def edges_for_chunk(c: ChunkPayload) -> list[Edge]:
    from .scheduling import teaching_slot, exam_lookup
    edges: list[Edge] = []

    def add(src, rel, dst, term=None):
        if src and dst and src != dst:
            edges.append(Edge(src, rel, dst, (c.chunk_id,), c.version_id, c.access_scope, term, department=c.department))

    if c.doc_type == "schedule" and c.course_codes:
        f = _fields(c.text)
        course = f"Course:{c.course_codes[0]}"
        if c.academic_term:
            add(course, "OFFERED_IN", f"Term:{c.academic_term}", c.academic_term)
        inst = next((v for k, v in f.items() if "instructor" in k or "faculty" in k), "")
        for name in re.split(r",|&|/|\band\b", inst):
            add(course, "TAUGHT_BY", person(name), c.academic_term)
        venue = next((v for k, v in f.items() if k in ("venue", "room")), "")
        if venue:
            add(course, "HELD_IN", "Venue:" + venue.upper(), c.academic_term)

    if c.doc_type == "schedule" and c.academic_term:
        if c.course_codes and (slot := teaching_slot(c.text)):
            add('Course:' + c.course_codes[0], 'SCHEDULED_AT', slot.key(c.academic_term), c.academic_term)
        if lookup := exam_lookup(c.text, c.academic_term):
            slots, destination = lookup
            for slot in dict.fromkeys(slots):
                add(slot.key(c.academic_term), 'EXAM_ON', destination, c.academic_term)

    if c.doc_type == "catalogue":
        own = course_codes(c.section_heading)
        if own and (m := PREREQ_RE.search(c.text)):
            for code in course_codes(m.group(1)):
                add(f"Course:{own[0]}", "REQUIRES", f"Course:{code}")

    # Person -WORKS_ON-> Topic, from a profile's "Research Interests" section. Its chunk is the evidence,
    # and it links the chunk to the person, so a profile in the top results seeds graph expansion.
    if c.doc_type == "people" and (m := INTERESTS_HEADING.search(c.section_heading)):
        who = person(m.group(1))
        for topic in split_topics(c.text):
            add(who, "WORKS_ON", f"Topic:{topic}")

    # Person -HEADS-> Lab, from lab listings of the form "<n>. <Lab name>" + "Faculty: A, B ...".
    if (fm := LAB_FACULTY.search(c.text)) and re.search(r"\blab|group\b", c.section_heading, re.I):
        lab = LAB_NAME.sub("", c.section_heading.split(" > ")[-1]).strip()
        if lab:
            for who in _names(fm.group(1)):
                add(who, "HEADS", f"Lab:{lab}")
    return edges


def build_edges(chunks: list[ChunkPayload]) -> list[Edge]:
    out: list[Edge] = []
    for c in chunks:
        out.extend(edges_for_chunk(c))
    from .policy_graph import build_policy_edges
    out.extend(build_policy_edges(chunks))
    from .link_graph import build_link_edges
    out.extend(build_link_edges(chunks))
    return out
