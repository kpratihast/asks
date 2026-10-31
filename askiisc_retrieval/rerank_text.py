"""What the cross-encoder reads for each passage.

Cross-encoders are trained on prose; schedule rows are terse key-value text ("Course ID: DS 221;
Credits: 3:1; Instructor: S.Vadhiyar & Y. Simmhan; ...") and were consistently ranked low (v9: the
q18 answer at rank 8 of 9). Rows are therefore rewritten as one sentence FOR SCORING ONLY. The stored
text, the citation and what the generator sees are unchanged.

The same holds for profile "Research Interests" lists and lab listings ("Faculty: ... Website: ... Lab
Description: ..."): on 2026-10-08 the reranker gave Danish Pruthi's profile 0.033 for an NLP question
(dev q20). They get a sentence form too. The retriever scores both forms and keeps the higher one.
"""
from __future__ import annotations

import re

from .contract import ChunkPayload

_SEASON = {"AUG": "Aug-Dec", "JAN": "Jan-Apr", "SUM": "summer"}

# Column names vary across 12 years of the CDS schedule page; first match wins.
_KEYS = {
    "code": ("course id", "course", "course code"),
    "title": ("course title", "name", "description", "title"),
    "who": ("instructor", "instructor(s)", "instructors", "faculty"),
    "where": ("venue", "room"),
    "credits": ("credits", "course credits", "credit"),
    "start": ("first class", "start date"),
    "days": ("days of week", "day", "days"),
    "time": ("timings", "time"),
}


def term_label(term: str | None) -> str | None:
    if not term:
        return None
    year, season = term.split("-")
    return f"{_SEASON.get(season, season)} {year}"


def _fields(text: str) -> dict[str, str]:
    """'Course ID: DS 221; Instructor: X' (stand-in) or 'Course ID: DS 221 | Instructor: X' (ingestion v3)."""
    out: dict[str, str] = {}
    for part in re.split(r";\s|\s\|\s", text):
        if ": " in part:
            k, v = part.split(": ", 1)
            out.setdefault(k.strip().lower(), v.strip())
    return out


def _pick(f: dict[str, str], what: str) -> str | None:
    return next((f[k] for k in _KEYS[what] if f.get(k)), None)


def schedule_sentence(chunk: ChunkPayload) -> str | None:
    """'In the Aug-Dec 2024 term, DS 221 Introduction to Scalable Systems was taught by S.Vadhiyar & Y. Simmhan
    in CDS 202 (credits 3:1, first class 06th Aug 2024).'  None if the row lacks a code, title or instructor."""
    f = _fields(chunk.text)
    code, title, who = _pick(f, "code"), _pick(f, "title"), _pick(f, "who")
    if not (code and title and who):
        return None
    term = term_label(chunk.academic_term)
    s = (f"In the {term} term, " if term else "") + f"{code} {title} was taught by {who}"
    if where := _pick(f, "where"):
        s += f" in {where}"
    extras = [f"{k} {v}" for k, v in (("credits", _pick(f, "credits")), ("days", _pick(f, "days")),
                                      ("time", _pick(f, "time")), ("first class", _pick(f, "start"))) if v]
    return s + (f" ({', '.join(extras)})" if extras else "") + "."


_INTERESTS = re.compile(r"(?:^|>\s*)([^>]+?)\s*>\s*Research Interests\s*$", re.I)
_LAB = re.compile(r"Faculty\s*:\s*(?P<who>.+?)\s*(?:Website\s*:\s*\S+\s*)?(?:Lab Description\s*:\s*(?P<desc>.*))?$",
                  re.I | re.S)


def profile_sentence(chunk: ChunkPayload) -> str | None:
    """'Danish Pruthi works on research in natural language processing (NLP), deep learning, ...'"""
    m = _INTERESTS.search(chunk.section_heading)
    if chunk.doc_type != "people" or not m:
        return None
    topics = " ".join(chunk.text.split()).rstrip(".")
    # ingestion v3 profile headings start with the page title: "Yogesh Simmhan - Computational and Data ..."
    who = re.split(r"\s+[–|\-]\s+", m.group(1).strip())[0]
    return f"{who} works on research in {topics}."


def lab_sentence(chunk: ChunkPayload) -> str | None:
    """'The Natural Language ... Lab is led by Danish Pruthi. <description>'"""
    m = _LAB.search(" ".join(chunk.text.split()))
    if not m or not re.search(r"\blab|group\b", chunk.section_heading, re.I):
        return None
    lab = re.sub(r"^\s*\d+\s*[.)]\s*", "", chunk.section_heading.split(" > ")[-1]).strip()
    s = f"The {lab} is led by {m.group('who').strip()}."
    return s + (f" {m.group('desc').strip()}" if m.group("desc") else "")


def rerank_input(chunk: ChunkPayload) -> str:
    if chunk.doc_type == "schedule" and (sentence := schedule_sentence(chunk)):
        return f"{chunk.title}: {sentence}"
    if sentence := profile_sentence(chunk) or lab_sentence(chunk):
        return sentence
    return chunk.embedding_input()
