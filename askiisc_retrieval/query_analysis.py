"""Cheap, deterministic query parsing: course codes and an explicit term, if any."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import current_term
from .departments import mentioned
from .parse import COURSE_RE, parse_term

MONTH_YEAR = re.compile(r"\b(aug(?:ust)?|jan(?:uary)?|summer)[a-z]*\.?,?\s*(\d{4})\b|\b(\d{4})\s*(aug(?:ust)?|jan(?:uary)?|summer)", re.I)
SEASON_ORDER = {"JAN": 0, "SUM": 1, "AUG": 2}
PREVIOUS = re.compile(r"\b(last|previous|prior)\s+(semester|term)\b", re.I)
# "this semester" / "currently" pin the current term explicitly. Without them, questions may be answered
# from the latest term up to now (e.g. "who teaches DS 211?" when it is not offered this term).
# Questions that ask for a person or a lab ("who works on X", "which professor/lab", "whom should I approach").
# They trigger a people pass: a vector search restricted to people (profile) chunks, so a matching
# research-interest chunk always reaches the reranker even when long general pages outscore it.
PEOPLE_INTENT = re.compile(
    r"\b(who|whom|whose|anyone|anybody|someone|somebody)\b|\b(professors?|prof|faculty|advisors?|advisers?|supervisors?|guides?|mentors?|"
    r"labs?|research groups?|groups?)\b", re.I)
THIS_TERM = re.compile(r"\b(this|current|ongoing|present)\s+(semester|term|session)\b|\bcurrently\b|\bright now\b", re.I)
INSTITUTE = re.compile(r"\b(?:IISc|Indian Institute of Science|institute[- ]wide|across (?:all )?departments)\b", re.I)
ADMISSIONS = re.compile(r"\badmissions?\b|\bapplication (?:portal|fee|deadline|form|process)\b", re.I)
APPLICATION = re.compile(r"\b(?:apply|applying|applications?|applicants?|admitted)\b", re.I)
ACADEMIC_APPLICATION = re.compile(r"\b(?:programmes?|programs?|degrees?|postgraduate|undergraduate)\b", re.I)
APPLY_TO_INSTITUTE = re.compile(r"\b(?:apply|applying)\s+(?:to|at)\s+(?:the\s+)?(?:IISc|Indian Institute of Science)\b", re.I)


def term_key(term: str) -> tuple[int, int]:
    y, s = term.split("-")
    return int(y), SEASON_ORDER[s]


def previous_term(term: str) -> str:
    """The previous regular semester (summer is skipped: 'last semester' in August means Jan-Apr)."""
    y, s = term.split("-")
    return f"{y}-JAN" if s in ("AUG", "SUM") else f"{int(y) - 1}-AUG"


@dataclass
class QueryInfo:
    query: str
    course_codes: list[str]
    explicit_term: str | None        # the user named a term
    target_term: str                 # explicit term, else the current one
    wants_people: bool = False       # asks for a person or lab -> people pass (see PEOPLE_INTENT)

    departments: tuple[str, ...] = ()

    @property
    def term_is_explicit(self) -> bool:
        return self.explicit_term is not None


def analyse(query: str, today_term: str | None = None) -> QueryInfo:
    now = today_term or current_term()
    codes: list[str] = []
    for a, b in COURSE_RE.findall(query.upper()):
        if f"{a}{b}" not in codes:
            codes.append(f"{a}{b}")
    term = parse_term(query)
    if term is None and (m := MONTH_YEAR.search(query)):
        mon = (m.group(1) or m.group(4)).lower()
        year = m.group(2) or m.group(3)
        term = f"{year}-{'AUG' if mon.startswith('aug') else 'SUM' if mon.startswith('sum') else 'JAN'}"
    if term is None and PREVIOUS.search(query):
        term = previous_term(now)
    if term is None and THIS_TERM.search(query):
        term = now
    return QueryInfo(query, codes, term, term or now, bool(PEOPLE_INTENT.search(query)), mentioned(query))


def institute_admissions(info: QueryInfo) -> bool:
    """General admissions rules come from institute sources, even in a one-department index."""
    if info.departments or info.course_codes:
        return False
    from .policy_graph import programmes, requested_facets

    query = info.query
    institute = INSTITUTE.search(query)
    if ADMISSIONS.search(query) or APPLY_TO_INSTITUTE.search(query):
        return True
    degree = programmes(query)
    if APPLICATION.search(query) and (degree or ACADEMIC_APPLICATION.search(query)):
        return True
    return bool(institute and degree and set(requested_facets(query)) & {
        "ELIGIBILITY", "SELECTION", "QUOTA", "INTAKE", "FEES", "FUNDING_AMOUNT", "FUNDING_CONDITIONS",
        "APPLICATION_DEADLINE", "INTERVIEW_DATES", "SESSION_START",
    })


def public_entrance_exam_context(query: str) -> bool:
    """Disambiguate public admission tests from protected semester/course timetables.

    This affects intent and login hints only; per-chunk ACLs still apply unconditionally.
    Explicit final/semester or exclusive-document requests retain the protected intent.
    """
    explicit = re.search(r"\bfinals?\b|end[ -]?(?:term|semester)|semester\s+exam|(?:course|class|lecture)\s+(?:schedule|timetable)|(?:faculty|professor|student)[ -](?:only|exclusive)|attached\s+(?:pdf|image)|signed\s+pdf", query, re.I)
    return not explicit and bool(re.search(r"\b(?:GATE|NET|JRF|JAM|JEST|entrance|qualifying|admissions?)\b", query, re.I))


def course_exam_intent(query: str) -> bool:
    return bool(re.search(r"\b(?:exam(?:ination)?s?|finals?)\b", query, re.I)) and not public_entrance_exam_context(query)

def requested_relations(query: str) -> tuple[str, ...]:
    """Map the requested answer type to graph relations, independent of course IDs/names."""
    if course_exam_intent(query):
        return ("EXAM_ON",)
    if re.search(r"\blabs?\b|\bresearch\s+groups?\b", query, re.I):
        return ("HEADS",)
    if re.search(r"pre.?requisites?|requires?|required courses?", query, re.I):
        return ("REQUIRES",)
    if re.search(r"\bwhere\b|\bvenue\b|\broom\b", query, re.I):
        return ("HELD_IN",)
    if re.search(r"teach|taught|instructor|\bcourses?\b", query, re.I):
        return ("TAUGHT_BY",)
    if re.search(r"research|works? on|interests?", query, re.I):
        return ("WORKS_ON", "HEADS")
    return ()
