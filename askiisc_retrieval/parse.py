"""HTML -> sections. Stub for retrieval development; the ingestion team's extractor replaces it later.

Two things here matter for retrieval and must survive into their version:
  * `academic_term` is set per section from the nearest term heading (the schedule page holds
    every term from 2014 onwards on ONE page, so a per-document term would be wrong);
  * schedule tables become one section per row, so each course-term pair is citable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

MAIN = [".entry-content", "article", "main", "#content", "#main", ".content", "body"]
DROP = ["script", "style", "noscript", "iframe", "form", "svg", "button", "nav", "header", "footer",
        "aside", "[role=navigation]", "[role=contentinfo]", ".breadcrumb", ".navbar", ".sidebar", "#sidebar",
        # Theme footers/menus are often plain <div>s (e.g. the CDS "Locate us" block). Keep these selectors
        # NARROW: page builders such as Elementor wrap ALL content in "...widget-container" / "...widget"
        # elements, so a broad [class*=widget] or [class*=menu] deletes whole pages.
        "[class*=footer]", "[id*=footer]", "[id*=colophon]", "[class*=nav-menu]", "[class*=main-menu]",
        "[class*=sub-menu]", "[id*=menu]", "[class*=scroll-up]", "[class*=scrollup]"]
# Last line of defence for boilerplate that survives the selectors
BOILERPLATE_RE = re.compile(r"(©|&copy|copyright)\s*\d{4}|website by|scroll up|locate us", re.I)
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5}
BLOCKS = {"p", "li", "dd", "dt", "pre", "blockquote"}

COURSE_RE = re.compile(r"\b(?!CDS\b)([A-Z]{1,3}[0-9]?)\s?-?\s?(\d{3})\b")
# A paragraph that *is* a course heading, e.g. "DS 252 (AUG) 3:1 Cloud Computing"
COURSE_HEADING_RE = re.compile(r"^\s*(?!CDS\b)[A-Z]{1,3}[0-9]?\s?\d{3}\s*\((?:AUG|JAN|Aug|Jan)\)\s*\d:\d")
_MON = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
# Every format on the CDS schedule page 2014-2026, e.g. "Aug - Dec 2026", "Jan 2024 - Apr 2024",
# "Feb-May (2021)", "Oct 2020 -Jan 2021", "May-June (summer) 2019", "(Aug-Dec, 2016)", "(Jan-May 2016)"
TERM_RE = re.compile(
    rf"\b{_MON}\s*(\d{{4}})?\s*(?:-|–|—|to)\s*{_MON}\s*(\d{{4}})?"
    rf"\s*(?:\(\s*summer\s*\))?\s*,?\s*\(?\s*(\d{{4}})?", re.I)
_SEASON = {m: "JAN" for m in ("jan", "feb", "mar", "apr")} | \
          {m: "SUM" for m in ("may", "jun", "jul")} | \
          {m: "AUG" for m in ("aug", "sep", "oct", "nov", "dec")}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def course_codes(text: str) -> list[str]:
    seen: list[str] = []
    for a, b in COURSE_RE.findall(text):
        c = f"{a}{b}"
        if c not in seen:
            seen.append(c)
    return seen


def is_course_code(code: str) -> bool:
    """'DS252' / 'E0251' yes; 'CDS202' (a room in the CDS building) or 'DS25' no."""
    m = re.fullmatch(r"([A-Z]{1,3}[0-9]?)(\d{3})", code or "")
    return bool(m) and course_codes(f"{m.group(1)} {m.group(2)}") == [code]


def parse_term(text: str) -> str | None:
    """'2026-AUG' | '2026-JAN' | '2019-SUM', from the START month and the first year mentioned."""
    m = TERM_RE.search(text)
    if not m:
        return None
    year = m.group(2) or m.group(4) or m.group(5)
    if not year:
        return None
    return f"{year}-{_SEASON[m.group(1)[:3].lower()]}"


@dataclass
class Section:
    heading_path: list[str]
    text: str
    academic_term: str | None = None
    kind: str = "text"                      # text | row
    course_codes: list[str] = field(default_factory=list)
    department: str | None = None

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else "(page body)"


@dataclass
class ParsedPage:
    url: str
    title: str
    sections: list[Section]
    root: str = ""                 # which container the content was read from (diagnostics)
    boiler_lines: int = 0          # footer-phrase lines removed (diagnostics)


def _table_rows(table: Tag) -> tuple[list[str], list[list[str]]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [norm(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return [], []
    has_th = table.find("th") is not None
    header = rows[0] if (has_th or any("course" in c.lower() for c in rows[0])) else []
    return header, rows[1:] if header else rows


def drop_report(html: str | bytes) -> list[tuple[str, int]]:
    """How many characters of text each DROP selector would remove (diagnostics)."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for sel in DROP:
        n = sum(len(norm(t.get_text(" "))) for t in soup.select(sel))
        for t in soup.select(sel):
            t.decompose()
        if n:
            out.append((sel, n))
    return out


def parse_html(url: str, html: str | bytes, *, tagged_departments: bool = False) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    title = norm(soup.title.get_text()) if soup.title else url
    title = re.split(r"\s+[|–-]\s+|:\s+", title)[0] or title   # drop " | Site" / ": Department of ..." suffix
    root_sel, root = next(((s, r) for s in MAIN if (r := soup.select_one(s))), ("(document)", soup))
    for sel in DROP:
        # Theme flags such as Divi's body class "et-tb-has-footer" are not footer elements.
        for t in root.select(sel):
            t.decompose()

    sections: list[Section] = []
    path: list[tuple[int, str, str | None]] = []
    buf: list[str] = []
    term: list[str | None] = [None]

    def section_department():
        return next((owner for _, _, owner in reversed(path) if owner), None)

    def flush():
        txt = "\n".join(buf).strip()
        if txt:
            heading = " > ".join(h for _, h, _ in path)
            sections.append(Section([h for _, h, _ in path], txt, term[0],
                                    course_codes=course_codes(heading + " " + txt),
                                    department=section_department()))
        buf.clear()

    def set_heading(level: int, text: str, department: str | None = None):
        flush()
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, text, department))
        if (t := parse_term(text)) is not None:
            term[0] = t
        elif level <= 2:
            term[0] = None                    # a new top-level section ends the previous term

    def walk(node: Tag):
        for ch in node.children:
            if isinstance(ch, NavigableString):
                if type(ch) is NavigableString and (t := norm(str(ch))):
                    buf.append(t)
                continue
            if not isinstance(ch, Tag):
                continue
            if ch.name in HEADINGS:
                if t := norm(ch.get_text(" ")):
                    owner = ch.get("data-askiisc-department") if tagged_departments else None
                    if owner:
                        from .departments import validate_codes
                        validate_codes([owner])
                    set_heading(HEADINGS[ch.name], t, owner)
            elif ch.name == "table":
                header, rows = _table_rows(ch)
                flush()
                for cells in rows:
                    if header and len(header) == len(cells):
                        text = "; ".join(f"{h}: {v}" for h, v in zip(header, cells) if v)
                    else:
                        text = " | ".join(cells)
                    codes = course_codes(cells[0] if cells else "")
                    label = (path[-1][1] if path else title) + (f" | {cells[0]}" if codes else "")
                    sections.append(Section([h for _, h, _ in path[:-1]] + [label], text, term[0], "row",
                                            course_codes=codes or course_codes(text), department=section_department()))
            elif ch.name in BLOCKS and not ch.find(["table"] + list(HEADINGS)):
                t = norm(ch.get_text(" "))
                if not t:
                    continue
                if ch.name in ("p", "dt") and COURSE_HEADING_RE.match(t) and len(t) < 160:
                    set_heading(4, t)                       # course headings styled as <p><strong>
                elif ch.name == "p" and parse_term(t) and len(t) < 120:
                    set_heading(3, t)                       # term headings styled as paragraphs
                else:
                    buf.append(("- " if ch.name == "li" else "") + t)
            else:
                walk(ch)

    walk(root)
    flush()
    # Remove footer-phrase LINES only. (Dropping whole sections that contained such a line threw away pages
    # without headings, where content and footer end up in one section: the faculty list, 2026-10-08.)
    cleaned, removed = [], 0
    for sec in sections:
        if BOILERPLATE_RE.search(sec.heading):
            removed += sec.text.count("\n") + 1
            continue
        lines = sec.text.split("\n")
        keep = [ln for ln in lines if not BOILERPLATE_RE.search(ln)]
        removed += len(lines) - len(keep)
        if any(ln.strip() for ln in keep):
            sec.text = "\n".join(keep)
            cleaned.append(sec)
    return ParsedPage(url, title, cleaned, root_sel, removed)


def remove_template_sections(pages: list[ParsedPage], min_pages: int = 10, share: float = 0.3,
                             similarity: float = 0.5) -> dict[str, int]:
    """Site template removal (sidebars such as "Recent News" / "Contact Us" repeated on every page).

    A section is template if its last heading occurs on >= `share` of all pages AND its words overlap
    (Jaccard >= `similarity`) with the most common text under that heading. Overlap rather than equality,
    because a news sidebar shifts by an item between pages saved on different days. Corpus-level and
    question-independent; skipped for small corpora (< min_pages) where frequencies mean nothing.
    Returns {heading: sections removed}.
    """
    from collections import Counter
    if len(pages) < min_pages:
        return {}
    leaf = lambda sec: sec.heading_path[-1].strip().lower() if sec.heading_path else ""
    words = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))
    per_page = Counter(h for p in pages for h in {leaf(sec) for sec in p.sections if leaf(sec)})
    template = {h for h, n in per_page.items() if n >= share * len(pages)}
    modal: dict[str, set[str]] = {}
    for h in template:
        texts = Counter(sec.text for p in pages for sec in p.sections if leaf(sec) == h)
        modal[h] = words(texts.most_common(1)[0][0])
    removed: Counter = Counter()
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    for p in pages:
        kept = []
        own = norm(p.title)
        for sec in p.sections:
            h = leaf(sec)
            # A template block is the page's own content on the page it is named after (the "Contact Us"
            # page itself). Never remove it there. (Fixed 2026-10-08: the contact page came out empty.)
            if h in template and norm(h) != own:
                w = words(sec.text)
                if w and len(w & modal[h]) / len(w | modal[h]) >= similarity:
                    removed[sec.heading_path[-1]] += 1
                    continue
            kept.append(sec)
        p.sections = kept
    return dict(removed)
