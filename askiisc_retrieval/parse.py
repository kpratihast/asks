"""HTML -> sections parser for multi-department IISc retrieval.

Two things here matter for retrieval and must survive into production:
   * `academic_term` is set per section from the nearest term heading (the schedule page holds
     every term from 2014 onwards on ONE page, so a per-document term would be wrong);
   * schedule tables become one section per row, so each course-term pair is citable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, NavigableString, Tag

from .web_links import EMAIL, anchor_emails, page_key

MAIN = [".entry-content", "article", "main", "#content", "#main", ".content", "body"]
DROP = [
    "script", "style", "noscript", "iframe", "form", "svg", "button", "nav", "header", "footer",
    "aside", "[role=navigation]", "[role=contentinfo]", ".breadcrumb", ".navbar", ".sidebar", "#sidebar",
    # Theme footers/menus are often plain <div>s (e.g. the CDS "Locate us" block). Keep these selectors
    # NARROW: page builders such as Elementor wrap ALL content in "...widget-container" / "...widget"
    # elements, so a broad [class*=widget] or [class*=menu] deletes whole pages.
    "[class*=footer]", "[id*=footer]", "[id*=colophon]", "[class*=nav-menu]", "[class*=main-menu]",
    "[class*=sub-menu]", "[id*=menu]", "[class*=scroll-up]", "[class*=scrollup]",
]

# Last line of defence for boilerplate that survives the selectors
BOILERPLATE_RE = re.compile(
    r"(©|&copy|copyright)\s*\d{4}|website by|scroll up|locate us|all rights reserved|powered by\s+wordpress",
    re.I,
)
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5}
BLOCKS = {"p", "li", "dd", "dt", "pre", "blockquote"}

# Table headers identifying room or timing data that must not be parsed as course numbers
VENUE_COLUMNS = {"venue", "room", "hall", "classroom", "location", "slot", "timings", "time"}

# Negative lookbehinds reject venue/room keywords.
# Negative lookahead (?!(?:CDS|CSA|ECE|DESE|RBCCPS|SERC)\b) prevents building/room numbers (e.g., CDS 202, CSA 112)
# Matches 1-3 uppercase letters (plus optional digit) and any 3 digits (\d{3}) to support both live courses and test fixtures.
COURSE_RE = re.compile(
    r"(?<!Room\s)(?<!Rm\.\s)(?<!Venue:\s)(?<!Venue\s)(?<!Hall\s)(?<!LH\s)(?<!CR\s)"
    r"\b(?!(?:CDS|CSA|ECE|DESE|RBCCPS|SERC)\b)"
    r"([A-Z]{1,3}[0-9]?)\s?-?\s?(\d{3})\b",
    re.I,
)

# A paragraph that is a course heading, e.g. "DS 252 (AUG) 3:1 Cloud Computing" or "E0 249 (AUG) 3:1"
COURSE_HEADING_RE = re.compile(
    r"^\s*(?!(?:CDS|CSA|ECE)\b)[A-Z]{1,3}[0-9]?\s?\d{3}\s*\((?:AUG|JAN|Aug|Jan)\)\s*\d:\d",
    re.I,
)

_MON = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
# Formats on schedule pages, e.g. "Aug - Dec 2026", "Jan 2024 - Apr 2024", "Feb-May (2021)"
TERM_RE = re.compile(
    rf"\b{_MON}\s*(\d{{4}})?\s*(?:-|–|—|to)\s*{_MON}\s*(\d{{4}})?"
    rf"\s*(?:\(\s*summer\s*\))?\s*,?\s*\(?\s*(\d{{4}})?",
    re.I,
)
_SEASON = {m: "JAN" for m in ("jan", "feb", "mar", "apr")} | \
          {m: "SUM" for m in ("may", "jun", "jul")} | \
          {m: "AUG" for m in ("aug", "sep", "oct", "nov", "dec")}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def infer_dept_from_url(url: str) -> str | None:
    """Infer the department identifier from common IISc hostnames."""
    u = url.lower()
    if "cds.iisc.ac.in" in u:
        return "CDS"
    if "csa.iisc.ac.in" in u:
        return "CSA"
    if "ece.iisc.ac.in" in u:
        return "ECE"
    if "ee.iisc.ac.in" in u:
        return "EE"
    return None


def course_codes(text: str) -> list[str]:
    """Extract course codes formatted as uppercase without spaces (e.g., 'DS252', 'CS777', 'E0249')."""
    seen: list[str] = []
    for a, b in COURSE_RE.findall(text or ""):
        c = f"{a.upper()}{b}"
        if c not in seen:
            seen.append(c)
    return seen


def is_course_code(code: str) -> bool:
    """'DS252' / 'E0251' / 'CS777' yes; 'CDS202' (a room in CDS) or 'DS25' no."""
    clean = re.sub(r"\s+", "", (code or "").upper())
    m = re.fullmatch(r"([A-Z]{1,3}[0-9]?)(\d{3})", clean)
    return bool(m) and course_codes(f"{m.group(1)} {m.group(2)}") == [m.group(1) + m.group(2)]


def parse_term(text: str) -> str | None:
    """'2026-AUG' | '2026-JAN' | '2019-SUM', from the START month and first year mentioned."""
    m = TERM_RE.search(text)
    if not m:
        return None
    # Group 1 = start month, Group 2/4/5 = year candidates (Group 3 is end month name)
    year = m.group(2) or m.group(4) or m.group(5)
    if not year:
        return None
    start_m = m.group(1).lower()[:3]
    season = _SEASON.get(start_m)
    if not season:
        return None
    return f"{year}-{season}"


@dataclass
class Section:
    heading_path: list[str]
    text: str
    academic_term: str | None = None
    kind: str = "text"                      # text | row
    course_codes: list[str] = field(default_factory=list)
    department: str | None = None
    links: list[tuple[str, str]] = field(default_factory=list)  # (text line, link target)
    contacts: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)


@dataclass
class ParsedPage:
    url: str
    title: str
    sections: list[Section]
    root: str = ""                          # which container the content was read from (diagnostics)
    boiler_lines: int = 0                   # footer-phrase lines removed (diagnostics)


def _table_rows(table: Tag) -> tuple[list[str], list[tuple[list[str], list[str]]]]:
    headers = [norm(th.get_text(" ")) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr"):
        cells = [norm(td.get_text(" ")) for td in tr.find_all("td")]
        if not cells:
            continue
        hrefs = [a["href"] for a in tr.find_all("a", href=True)]
        rows.append((cells, hrefs))
    return headers, rows


def drop_report(html: str | bytes) -> list[tuple[str, int]]:
    """Report how many characters of text each DROP selector would remove."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for sel in DROP:
        n = sum(len(t.get_text(" ")) for t in soup.select(sel))
        if n:
            out.append((sel, n))
    return out


def parse_html(
    url: str,
    html: str | bytes,
    *,
    tagged_departments: bool = False,
    default_department: str | None = None,
) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    title = norm(soup.title.get_text()) if soup.title else url
    title = re.split(r"\s+[|–-]\s+|:\s+", title)[0] or title  # drop " | Site" suffix
    root_sel, root = next(((s, r) for s in MAIN if (r := soup.select_one(s))), ("(document)", soup))

    for sel in DROP:
        for t in root.select(sel):
            t.decompose()

    # Preserve explicit email targets attached to generic anchor labels like "mail"
    for anchor in root.find_all("a", href=True):
        for address in anchor_emails(anchor["href"]):
            if address not in anchor.get_text(" ").lower():
                anchor.append(NavigableString(f" (email: {address})"))

    fallback_dept = default_department or infer_dept_from_url(url)
    sections: list[Section] = []
    path: list[tuple[int, str, str | None]] = []
    buf: list[str] = []
    links: list[tuple[str, str]] = []
    term: list[str | None] = [None]

    def section_department():
        explicit = next((owner for _, _, owner in reversed(path) if owner), None)
        return explicit or fallback_dept

    def append(text: str, hrefs=()):
        buf.append(text)
        for h in hrefs:
            if target := page_key(h, url):
                links.append((text, target))

    def flush():
        txt = "\n".join(buf).strip()
        if txt:
            heading = " > ".join(h for _, h, _ in path)
            sections.append(
                Section(
                    [h for _, h, _ in path],
                    txt,
                    term[0],
                    course_codes=course_codes(heading + " " + txt),
                    department=section_department(),
                    links=list(dict.fromkeys(links)),
                    contacts=[line for line in buf if EMAIL.search(line)],
                )
            )
        buf.clear()
        links.clear()

    def set_heading(level: int, text: str, node: Tag | None = None):
        flush()
        dept = node.get("data-askiisc-department") if node and tagged_departments else None
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, text, dept))
        if (t := parse_term(text)) is not None:
            term[0] = t
        elif level <= 2:
            term[0] = None  # top-level section boundary resets the active academic term
        if node is not None:
            for anchor in node.find_all("a", href=True):
                if label := norm(anchor.get_text(" ")):
                    append(label, [anchor["href"]])

    def walk(el):
        for ch in el.children:
            if isinstance(ch, NavigableString):
                if t := norm(str(ch)):
                    append(t)
            elif not isinstance(ch, Tag):
                continue
            elif ch.name in HEADINGS:
                set_heading(HEADINGS[ch.name], norm(ch.get_text(" ")), node=ch)
            elif ch.name == "table":
                header, rows = _table_rows(ch)
                flush()

                # Flag venue columns to ignore when parsing courses
                col_is_venue = [
                    any(v in h.lower() for v in VENUE_COLUMNS) if header else False
                    for h in (header or [])
                ]

                for cells, hrefs in rows:
                    if header and len(header) == len(cells):
                        text = "; ".join(f"{h}: {v}" for h, v in zip(header, cells) if v)
                        # Extract course codes exclusively from non-venue cells
                        non_venue_text = " ".join(
                            c for idx, c in enumerate(cells)
                            if not (idx < len(col_is_venue) and col_is_venue[idx])
                        )
                        row_codes = course_codes(non_venue_text)
                    else:
                        text = " | ".join(cells)
                        row_codes = course_codes(cells[0] if cells else "") or course_codes(text)

                    first_col_codes = course_codes(cells[0] if cells else "")
                    label = (path[-1][1] if path else title) + (f" | {cells[0]}" if first_col_codes else "")
                    sections.append(
                        Section(
                            [h for _, h, _ in path[:-1]] + [label],
                            text,
                            term[0],
                            "row",
                            course_codes=row_codes,
                            department=section_department(),
                            links=[(text, target) for href in hrefs if (target := page_key(href, url)) and target != page_key(url)],
                            contacts=[text] if EMAIL.search(text) else [],
                        )
                    )
            elif ch.name == "div" and ch.find("a", href=lambda href: href and href.lower().startswith("mailto:")):
                anchors = ch.find_all("a", href=True, limit=3)
                t = norm(ch.get_text(" "))
                if (
                    len(anchors) == 2 and len(t) <= 350 and len(EMAIL.findall(t)) == 1
                    and not ch.find(["table", "li", *HEADINGS])
                    and sum(bool(anchor_emails(a["href"])) for a in anchors) == 1
                    and sum(bool(page_key(a["href"], url)) for a in anchors) == 1
                ):
                    append(t, [a["href"] for a in anchors])
                else:
                    walk(ch)
            elif ch.name in BLOCKS:
                t = norm(ch.get_text(" "))
                if not t:
                    continue
                if ch.name in ("p", "dt") and COURSE_HEADING_RE.match(t) and len(t) < 160:
                    set_heading(4, t, node=ch)
                elif ch.name == "p" and parse_term(t) and len(t) < 120:
                    set_heading(3, t, node=ch)
                else:
                    append(("- " if ch.name == "li" else "") + t, [a["href"] for a in ch.find_all("a", href=True)])
            elif ch.name == "a" and (t := norm(ch.get_text(" "))):
                append(t, [ch["href"]] if ch.get("href") else [])
            else:
                walk(ch)

    walk(root)
    flush()

    # Remove footer-phrase lines without dropping entire parent sections
    cleaned, removed = [], 0
    for sec in sections:
        if BOILERPLATE_RE.search(sec.heading):
            removed += len(sec.text.splitlines())
            continue
        lines = sec.text.splitlines()
        keep = [line for line in lines if not BOILERPLATE_RE.search(line)]
        removed += len(lines) - len(keep)
        if not keep:
            continue
        sec.text = "\n".join(keep)
        sec.links = [
            (line, target) for line, target in sec.links
            if line in keep or any(line in h for h in sec.heading_path)
        ]
        sec.contacts = [line for line in sec.contacts if line in keep]
        cleaned.append(sec)

    return ParsedPage(url, title, cleaned, root_sel, removed)


def remove_template_sections(
    pages: list[ParsedPage],
    min_pages: int = 10,
    share: float = 0.3,
    similarity: float = 0.5,
) -> dict[str, int]:
    """Site template removal for repetitive sidebars across pages.

    A section is template if its last heading occurs on >= `share` of pages and
    its words overlap (Jaccard >= `similarity`) with the modal text of that heading.
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
    norm_text = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

    for p in pages:
        kept = []
        own = norm_text(p.title)
        for sec in p.sections:
            h = leaf(sec)
            if h in template and norm_text(h) != own:
                w = words(sec.text)
                if w and len(w & modal[h]) / len(w | modal[h]) >= similarity:
                    removed[sec.heading_path[-1]] += 1
                    continue
            kept.append(sec)
        p.sections = kept

    return dict(removed)