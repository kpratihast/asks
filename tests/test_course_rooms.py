"""Tests for extracting course codes and mapping them to their assigned room numbers and venues."""

import re
from bs4 import BeautifulSoup
import pytest

COURSE_RE = re.compile(
    r"\b(?!(?:CDS|CSA|ECE|DESE|RBCCPS|SERC)\b)"
    r"([A-Z]{2}|E[0-9])\s?-?\s?([1-3]\d{2})\b",
    re.I,
)

ROOM_TEXT_RE = re.compile(
    r"(?:in|at|room|venue|hall|lh|cr)\s*[:.-]?\s*"
    r"((?:Room\s*\d+|LH\s*[-]?\d+|CR\s*[-]?\d+|CDS\s*\d+|CSA\s*\d+|Auditorium[^\.,;\n]*|\b[A-Z0-9]+-[0-9]+))",
    re.I,
)

VENUE_HEADERS = {"venue", "room", "hall", "location", "classroom"}
COURSE_HEADERS = {"course", "code", "subject", "course no", "course #"}


def extract_course_rooms_from_table(html: str) -> dict[str, str]:
    """Extract course-to-room mappings from schedule HTML tables."""
    soup = BeautifulSoup(html, "html.parser")
    mappings = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        header_cells = rows[0].find_all(["th", "td"])
        col_types = {}

        for idx, cell in enumerate(header_cells):
            text = cell.get_text(strip=True).lower()
            if any(h in text for h in COURSE_HEADERS):
                col_types[idx] = "course"
            elif any(v in text for v in VENUE_HEADERS):
                col_types[idx] = "room"

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            course_code = None
            room = None

            for idx, cell in enumerate(cells):
                cell_text = cell.get_text(" ", strip=True)
                col_role = col_types.get(idx)

                if col_role == "course":
                    m = COURSE_RE.search(cell_text)
                    if m:
                        course_code = f"{m.group(1).upper()} {m.group(2)}"
                elif col_role == "room":
                    room = cell_text
                elif not col_types:
                    m = COURSE_RE.search(cell_text)
                    if m and not course_code:
                        course_code = f"{m.group(1).upper()} {m.group(2)}"
                    elif ROOM_TEXT_RE.search(cell_text) and not room:
                        room = cell_text

            if course_code and room:
                mappings[course_code] = room

    return mappings


def extract_course_rooms_from_text(text: str) -> dict[str, str]:
    """Extract course-to-room mappings from sentence-level text."""
    mappings = {}
    sentences = re.split(r"[.\n;]", text)

    for sentence in sentences:
        course_match = COURSE_RE.search(sentence)
        room_match = ROOM_TEXT_RE.search(sentence)

        if course_match and room_match:
            code = f"{course_match.group(1).upper()} {course_match.group(2)}"
            venue = room_match.group(1).strip()
            mappings[code] = venue

    return mappings


# --- Pytest Test Cases ---

def test_extract_course_and_rooms_from_schedule_table():
    sample_html = """
    <table>
      <thead>
        <tr>
          <th>Course No</th>
          <th>Course Title</th>
          <th>Venue / Room</th>
          <th>Credits</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>E0 249</td>
          <td>Approximation Algorithms</td>
          <td>Room 201 (CSA Hall)</td>
          <td>3:1</td>
        </tr>
        <tr>
          <td>DS 252</td>
          <td>Cloud Computing</td>
          <td>CDS 202</td>
          <td>3:1</td>
        </tr>
        <tr>
          <td>NE 201</td>
          <td>Micro and Macro Nano-systems</td>
          <td>CR 2</td>
          <td>3:0</td>
        </tr>
        <tr>
          <td>CP 214</td>
          <td>Design Methodology</td>
          <td>LH-3</td>
          <td>2:1</td>
        </tr>
      </tbody>
    </table>
    """
    results = extract_course_rooms_from_table(sample_html)

    assert results["E0 249"] == "Room 201 (CSA Hall)"
    assert results["DS 252"] == "CDS 202"
    assert results["NE 201"] == "CR 2"
    assert results["CP 214"] == "LH-3"


def test_extract_course_and_rooms_from_paragraphs():
    sample_text = (
        "Introductory lecture for DS 221 will meet in LH 1 on Monday. "
        "Interviews for E0 249 will be conducted at Room 101. "
        "Lab demonstrations for CP 214 take place in CR 2."
    )
    results = extract_course_rooms_from_text(sample_text)

    assert results["DS 221"] == "LH 1"
    assert results["E0 249"] == "Room 101"
    assert results["CP 214"] == "CR 2"


def test_room_labels_not_mislabeled_as_courses():
    sample_html = """
    <table>
      <tr><th>Course</th><th>Venue</th></tr>
      <tr><td>E0 225</td><td>Room 101</td></tr>
    </table>
    """
    results = extract_course_rooms_from_table(sample_html)

    assert "Room 101" not in results
    assert "E0 225" in results
    assert results["E0 225"] == "Room 101"