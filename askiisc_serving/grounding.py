"""Claim-level grounding for generated answers (FR-11).

The marker check in generator.verify proves that every [S#] exists. This module checks the content:

* A refusal phrase at the start ("I could not find ...") makes the whole answer a refusal, so facts padded
  after it are not presented as an answer.
* A refusal phrase after facts ("... I could not find the fee") is a self-contradiction; the caller returns
  source excerpts instead of contradictory text.
* A stated count followed by a list of a different length ("three types:" then six items) is also a
  self-contradiction.
* Every claim (sentence or list item) must be supported by the passages it cites: each number and each
  specific name or term in the claim must appear in those passages. Unsupported claims are removed. A claim
  without its own marker is checked against the markers of its paragraph, or else of the whole answer.

This is a lexical check. It catches invented names, terms and numbers; it cannot prove that a number that
does appear in the passage was given the right label (that needs an entailment model).
"""
from __future__ import annotations

import re

MARKER_RE = re.compile(r"\[(S\d+)\]")

# Phrases a model uses to say the evidence does not answer the question.
_REFUSAL = re.compile(
    r"\b(?:i\s+)?(?:could\s*not|couldn['’]t|cannot|can['’]t|was\s+unable\s+to|am\s+unable\s+to|did\s+not|didn['’]t)\s+"
    r"(?:find|locate|identify|determine)\b"
    r"|\b(?:is|are|was|were)\s+not\s+(?:explicitly\s+|specifically\s+|directly\s+)?"
    r"(?:listed|mentioned|stated|specified|available|provided|given|included|found)\s+(?:in|on|by|within|from)\s+"
    r"(?:the\s+)?(?:available|provided|given|retrieved|cited|above|these|those|this|iisc)?\s*"
    r"(?:iisc\s+)?(?:sources?|evidence|documents?|information|passages?|context|website)"
    r"|\bno\s+(?:specific\s+|relevant\s+|such\s+)?information\s+(?:is\s+available\s+)?(?:on|about|regarding|for)\b"
    r"|\b(?:the\s+)?(?:available|provided|given|retrieved)\s+(?:sources?|evidence|documents?|information|passages?)\s+"
    r"(?:do(?:es)?\s+not|don['’]t|doesn['’]t)\s+(?:contain|mention|include|provide|specify|state|list)\b"
    r"|\bnot\s+(?:explicitly\s+)?(?:mentioned|listed|stated)\s+in\s+the\s+(?:available|provided)\b",
    re.I)

# Capitalised words that are ordinary English or too generic to identify a fact.
_GENERIC = set("""
a an the this these those that there here it its they them their he his she her we our you your i
however according note also and or but so for in on at of to by with from as is are was were be been
yes no all some each both other any many most more few several such following example examples
students student department departments institute institutes programme programmes program programs
course courses faculty research information source sources available website page pages section list
lab labs laboratory centre center fee fees scholarship scholarships fellowship fellowships admission
admissions application applications degree degrees area areas topic topics office unit details detail
prof professor professors dr sir mr ms mrs rs inr total annual monthly semester semesters year years
indian science iisc bangalore bengaluru india
""".split())

_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[*\-•\d])")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
_CAP_TOKEN = re.compile(r"[A-Za-z][A-Za-z.()'’&-]*")


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def _passage_text(hit: dict) -> str:
    fields = [hit.get(k) for k in ("title", "section", "programme", "department", "instructor", "location", "text")]
    fields += list(hit.get("graph_paths", []) or [])
    for support in hit.get("graph_support", []) or []:
        fields += [str(v) for v in support.values() if isinstance(v, (str, int, float))]
    return "\n".join(str(f) for f in fields if f)


def refusal_position(answer: str) -> str | None:
    """'start' if the answer opens by saying the sources lack the answer, 'later' if it says so after
    stating facts, else None."""
    text = MARKER_RE.sub("", answer).strip()
    match = _REFUSAL.search(text)
    if not match:
        return None
    first_paragraph = re.split(r"\n\s*\n", text, maxsplit=1)[0]
    first_sentence = _SENTENCE_SPLIT.split(first_paragraph, maxsplit=1)[0]
    return "start" if _REFUSAL.search(first_sentence) else "later"


def _terms(claim: str) -> list[str]:
    """Specific names and terms: maximal runs of capitalised tokens, without generic words."""
    body = _LIST_PREFIX.sub("", MARKER_RE.sub(" ", claim))
    terms, run, start = [], [], 0
    tokens = _CAP_TOKEN.findall(body)

    def close(end):
        # A lone capitalised ordinary word that opens the claim ("Taught by ...") is just sentence case.
        lone_opener = start == 0 and len(run) == 1 and not any(c.isupper() for c in run[0][1:])
        if run and not lone_opener:
            terms.append(" ".join(run))

    for i, token in enumerate(tokens):
        word = token.strip(".()'’&-")
        capitalised = bool(word) and (word[0].isupper() or any(c.isupper() for c in word[1:]))
        if capitalised and word.lower() not in _GENERIC:
            if not run:
                start = i
            run.append(word)
            continue
        close(i)
        run = []
    close(len(tokens))
    return [t for t in terms if len(_alnum(t)) >= 3]


def _claims(answer: str) -> list[tuple[int, str]]:
    """(paragraph index, claim) for each sentence or list item."""
    out = []
    for p_index, paragraph in enumerate(re.split(r"\n\s*\n", answer)):
        for line in paragraph.split("\n"):
            if not line.strip():
                continue
            parts = [line] if _LIST_PREFIX.match(line) else _SENTENCE_SPLIT.split(line)
            out.extend((p_index, part) for part in parts if part.strip())
    return out


def unsupported_claims(answer: str, selected: list[dict], question: str = "") -> list[str]:
    """Claims whose numbers or specific terms are not in the passages they cite. Words and numbers that
    only restate the question are not new facts, and a lead-in line ending in a colon is not a claim."""
    asked_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    asked_numbers = _numbers(question)
    passages = {h["prompt_marker"]: _passage_text(h) for h in selected}
    answer_markers = [m for m in MARKER_RE.findall(answer) if m in passages]
    paragraph_markers: dict[int, list[str]] = {}
    for p_index, paragraph in enumerate(re.split(r"\n\s*\n", answer)):
        paragraph_markers[p_index] = [m for m in MARKER_RE.findall(paragraph) if m in passages]
    failed = []
    for p_index, claim in _claims(answer):
        if MARKER_RE.sub("", claim).strip().endswith(":"):
            continue
        own = [m for m in MARKER_RE.findall(claim) if m in passages]
        markers = own or paragraph_markers.get(p_index) or answer_markers
        if not markers:
            continue                      # marker presence is the citation check's job
        evidence = "\n".join(passages[m] for m in markers)
        words, numbers = set(re.findall(r"[a-z0-9]+", evidence.lower())), _numbers(evidence)
        claim_numbers = {n for n in _numbers(_LIST_PREFIX.sub("", MARKER_RE.sub(" ", claim)))
                         if not re.fullmatch(r"[1-9]", n)}       # small counts and list numbers carry no fact
        missing_numbers = claim_numbers - numbers - asked_numbers
        # Every word of a specific term must occur in the cited evidence (word order may be paraphrased).
        missing_terms = [t for t in _terms(claim)
                         if any(w not in words and w not in asked_words
                                for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 1)]
        if missing_numbers or missing_terms:
            failed.append(claim.strip())
    return failed


def remove_claims(answer: str, claims: list[str]) -> str:
    lines = answer.split("\n")
    for claim in claims:
        for i, line in enumerate(lines):
            if line.strip() == claim:
                lines[i] = None            # a whole list item or line: drop the line itself
                break
            if claim in line:
                lines[i] = line.replace(claim, "", 1)
                break
        lines = [line for line in lines if line is not None]
    kept = "\n".join(lines)
    kept = re.sub(r"[ \t]+\n", "\n", kept)
    kept = re.sub(r"\n{3,}", "\n\n", kept)
    return kept.strip()


_COUNT_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_COUNTED = re.compile(
    r"\b(" + "|".join(list(_COUNT_WORDS)[2:]) + r"|[2-9]|1[0-2])\s+(?:\w+\s+){0,2}?"
    r"(?:types?|kinds?|programm?es?|programs?|degrees?|courses?|topics?|areas?|labs?|items?|options?|"
    r"categories|category|ways?|steps?|streams?|tracks?|groups?|stages?|phases?|components?|requirements?)\b", re.I)


def count_mismatch(answer: str) -> str | None:
    """A stated count ("three types of degrees") followed by a list of a different length is a
    self-contradiction. Returns a short description, or None."""
    lines = MARKER_RE.sub("", answer).split("\n")
    for i, line in enumerate(lines):
        m = _COUNTED.search(line)
        if not m:
            continue
        stated = m.group(1).lower()
        stated = _COUNT_WORDS.get(stated, int(stated) if stated.isdigit() else None)
        # the list must start within the next three lines (a lead-in may sit between)
        j, items = i + 1, 0
        while j < len(lines) and j <= i + 3 and not _LIST_PREFIX.match(lines[j]):
            j += 1
        while j < len(lines) and (_LIST_PREFIX.match(lines[j]) or not lines[j].strip()):
            items += bool(_LIST_PREFIX.match(lines[j]))
            if not lines[j].strip() and j + 1 < len(lines) and not _LIST_PREFIX.match(lines[j + 1]):
                break
            j += 1
        if items and stated and items != stated:
            return f"states {stated} but lists {items}"
    return None


def has_content(answer: str) -> bool:
    return bool(MARKER_RE.search(answer)) and any(c.isalnum() for c in MARKER_RE.sub("", answer))


def ground(answer: str, selected: list[dict], abstain_text: str, question: str = "") -> tuple[str, dict]:
    """Apply the refusal, contradiction and claim-support rules. Returns (answer, info for verification)."""
    position = refusal_position(answer)
    if position == "start":
        return abstain_text, {"refusal_reason": "model_abstained", "padding_removed": answer.strip() != abstain_text}
    if position == "later":
        return answer, {"self_contradiction": True}
    mismatch = count_mismatch(answer)
    if mismatch:
        return answer, {"self_contradiction": True, "count_mismatch": mismatch}
    failed = unsupported_claims(answer, selected, question)
    if not failed:
        return answer, {"unsupported_claims": []}
    claims = [c.strip() for _, c in _claims(answer) if not MARKER_RE.sub("", c).strip().endswith(":")]
    opening_unsupported = bool(claims) and claims[0] in failed
    kept = remove_claims(answer, failed)
    if opening_unsupported or not has_content(kept):
        # The direct answer is not in the evidence: refuse rather than present the remaining padding.
        return abstain_text, {"unsupported_claims": failed, "refusal_reason": "unsupported_claims"}
    return kept, {"unsupported_claims": failed}
