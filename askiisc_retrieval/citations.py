"""Citations: what retrieval hands to serving so every answer can point back to its sources.

  build_citations(passages)    one Citation per (page, section); sets passage.marker = "S1", "S2", ...
  format_context(result)       the passages, labelled with their markers, for the generation prompt
  check_answer(answer, result) which markers an answer used, which do not exist, whether it cited at all
  render_sources(result, used) the "Sources:" block shown to the user, for the markers actually used

Only passages that passed the snapshot + scope audit can become citations, so a citation can never point
to a superseded version or a document the user may not see. Graph edges are never cited: graph-added
passages are ordinary chunks and are cited like any other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contract import Citation, Passage, RetrievalResult

MARKER_RE = re.compile(r"\[(S\d+)\]")


def _excerpt(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " ..."


def build_citations(passages: list[Passage]) -> list[Citation]:
    """Markers follow passage order (best first). Chunks from the same page section share a marker."""
    by_key: dict[tuple[str, ...], Citation] = {}
    out: list[Citation] = []
    for p in passages:
        c = p.chunk
        key = (c.source_url, c.section_heading, c.version_id, c.access_scope, c.department)
        cit = by_key.get(key)
        if cit is None:
            cit = Citation(marker=f"S{len(out) + 1}", source_url=c.source_url, title=c.title,
                           section_heading=c.section_heading, version_id=c.version_id,
                           academic_term=c.academic_term, chunk_ids=[], excerpt=_excerpt(c.text),
                           rerank_score=p.rerank_score, via=p.via, access_scope=c.access_scope, department=c.department)
            by_key[key] = cit
            out.append(cit)
        cit.chunk_ids.append(c.chunk_id)
        if p.rerank_score is not None and (cit.rerank_score is None or p.rerank_score > cit.rerank_score):
            cit.rerank_score = p.rerank_score
        p.marker = cit.marker
    return out


def format_context(result: RetrievalResult, max_chars: int | None = None) -> str:
    """Prompt block for the generator. Passages sharing a marker are merged under one header."""
    blocks: dict[str, list[str]] = {}
    for p in result.passages:
        blocks.setdefault(p.marker, []).append(p.chunk.text.strip())
    parts, used = [], 0
    for cit in result.citations:
        term = f" | term {cit.academic_term}" if cit.academic_term else ""
        block = f"[{cit.marker}] {cit.title} > {cit.section_heading}{term}\n" + "\n".join(blocks.get(cit.marker, []))
        if max_chars is not None and used + len(block) > max_chars and parts:
            break                                       # lowest-ranked sources are dropped first
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


@dataclass
class CitationCheck:
    used: list[str]                     # markers in the answer that exist, in order of first use
    unknown: list[str]                  # markers in the answer that do not exist (hallucinated)
    cited: bool                         # the answer cites at least one real source
    sources: list[Citation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.cited and not self.unknown


def check_answer(answer: str, result: RetrievalResult) -> CitationCheck:
    valid = {c.marker: c for c in result.citations}
    used, unknown = [], []
    for m in MARKER_RE.findall(answer):
        bucket = used if m in valid else unknown
        if m not in bucket:
            bucket.append(m)
    return CitationCheck(used, unknown, bool(used), [valid[m] for m in used])


def render_sources(result: RetrievalResult, used: list[str] | None = None) -> str:
    """User-facing source list. With `used`, only the sources the answer actually cites."""
    keep = [c for c in result.citations if used is None or c.marker in used]
    lines = []
    for c in keep:
        term = f", {c.academic_term}" if c.academic_term else ""
        lines.append(f"[{c.marker}] {c.title} > {c.section_heading}{term}\n     {c.source_url}")
    return "Sources:\n" + "\n".join(lines) if lines else ""


def result_to_dict(result: RetrievalResult, include_text: bool = True) -> dict:
    """JSON-ready retrieval result: what serving consumes and what eval stores per question."""
    from dataclasses import asdict
    return {
        "query": result.query,
        "departments": list(result.departments),
        "department_status": result.department_status,
        "graph_proofs": result.graph_proofs,
        "connection_status": result.connection_status,
        "policy_trace": result.policy_trace,
        "snapshot_id": result.snapshot_id,
        "target_term": result.target_term,
        "evidence_sufficient": result.evidence_sufficient,
        "rerank_top": result.rerank_top,
        "degraded": result.degraded,
        "degraded_reason": result.degraded_reason,
        "citations": [asdict(c) for c in result.citations],
        "passages": [{"marker": p.marker, "chunk_id": p.chunk.chunk_id, "source_url": p.chunk.source_url,
                      "version_id": p.chunk.version_id,
                      "section_heading": p.chunk.section_heading, "department": p.chunk.department,
                      "access_scope": p.chunk.access_scope, "via": p.via, "vector_score": round(p.score, 4),
                      "rerank_score": None if p.rerank_score is None else round(p.rerank_score, 4),
                      "graph_path": p.graph_path, "graph_support": p.graph_support,
                      **({"text": p.chunk.text} if include_text else {})}
                     for p in result.passages],
        "timings_ms": result.timings_ms,
    }
