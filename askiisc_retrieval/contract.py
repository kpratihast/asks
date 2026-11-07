"""The interface between ingestion, retrieval and serving.

Share this file with the ingestion team: every chunk they write to Qdrant must carry exactly the
fields of `ChunkPayload`, and serving consumes `RetrievalResult`. If a field needs to change, change it
here together, rather than each side patching around it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Access levels a passage can carry. "staff" is the older name for admin-only content and is still accepted.
SCOPES = ("public", "student", "faculty", "admin", "staff")
DOC_TYPES = ("policy", "admissions", "catalogue", "schedule", "handbook", "people", "research",
             "contact", "about", "calendar", "resources", "events", "news", "page")


@dataclass
class ChunkPayload:
    """One record in Qdrant. The vector is stored alongside; everything else is payload."""
    chunk_id: str                 # stable within a version: hash(version_id, ordinal)
    document_id: str              # stable per source URL across versions
    version_id: str               # changes whenever the extracted content changes
    source_url: str
    title: str                    # document title
    section_heading: str          # citation location within the document
    text: str                     # the citable text (what the user sees in an excerpt)
    access_scope: str             # public | student | faculty | admin (staff = old name for admin)
    doc_type: str                 # see DOC_TYPES
    department: str               # e.g. "CDS"
    academic_term: Optional[str]  # per SECTION, e.g. "2026-AUG", "2026-JAN"; None if not term-bound
    course_codes: list[str]       # normalised, e.g. ["DS252"]; empty if none
    embedding_model: str          # e.g. "BAAI/bge-m3"
    ordinal: int                  # position within the version
    links: list[str] = field(default_factory=list)  # HTTP targets written in this passage
    contacts: list[str] = field(default_factory=list)  # source lines that publish an email

    def to_dict(self) -> dict:
        return asdict(self)

    def embedding_input(self) -> str:
        """What gets embedded: title + heading give short chunks their context."""
        return f"{self.title} | {self.section_heading}\n{self.text}"


@dataclass
class Passage:
    chunk: ChunkPayload
    score: float                  # vector similarity; graph passages carry their seed's score
    via: str = "vector"           # vector | course_code | people | graph
    graph_path: Optional[str] = None   # e.g. "Course:DS252 -TAUGHT_BY-> Person:yogesh simmhan"
    rerank_score: Optional[float] = None   # cross-encoder relevance in [0, 1]; None if reranking is off
    marker: Optional[str] = None           # citation marker ("S1") this passage is cited under
    graph_support: dict = field(default_factory=dict)  # relation, endpoints, question coverage; selection only


@dataclass
class Citation:
    """One citable source = one page section in the active snapshot. What the user sees as [S1]."""
    marker: str                   # "S1", "S2", ... in rank order
    source_url: str
    title: str
    section_heading: str          # location within the page
    version_id: str               # exact version cited (FR-3: answers trace to a version)
    academic_term: Optional[str]
    chunk_ids: list[str]          # chunks behind this citation
    excerpt: str                  # short quote for display / verification
    rerank_score: Optional[float] = None
    via: str = "vector"
    access_scope: str = "public"
    department: str = ""


@dataclass
class RetrievalResult:
    query: str
    passages: list[Passage]
    snapshot_id: str
    degraded: bool = False                 # True when graph expansion failed and we fell back (FR-9)
    degraded_reason: Optional[str] = None
    target_term: Optional[str] = None      # term the query was resolved to, for transparency
    timings_ms: dict = field(default_factory=dict)
    # Abstention signal for serving (FR-11): best reranker score, and whether it clears the calibrated
    # threshold. None when reranking is off/failed or the threshold is not calibrated yet.
    rerank_top: Optional[float] = None
    evidence_sufficient: Optional[bool] = None
    rerank_error: Optional[str] = None
    citations: list = field(default_factory=list)   # list[Citation], built from `passages`

    graph_proofs: list[dict] = field(default_factory=list)
    connection_status: Optional[str] = None
    policy_trace: dict = field(default_factory=dict)
    departments: tuple[str, ...] = ()
    department_status: str | None = None

    @property
    def top_score(self) -> float:
        return max((p.score for p in self.passages if p.via != "graph"), default=0.0)
