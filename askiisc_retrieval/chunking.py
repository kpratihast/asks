"""Sections -> ChunkPayloads. Sentence-aligned windows that never cross a heading, so every chunk has
exactly one citation location. Table rows are already small and stay one chunk each."""
from __future__ import annotations

import hashlib
import re

from .config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS
from .contract import ChunkPayload
from .parse import ParsedPage, course_codes

SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])|\n+")


def h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def windows(text: str, size: int, overlap: int) -> list[str]:
    units = [u.strip() for u in SENT.split(text) if u and u.strip()]
    out, cur, n = [], [], 0
    for u in units:
        w = len(u.split())
        if cur and n + w > size:
            out.append(" ".join(cur))
            back, bn = [], 0
            for x in reversed(cur):                 # carry whole sentences as overlap
                if bn + len(x.split()) > overlap:
                    break
                back.insert(0, x)
                bn += len(x.split())
            cur, n = back, bn
        cur.append(u)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_page(page: ParsedPage, *, version_id: str, document_id: str, scope: str, doc_type: str,
               department: str, embedding_model: str) -> list[ChunkPayload]:
    chunks: list[ChunkPayload] = []
    for sec in page.sections:
        pieces = [sec.text] if sec.kind == "row" else windows(sec.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
        for piece in pieces:
            ordinal = len(chunks)
            codes = sec.course_codes if sec.kind == "row" else sorted(
                set(course_codes(sec.heading)) | set(course_codes(piece)))
            chunks.append(ChunkPayload(
                chunk_id=h(f"{version_id}:{ordinal}", 32), document_id=document_id, version_id=version_id,
                source_url=page.url, title=page.title, section_heading=sec.heading, text=piece,
                access_scope=scope, doc_type=doc_type, department=sec.department or department,
                academic_term=sec.academic_term, course_codes=codes,
                embedding_model=embedding_model, ordinal=ordinal))
    return chunks
