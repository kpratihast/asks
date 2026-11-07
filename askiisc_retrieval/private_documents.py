"""Operator-controlled document classification; never infer permissions from document text."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re

from .departments import DEFAULT, SHARED, mentioned, validate_codes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class DocumentCatalog:
    def __init__(self, manifest: Path | None = None):
        self.manifest = manifest or Path(os.environ.get('ASKIISC_PRIVATE_MANIFEST', ROOT / 'config' / 'restricted_documents.json'))
        self.documents = []
        if self.manifest.exists():
            raw = json.loads(self.manifest.read_text(encoding='utf-8'))
            self.documents = raw['documents']
        ids = set()
        for d in self.documents:
            if not re.fullmatch(r'[a-z0-9-]{1,80}', d['id']) or d['id'] in ids:
                raise ValueError('Invalid or duplicate private document id')
            ids.add(d['id'])
            if d['access_scope'] not in ('student', 'faculty'):
                raise ValueError('Private documents need an explicit student or faculty access_scope')
            validate_codes([d.get('department', DEFAULT)])
            if d.get('doc_type') not in ('schedule', 'calendar', 'policy', 'page'):
                raise ValueError('Invalid private document type')
            for p in d.get('gate_patterns', []):
                re.compile(p, re.I)

    def path(self, document, key='original'):
        root = (self.manifest.parent.parent / '.local' / 'private').resolve()
        path = (self.manifest.parent.parent / document[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Private document paths must remain in .local/private')
        return path

    def find(self, document_id):
        return next((d for d in self.documents if d['id'] == document_id), None)

    def matching(self, question):
        from .query_analysis import public_entrance_exam_context
        entrance = public_entrance_exam_context(question)
        departments = mentioned(question)
        years = set(re.findall(r'\b20\d{2}\b', question))
        return [d for d in self.documents
                if (not departments or d.get('department', DEFAULT) in (*departments, SHARED))
                and not (entrance and d.get('query_topic') in ('course_schedule', 'course_examination'))
                and (not years or not d.get('academic_term') or d['academic_term'][:4] in years)
                and any(re.search(p, question, re.I) for p in d.get('gate_patterns', []))]

    def denied_scopes(self, question, allowed):
        """Login hint only. Retrieval enforces scopes for every query, independent of patterns."""
        return sorted({d['access_scope'] for d in self.matching(question) if d['access_scope'] not in allowed})

    def page_inputs(self):
        from .ingest_stub import PageInput
        out = []
        for d in self.documents:
            source, extracted = self.path(d), self.path(d, 'extracted')
            if not source.is_file() or not extracted.is_file():
                raise ValueError(f'Private document {d["id"]} is missing. Import it before rebuilding.')
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            if checksum != d['sha256']:
                raise ValueError(f'Private document {d["id"]} changed; review its extraction before ingestion.')
            if hashlib.sha256(extracted.read_bytes()).hexdigest() != d['extracted_sha256']:
                raise ValueError(f'Private document {d["id"]} extraction changed; update its reviewed checksum.')
            out.append(PageInput('/v1/documents/' + d['id'], extracted.read_bytes(), d['doc_type'],
                                 d['access_scope'], d.get('department', DEFAULT), version_tag=checksum))
        return out


def pdf_to_html(path: Path, title: str, term: str):
    """Offline PDF extraction. No upload or model request is made."""
    from pypdf import PdfReader
    blocks = []
    layout_pages = []
    reader = PdfReader(path)
    if len(reader.pages) > 200:
        raise ValueError('Split PDFs longer than 200 pages into logical sections before import')
    total_chars = 0
    for n, page in enumerate(reader.pages, 1):
        text = page.extract_text(extraction_mode='layout') or ''
        if len(text.strip()) < 40:
            raise ValueError(f'Page {n} needs OCR and manual review before ingestion')
        total_chars += len(text)
        if total_chars > 2 * 1024 * 1024:
            raise ValueError('PDF text exceeds the import budget; split it into logical sections')
        layout_pages.append(text)
        # Keep layout rows separate: each timetable row retains its own date and session.
        for line in text.splitlines():
            if line.strip():
                blocks.append('<p>' + html.escape(line.strip()) + '</p>')
    from .parse import parse_term
    from .scheduling import exam_table_html
    academic_term = parse_term(term)
    structured = exam_table_html(layout_pages, title, term, academic_term) if academic_term else None
    if structured:
        return structured
    return '<html><head><title>'  + html.escape(title) + '</title></head><body><main><h1>' + html.escape(title) + '</h1><h2>' + html.escape(term) + '</h2>' + ''.join(blocks) + '</main></body></html>'
