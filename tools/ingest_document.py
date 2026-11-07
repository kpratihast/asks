#!/usr/bin/env python3
"""Import an operator-selected student/professor document into a fresh local index."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MAX_FILE = 32 * 1024 * 1024
MAX_TEXT = 2 * 1024 * 1024
ROLES = {'student': 'student', 'prof': 'faculty', 'professor': 'faculty', 'faculty': 'faculty'}
TEXT_TYPES = {'.txt', '.md', '.html', '.htm'}
IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.webp'}


def read_file(path, limit):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'Input must be a regular file no larger than {limit // 1024} KiB')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Input exceeded its size limit while reading')
    return path, data


def text_html(data, suffix, title, term):
    text = data.decode('utf-8-sig')
    if suffix in {'.html', '.htm'}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        for tag in soup(['script', 'style', 'iframe', 'object', 'embed', 'form', 'svg']):
            tag.decompose()
        body = str(soup.body or soup)
    else:
        body = ''.join('<p>' + html.escape(p) + '</p>' for p in text.splitlines() if p.strip())
    return ('<html><head><title>' + html.escape(title) + '</title></head><body><main><h1>' +
            html.escape(title) + '</h1>' + ('<h2>' + html.escape(term) + '</h2>' if term else '') +
            body + '</main></body></html>')


def extract(original, extracted, title, term):
    if extracted:
        path, data = read_file(extracted, MAX_TEXT)
        if path.suffix.lower() not in TEXT_TYPES:
            raise ValueError('--extracted must be reviewed UTF-8 HTML, TXT or Markdown')
        return text_html(data, path.suffix.lower(), title, term)
    if original.suffix.lower() == '.pdf':
        # Parsing happens in a disposable child, never in the running chat process.
        command = [sys.executable, str(Path(__file__).resolve()), '--extract-pdf', str(original), title, term]
        try:
            result = subprocess.run(command, check=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise ValueError('PDF extraction timed out. Supply a reviewed --extracted file.') from None
        except subprocess.CalledProcessError:
            raise ValueError('PDF needs OCR/review or cannot be read. Supply --extracted reviewed.html.') from None
        if len(result.stdout) > MAX_TEXT:
            raise ValueError('Extracted text is too large; split the document into logical sections')
        return result.stdout.decode('utf-8')
    if original.suffix.lower() in IMAGE_TYPES:
        raise ValueError('Images need a reviewed transcription: pass --extracted reviewed.html')
    _, data = read_file(original, MAX_TEXT)
    return text_html(data, original.suffix.lower(), title, term)


def build_index(base, target, manifest, root):
    env = {**os.environ, 'ASKIISC_PRIVATE_MANIFEST': str(manifest), 'ASKIISC_DATA_DIR': str(target),
           'EMAIL_AUTH_ENABLED': '0'}
    command = [sys.executable, str(ROOT / 'tools/import_private.py'),
               '--base-data-dir', str(base), '--output-data-dir', str(target)]
    subprocess.run(command, env=env, cwd=root, check=True)


def ingest(args, root=ROOT, builder=build_index):
    root = Path(root).resolve()
    source, original_bytes = read_file(args.file, MAX_FILE)
    suffix = source.suffix.lower()
    if suffix not in TEXT_TYPES | IMAGE_TYPES | {'.pdf'}:
        raise ValueError('Supported originals: PDF, TXT, Markdown, HTML, PNG, JPG, JPEG and WebP')
    title = (args.title or source.stem.replace('_', ' ')).strip()
    if not title or len(title) > 200 or any(ord(c) < 32 for c in title):
        raise ValueError('Use a nonempty title of at most 200 characters, without control characters')
    doc_id = args.id or re.sub(r'[^a-z0-9]+', '-', source.stem.lower()).strip('-')[:64]
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', doc_id):
        raise ValueError('Set --id to 1–80 lowercase letters, digits or hyphens, starting with a letter/digit')
    term = args.term or ''
    academic_term = ''
    if term:
        from askiisc_retrieval.parse import parse_term
        if re.fullmatch(r'20\d{2}-(AUG|JAN|SUM)', term):
            year, season = term.split('-')
            academic_term = term
            term = {'AUG': 'Aug-Dec ', 'JAN': 'Jan-Apr ', 'SUM': 'May-Jul '}[season] + year
        else:
            academic_term = parse_term(term)
        if not academic_term:
            raise ValueError('Use --term 2026-AUG, 2027-JAN, or a range such as "Aug-Dec 2026"')
    profile = json.loads((root / 'config/local.json').read_text())
    base = Path(args.base_data_dir or os.environ.get('ASKIISC_DATA_DIR') or profile['ASKIISC_DATA_DIR']).expanduser().resolve()
    if not (base / 'snapshot.json').is_file():
        raise ValueError('Base index has no snapshot.json; set --base-data-dir')
    digest = hashlib.sha256(original_bytes).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = Path(args.output_data_dir or root / 'data' / f'import-{doc_id}-{stamp}').expanduser().resolve()
    if target.exists() or target == base or target.is_relative_to(base) or base.is_relative_to(target):
        raise ValueError('Output must be a fresh directory separate from the active/base index')
    local = root / '.local'
    local.mkdir(mode=0o700, exist_ok=True)
    private = local / 'private'
    private.mkdir(mode=0o700, exist_ok=True)
    if private.is_symlink() or local.is_symlink():
        raise ValueError('Private storage must not be a symlink')
    manifest = root / 'config/restricted_documents.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    created = []
    candidate = None
    promoted = False
    with (local / 'document-import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = manifest.read_bytes() if manifest.exists() else b'{"documents":[]}'
        catalog = json.loads(before)
        previous = next((d for d in catalog['documents'] if d['id'] == doc_id), None)
        if previous and not args.replace:
            raise ValueError('Document ID already exists; use --replace to explicitly revise it')
        def save(name, content):
            path = private / name
            if path.is_symlink():
                raise ValueError('Private input path must not be a symlink')
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError('Existing immutable input differs; choose a different document ID')
            else:
                with path.open('xb') as stream:
                    path.chmod(0o600)
                    stream.write(content)
                created.append(path)
            return path
        try:
            original = save(f'{doc_id}-{digest[:16]}{suffix}', original_bytes)
            extracted = extract(original, args.extracted, title, term).encode('utf-8')
            if len(extracted) > MAX_TEXT:
                raise ValueError('Extracted text is too large; split the document into logical sections')
            from askiisc_retrieval.parse import parse_html
            from askiisc_serving.security import instruction_attack
            parsed = parse_html('/v1/documents/' + doc_id, extracted)
            text = '\n'.join(s.text for s in parsed.sections)
            if len(text.strip()) < 100:
                raise ValueError('Too little extractable content (minimum 100 characters); supply a reviewed transcription')
            if instruction_attack(text) or instruction_attack(title):
                raise ValueError('Extraction contains instruction overrides; review and remove them before ingestion')
            extracted_digest = hashlib.sha256(extracted).hexdigest()
            html_path = save(f'{doc_id}-{extracted_digest[:16]}.html', extracted)
            topics = args.topic or []
            if any(not t.strip() or len(t) > 150 for t in topics) or len(topics) > 20:
                raise ValueError('Use at most 20 nonempty --topic phrases, each at most 150 characters')
            patterns = [r'(?<!\w)' + re.escape(t.strip()) + r'(?!\w)' for t in [title, doc_id, *topics]]
            doc = {'id': doc_id, 'title': title, 'access_scope': ROLES[args.role], 'doc_type': args.doc_type,
                   'academic_term': academic_term, 'original': str(original.relative_to(root)),
                   'extracted': str(html_path.relative_to(root)), 'sha256': digest,
                   'extracted_sha256': extracted_digest, 'gate_patterns': patterns,
                   'classification': 'Explicit operator CLI role selection; review extraction before deployment'}
            documents = [d for d in catalog['documents'] if d['id'] != doc_id] + [doc]
            fd, name = tempfile.mkstemp(prefix='.document-import-', suffix='.json', dir=manifest.parent)
            candidate = Path(name)
            with os.fdopen(fd, 'w') as stream:
                json.dump({**catalog, 'documents': documents}, stream, indent=2)
                stream.write('\n')
            from askiisc_retrieval.private_documents import DocumentCatalog
            DocumentCatalog(candidate).page_inputs()
            with tempfile.TemporaryDirectory(prefix='.document-build-', dir=target.parent) as tmp:
                staged = Path(tmp) / 'index'
                builder(base, staged, candidate, root)
                chunks = [json.loads(line) for line in (staged / 'chunks.jsonl').read_text().splitlines()]
                added = [c for c in chunks if c['source_url'] == '/v1/documents/' + doc_id]
                if not added or any(c['access_scope'] != ROLES[args.role] for c in added):
                    raise ValueError('Import did not produce correctly classified chunks; the active setup is unchanged')
                if target.exists() or (manifest.read_bytes() if manifest.exists() else b'{"documents":[]}') != before:
                    raise ValueError('Output or catalog changed during ingestion; retry after reviewing the other change')
                staged.rename(target)
                try:
                    candidate.replace(manifest)
                except BaseException:
                    shutil.rmtree(target)
                    raise
                promoted = True
            report = {'document_id': doc_id, 'access_scope': ROLES[args.role], 'document_chunks': len(added),
                      'total_chunks': len(chunks), 'data_dir': str(target), 'review_extraction': str(html_path),
                      'active_local_config_changed': False, 'aws_deployed': False,
                      'test_command': f'ASKIISC_DATA_DIR={shlex.quote(str(target))} ./run-local.sh --port 8102',
                      'deploy_command': './deploy-aws.sh --build --data-dir ' + shlex.quote(str(target))}
            (target / 'document-import.json').write_text(json.dumps(report, indent=2) + '\n')
            return report
        finally:
            if candidate:
                candidate.unlink(missing_ok=True)
            if not promoted:
                for path in reversed(created):
                    path.unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file', required=True, type=Path, help='Original local document')
    p.add_argument('--role', required=True, choices=ROLES, help='student or prof (faculty/professor are aliases)')
    p.add_argument('--id', help='Stable document ID; defaults to a slug of the original filename')
    p.add_argument('--title', help='Human-readable source title')
    p.add_argument('--term', help='Optional academic term, e.g. 2026-AUG')
    p.add_argument('--doc-type', choices=['page', 'policy', 'schedule', 'calendar'], default='page')
    p.add_argument('--topic', action='append', help='Literal topic phrase that should prompt login; repeatable')
    p.add_argument('--extracted', type=Path, help='Reviewed HTML/TXT/Markdown transcription; required for images/scans')
    p.add_argument('--replace', action='store_true', help='Explicitly replace an existing document ID, including its role')
    p.add_argument('--base-data-dir', type=Path, help='Default: ASKIISC_DATA_DIR or config/local.json')
    p.add_argument('--output-data-dir', type=Path, help='Fresh directory; otherwise generate a unique path under data/')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(ingest(args), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--extract-pdf':
        # Local extraction only. Do not initialize model clients or send documents away.
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        from askiisc_retrieval.private_documents import pdf_to_html
        result = pdf_to_html(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
        if len(result.encode('utf-8')) > MAX_TEXT:
            raise SystemExit('Extraction too large')
        print(result)
    else:
        main()
