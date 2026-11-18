"""#46: the restricted-document catalogue must not refuse questions whose answers are on public pages.

The sign-in hint is given only when the caller's own (scope-filtered) evidence gives no verified answer.
Key-file deployment with the checked-in catalogue (config/restricted_documents.json), as in the report."""
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from askiisc_retrieval.ingest_stub import PageInput, build_index
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.snapshot import JsonSnapshotProvider
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore, QdrantStore
from askiisc_serving.app import create_app
from askiisc_serving.auth import KeyStore
from askiisc_serving.email_auth import AuthConfig, EmailAuth
from askiisc_serving.generator import ABSTAIN, Generator
from askiisc_serving.retrieval_client import LocalRetrieval
from askiisc_serving.settings import Settings
from test_serving import FakeModel, sse_body

GUEST, STUDENT = 'gate-guest-key-0123456789abcdef', 'gate-student-key-0123456789abcdef'
SCHEDULE = 'https://cds.iisc.ac.in/courses/schedule/'
DEGREES = 'https://cds.iisc.ac.in/admissions/course-degrees/'
DS252 = 'https://cds.iisc.ac.in/ds252'
DS221 = 'https://cds.iisc.ac.in/courses/ds221'

# question -> (text the answer needs, public page that has it, or None when only a restricted document has it)
QUESTIONS = {
    'Where is DS 252 held in the Aug 2026 term?': ('Venue: CDS 202', SCHEDULE),
    'Do hardcore courses count toward the 36 credits of the M.Tech CDS programme?': ('hard core courses', DEGREES),
    'Does DS 252 have a final exam?': ('one final exam', DS252),
    'When is the DS 221 final exam?': ('Final Exam on Monday', DS221),
    'When is the DS 299 final exam?': ('FACULTY_SECRET_DATE', None),
}


class EvidenceModel(FakeModel):
    """Answers, citing every source, only when the evidence it was given contains the answer; else refuses."""

    def handler(self, request):
        if request.url.path == '/v1/models':
            return super().handler(request)
        body = json.loads(request.content)
        self.requests.append(body)
        prompt = body['messages'][1]['content']
        question = json.loads(re.search(r'Question \(untrusted JSON string\): (".*?")\n', prompt).group(1))
        markers = sorted(set(re.findall(r'^\[(S\d+)\]', prompt, re.M)))
        needed = QUESTIONS[question][0]
        answer = ('The authorized sources contain the requested information ' + ''.join(f'[{m}]' for m in markers)
                  if needed in prompt else ABSTAIN)
        return httpx.Response(200, content=sse_body(answer), headers={'content-type': 'text/event-stream'})


def page(title, text):
    return f'<title>{title}</title><main><h1>{title}</h1><p>{text}</p></main>'


@pytest.fixture
def gateway(tmp_path):
    emb = HashEmbedder()
    store = QdrantStore('public-gate', emb.dim, path=tmp_path / 'qdrant')
    pages = [
        PageInput(SCHEDULE, page('Course Schedule Aug-Dec 2026', 'Course ID: DS 252; Credits: 3:1; Course Title: '
                  'Introduction to Cloud Computing; Instructor: Yogesh Simmhan; Venue: CDS 202.'), 'schedule', 'public'),
        PageInput(DEGREES, page('Course Degrees', 'The M.Tech CDS programme needs 36 credits of coursework, including '
                  'hard core courses (14 credits) and soft core courses chosen with the faculty advisor.'), 'policy', 'public'),
        PageInput(DS252, page('DS 252 Introduction to Cloud Computing', 'DS 252 covers cloud computing, distributed '
                  'storage and data processing. Assessment: projects, quizzes and one final exam [ 22 points ].'), 'page', 'public'),
        PageInput(DS221, page('DS 221 Introduction to Scalable Systems', 'DS 221 covers parallel programming and scalable '
                  'systems for data science. DS 221 Final Exam on Monday, November 30 forenoon.'), 'page', 'public'),
        PageInput('/v1/documents/faculty-exam-timetable-ad2026', page('Final examination timetable',
                  'DS 299 final examination date: FACULTY_SECRET_DATE. Forenoon session. This faculty timetable lists '
                  'the final examinations of the Aug-Dec 2026 term.'), 'schedule', 'faculty'),
        PageInput('/v1/documents/student-course-schedule-ad2026', page('Student course schedule',
                  'DS 299 venue: STUDENT_SECRET_ROOM. This student schedule lists course venues, days of the '
                  'week and timings for the Aug-Dec 2026 term.'), 'schedule', 'student'),
    ]
    report = build_index(pages, emb, store, tmp_path / 'graph.json', tmp_path / 'snapshot.json')
    assert report['chunks'] >= len(pages), report['pages']       # every page indexed (none below MIN_CONTENT_CHARS)
    r = Retriever(emb, store, JsonSnapshotProvider(tmp_path / 'snapshot.json'),
                  JsonGraphStore.load(tmp_path / 'graph.json'), today_term='2026-AUG')
    model = EvidenceModel('unused')
    settings = Settings(served_model='test-model', rate_per_min=200)
    gen = Generator(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(model.handler)))
    keys = KeyStore({GUEST: {'tenant': 'applicants', 'role': 'guest'},
                     STUDENT: {'tenant': 'students', 'role': 'student'}})
    app = create_app(settings, retrieval=LocalRetrieval(r), generator=gen, keys=keys,
                     email_auth=EmailAuth(AuthConfig(db=tmp_path / 'auth.sqlite3', secure_cookie=False)))
    with TestClient(app) as client:
        yield client, model
    store.close()


def ask(client, question, key, stream):
    response = client.post('/v1/ask', json={'question': question, 'mode': 'vector', 'stream': stream},
                           headers={'Authorization': 'Bearer ' + key})
    if stream and response.status_code == 200:
        done = re.search(r'event: done\ndata: (.*)\n', response.text).group(1)
        return response.status_code, json.loads(done), response.text
    return response.status_code, response.json(), response.text


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('key', [GUEST, STUDENT])
@pytest.mark.parametrize('question', [q for q, (_, url) in QUESTIONS.items() if url])
def test_questions_answered_by_public_pages_are_answered_for_every_role(gateway, question, key, stream):
    client, model = gateway
    status, body, text = ask(client, question, key, stream)
    assert status == 200 and body['status'] == 'answered', text
    assert QUESTIONS[question][1] in {c['url'] for c in body['citations'] if c.get('used', True)}
    assert 'required_scopes' not in body
    hidden = 'SECRET' if key == GUEST else 'FACULTY_SECRET_DATE'      # a student may see the student schedule
    assert hidden not in text + json.dumps(model.requests)


@pytest.mark.parametrize('stream', [False, True])
def test_a_question_only_a_restricted_document_answers_still_gets_the_sign_in_hint(gateway, stream):
    client, model = gateway
    question = 'When is the DS 299 final exam?'
    status, body, text = ask(client, question, GUEST, stream)
    assert status == 401 and body['login_required'] and body['status'] == 'login_required'
    status, body, text2 = ask(client, question, STUDENT, stream)
    assert status == 403 and body['status'] == 'access_denied' and not body['login_required']
    assert 'FACULTY_SECRET_DATE' not in text + text2 + json.dumps(model.requests)


def test_guest_evidence_never_includes_restricted_documents(gateway):
    client, model = gateway
    for question in QUESTIONS:
        ask(client, question, GUEST, False)
    assert 'SECRET' not in json.dumps(model.requests)
