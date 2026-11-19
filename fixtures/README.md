# Synthetic fixtures

These pages are invented. They are indexed together with the real CDS pages so that the
version and scope filters can be tested on every run:

| File | What it tests | Must never appear when |
|---|---|---|
| `superseded_mtech_requirements.html` | version filter | always (it is not in the active snapshot) |
| `student_notice_lab_access.html` | scope filter | the user's scopes are only `public` |
| `superseded_course_schedule.html`, `superseded_course_descriptions.html` | version filter in `tools/eval_stale.py` / `test_stale12.yaml` (#44); not listed in `fixtures.yaml` | always (indexed only by that script, as inactive versions) |

Each contains a marker (`SUPERSEDED-FIXTURE`, `STUDENT-ONLY-FIXTURE`) that the tests and the
eval script search for. Do not use them as evidence for real questions.
