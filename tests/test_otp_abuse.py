"""#47: requests from someone else must not invalidate a user's pending code or use up their allowance."""
import sqlite3

import pytest

from askiisc_serving.email_auth import AuthConfig, AuthError, EmailAuth

VICTIM, OWN_IP, ATTACKER = "victim@iisc.ac.in", "198.51.100.7", "203.0.113.9"


class Mail:
    available = True

    def __init__(self):
        self.sent = []

    def send(self, to, code):
        self.sent.append((to, code))


@pytest.fixture
def auth(tmp_path):
    now = [1_800_000_000.0]
    a = EmailAuth(AuthConfig(db=tmp_path / "auth.sqlite3", secure_cookie=False), Mail(), lambda: now[0])
    a.now = now
    return a


def code_for(auth, n):
    return [c for to, c in auth.mailer.sent if to == VICTIM][n]


def test_another_clients_request_does_not_invalidate_the_pending_code(auth):
    mine = auth.issue(VICTIM, OWN_IP)["challenge_id"]
    auth.now[0] += 61
    auth.issue(VICTIM, ATTACKER)
    token, user = auth.verify(mine, code_for(auth, 0))                 # rt-baseline: 400 invalid or expired
    assert token and user["email"] == VICTIM


def test_another_client_cannot_use_up_the_owners_allowance_or_cooldown(auth):
    auth.issue(VICTIM, OWN_IP)
    for _ in range(4):                                                 # 5 requests for the victim in the hour
        auth.now[0] += 61
        auth.issue(VICTIM, ATTACKER)
    auth.now[0] += 1
    assert auth.issue(VICTIM, OWN_IP)["challenge_id"]                  # rt-baseline: 429 for an hour
    auth.now[0] += 61
    auth.issue(VICTIM, ATTACKER)                                       # the attacker's 5th request this hour
    with pytest.raises(AuthError) as err:                              # ... and their limit is still 5/hour
        auth.now[0] += 61
        auth.issue(VICTIM, ATTACKER)
    assert err.value.status == 429


def test_one_inbox_is_capped_across_clients(auth):
    sent = 0
    for i in range(30):
        auth.now[0] += 61
        try:
            auth.issue(VICTIM, f"203.0.113.{i}")
            sent += 1
        except AuthError as e:
            assert e.status == 429
    assert sent == 15                                                  # per-address cap per hour


def test_existing_database_with_one_code_per_address_is_upgraded(tmp_path):
    db = tmp_path / "auth.sqlite3"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE challenges (id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, digest TEXT NOT NULL, "
                  "expires REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0)")
    now = [1_800_000_000.0]
    auth = EmailAuth(AuthConfig(db=db, secure_cookie=False), Mail(), lambda: now[0])
    first = auth.issue(VICTIM, OWN_IP)["challenge_id"]
    now[0] += 61
    auth.issue(VICTIM, ATTACKER)
    assert auth.verify(first, [c for _, c in auth.mailer.sent][0])[0]


def test_own_resend_still_invalidates_the_requesters_previous_code(auth):
    first = auth.issue(VICTIM, OWN_IP)["challenge_id"]
    auth.now[0] += 61
    second = auth.issue(VICTIM, OWN_IP)["challenge_id"]
    with pytest.raises(AuthError):
        auth.verify(first, code_for(auth, 0))                          # only the latest own code is valid
    assert auth.verify(second, code_for(auth, 1))[0]
