"""Email OTPs and revocable opaque sessions. Roles always come from the server directory."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager, closing
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

from .auth import Principal

ROOT = Path(__file__).resolve().parent.parent
COOKIE = 'askiisc_session'


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    # Exact domain only; no display names, plus aliases, subdomains or substring matching.
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}@iisc\.ac\.in', email):
        raise AuthError(422, 'Use your @iisc.ac.in email address.')
    return email


class AuthError(Exception):
    def __init__(self, status: int, message: str, retry_after: int | None = None):
        self.status, self.message, self.retry_after = status, message, retry_after
        super().__init__(message)


@dataclass
class AuthConfig:
    db: Path = field(default_factory=lambda: Path(os.environ.get('AUTH_DB_PATH', str(Path(os.environ.get('ASKIISC_DATA_DIR', ROOT / 'data' / 'local')) / 'auth' / 'sessions.sqlite3'))))
    directory: Path = field(default_factory=lambda: Path(os.environ.get('FACULTY_DIRECTORY', ROOT / 'config' / 'faculty_directory.json')))
    professor_test_emails: tuple[str, ...] = field(default_factory=lambda: tuple(
        email.strip() for email in os.environ.get('AUTH_PROFESSOR_TEST_EMAILS', '').split(',') if email.strip()))
    otp_ttl: int = 600
    session_ttl: int = 8 * 3600
    cooldown: int = 60
    max_attempts: int = 5
    secure_cookie: bool = field(default_factory=lambda: os.environ.get('AUTH_COOKIE_SECURE', '1') == '1')
    smtp_host: str = field(default_factory=lambda: os.environ.get('SMTP_HOST', ''))
    smtp_port: int = field(default_factory=lambda: int(os.environ.get('SMTP_PORT', '587')))
    smtp_user: str = field(default_factory=lambda: os.environ.get('SMTP_USERNAME', ''))
    smtp_password: str = field(default_factory=lambda: os.environ.get('SMTP_PASSWORD', ''), repr=False)
    smtp_from: str = field(default_factory=lambda: os.environ.get('SMTP_FROM', ''))


class SMTPMailer:
    def __init__(self, config: AuthConfig):
        self.s = config

    @property
    def available(self):
        return bool(self.s.smtp_host and self.s.smtp_from and self.s.smtp_user and self.s.smtp_password)

    def send(self, recipient: str, code: str):
        s = self.s
        msg = EmailMessage()
        msg['From'], msg['To'], msg['Subject'] = s.smtp_from, recipient, 'Your Ask IISc sign-in code'
        msg.set_content(f'Your Ask IISc sign-in code is: {code}\n\nIt expires in 10 minutes and can only be used once. '
                        'If you did not request this code, ignore this message. Never share it.')
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            smtp.login(s.smtp_user, s.smtp_password)
            smtp.send_message(msg)


class EmailAuth:
    def __init__(self, config: AuthConfig | None = None, mailer=None, clock=time.time):
        self.s = config or AuthConfig()
        self.mailer = mailer or SMTPMailer(self.s)
        self.clock = clock
        self.test_professors = frozenset(normalize_email(email) for email in self.s.professor_test_emails)
        raw = json.loads(self.s.directory.read_text(encoding='utf-8'))
        self.faculty = {normalize_email(email): entry['name'] for entry in raw['faculty'] for email in entry['emails']}

    @property
    def available(self):
        return bool(self.mailer.available)

    @contextmanager
    def _db(self):
        self.s.db.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        c = sqlite3.connect(str(self.s.db), timeout=15)
        os.chmod(self.s.db, 0o600)
        c.row_factory = sqlite3.Row
        c.executescript('''
            CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS challenges (
                id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, digest TEXT NOT NULL,
                expires REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sends (email TEXT NOT NULL, ip TEXT NOT NULL, at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS sends_at ON sends(at);
            CREATE TABLE IF NOT EXISTS sessions (digest TEXT PRIMARY KEY, email TEXT NOT NULL, expires REAL NOT NULL);
        ''')
        c.execute('INSERT OR IGNORE INTO meta VALUES (?, ?)', ('otp_secret', secrets.token_bytes(32)))
        c.commit()
        try:
            with c:
                yield c
        finally:
            c.close()

    @staticmethod
    def _digest(c, challenge, code):
        key = c.execute('SELECT value FROM meta WHERE name=?', ('otp_secret',)).fetchone()[0]
        return hmac.new(key, f'{challenge}:{code}'.encode(), hashlib.sha256).hexdigest()

    def role_for(self, email):
        return 'faculty' if email in self.faculty or email in self.test_professors else 'student'

    def issue(self, value: str, ip: str):
        email, now = normalize_email(value), self.clock()
        if not self.available:
            raise AuthError(503, 'Email sign-in is awaiting sender configuration. Public information is available without login.')
        with self._db() as c:
            c.execute('BEGIN IMMEDIATE')
            c.execute('DELETE FROM sends WHERE at < ?', (now - 3600,))
            c.execute('DELETE FROM challenges WHERE expires < ?', (now,))
            c.execute('DELETE FROM sessions WHERE expires < ?', (now,))
            last = c.execute('SELECT MAX(at) FROM sends WHERE email=?', (email,)).fetchone()[0]
            if last is not None and now - last < self.s.cooldown:
                raise AuthError(429, 'Please wait before requesting another code.', max(1, int(self.s.cooldown - (now-last))))
            per_email = c.execute('SELECT COUNT(*) FROM sends WHERE email=? AND at>?', (email, now - 3600)).fetchone()[0]
            per_ip = c.execute('SELECT COUNT(*) FROM sends WHERE ip=? AND at>?', (ip, now - 3600)).fetchone()[0]
            if per_email >= 5 or per_ip >= 20:
                raise AuthError(429, 'Too many code requests. Try again later.', 3600)
            challenge, code = secrets.token_urlsafe(24), f'{secrets.randbelow(1000000):06d}'
            c.execute('INSERT OR REPLACE INTO challenges(id,email,digest,expires,attempts) VALUES (?,?,?,?,0)',
                      (challenge, email, self._digest(c, challenge, code), now + self.s.otp_ttl))
            c.execute('INSERT INTO sends VALUES (?,?,?)', (email, ip, now))
        try:
            self.mailer.send(email, code)
        except Exception:
            with self._db() as c:
                c.execute('DELETE FROM challenges WHERE id=?', (challenge,))
            # Do not disclose SMTP credentials, recipient addresses or provider error bodies.
            raise AuthError(503, 'The code could not be sent. Please try again later.') from None
        return {'challenge_id': challenge, 'expires_in': self.s.otp_ttl,
                'resend_after': self.s.cooldown, 'message': 'A sign-in code has been sent to your IISc email.'}

    def verify(self, challenge: str, code: str):
        now, error, token, email = self.clock(), None, None, None
        with self._db() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT * FROM challenges WHERE id=?', (challenge,)).fetchone()
            if row is None or row['expires'] <= now or row['attempts'] >= self.s.max_attempts:
                error = 'This code is invalid or expired. Request a new code.'
            elif not re.fullmatch(r'\d{6}', code) or not hmac.compare_digest(row['digest'], self._digest(c, challenge, code)):
                c.execute('UPDATE challenges SET attempts=attempts+1 WHERE id=?', (challenge,))
                error = 'This code is invalid or expired. Request a new code.'
            else:
                email = row['email']
                c.execute('DELETE FROM challenges WHERE id=?', (challenge,))
                token = secrets.token_urlsafe(32)
                c.execute('INSERT INTO sessions VALUES (?,?,?)',
                          (hashlib.sha256(token.encode()).hexdigest(), email, now + self.s.session_ttl))
        if error:
            raise AuthError(400, error)
        return token, self.identity(email)

    def identity(self, email):
        return {'authenticated': True, 'email': email, 'role': self.role_for(email),
                'display_role': 'Professor' if self.role_for(email) == 'faculty' else 'Student'}

    def lookup(self, token):
        if not token or len(token) > 128:
            return None
        with self._db() as c:
            row = c.execute('SELECT email, expires FROM sessions WHERE digest=?',
                            (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if row is None or row['expires'] <= self.clock():
            return None
        email = row['email']
        return Principal('email-' + hashlib.sha256(email.encode()).hexdigest()[:16],
                         'iisc-' + self.role_for(email), self.role_for(email), email)

    @staticmethod
    def revoke_session(token, db):
        # Logout still revokes old cookies while email auth is disabled, without loading
        # a faculty directory, SMTP settings, or creating a new auth database.
        if not token or not db.is_file():
            return
        with closing(sqlite3.connect(str(db), timeout=15)) as c:
            if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone():
                with c:
                    c.execute('DELETE FROM sessions WHERE digest=?', (hashlib.sha256(token.encode()).hexdigest(),))

    def logout(self, token):
        self.revoke_session(token, self.s.db)
