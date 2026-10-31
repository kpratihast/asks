"""API-key authentication and scope (FR-6).

Keys live in a YAML file named by SERVING_KEYS_FILE (never in code), e.g.

    keys:
      <random key>: {tenant: applicants, role: guest}
      <random key>: {tenant: students,   role: student}
      <random key>: {tenant: faculty,    role: faculty}
      <random key>: {tenant: office,     role: admin}

Role -> which document scopes retrieval may return. A guest key can only ever see public documents,
whatever the request body says.

    guest    public                                   applicants, the public
    student  public, student                          current students
    faculty  public, student, faculty                 faculty members
    admin    public, student, faculty, admin, staff   department office / administrators
    staff    same as admin (older role name, kept so existing key files still work)
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path

import yaml

ROLE_SCOPES = {
    "guest":   ["public"],
    "student": ["public", "student"],
    "faculty": ["public", "student", "faculty"],
    "admin":   ["public", "student", "faculty", "admin", "staff"],
}
ROLE_SCOPES["staff"] = ROLE_SCOPES["admin"]          # older name for admin


@dataclass(frozen=True)
class Principal:
    key_id: str          # short, non-secret id for logs and rate limiting
    tenant: str
    role: str
    email: str | None = None

    @property
    def scopes(self) -> list[str]:
        return ROLE_SCOPES[self.role]


@dataclass
class Scope:
    scopes: list[str]
    programme: str | None
    academic_year: str | None


class KeyStore:
    def __init__(self, keys: dict[str, dict]):
        self._keys: dict[str, Principal] = {}
        for i, (k, v) in enumerate(keys.items()):
            role = v.get("role", "guest")
            if role not in ROLE_SCOPES:
                raise ValueError(f"unknown role {role!r} for tenant {v.get('tenant')}; "
                                 f"use one of {sorted(ROLE_SCOPES)}")
            if len(k) < 16:
                raise ValueError("API keys must be at least 16 characters")
            self._keys[k] = Principal(f"{v.get('tenant', 'default')}#{i}", v.get("tenant", "default"), role)

    @classmethod
    def from_file(cls, path: Path) -> "KeyStore":
        return cls((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("keys", {}))

    def lookup(self, authorization: str | None) -> Principal | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        presented = authorization[7:].strip()
        for k, p in self._keys.items():                  # constant-time compare per key
            if hmac.compare_digest(k, presented):
                return p
        return None

    @property
    def tenants(self) -> set[str]:
        return {p.tenant for p in self._keys.values()}
