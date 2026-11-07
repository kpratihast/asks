"""One registry for source ownership, question routing and department labels."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

SHARED = "IISc"
DEFAULT = "CDS"
REGISTRY = Path(__file__).resolve().parent.parent / "config/departments.yaml"


@dataclass(frozen=True)
class Department:
    name: str
    aliases: re.Pattern
    course_prefixes: tuple[str, ...]


@lru_cache(maxsize=8)
def registry(path: Path) -> dict[str, Department]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["departments"]
    owners = {}
    prefix_owners = {}
    result = {}
    for code, spec in data.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_-]{1,15}", code) or code == SHARED.upper():
            raise ValueError(f"Invalid department code: {code}")
        name = spec["name"].strip()
        aliases = {code, name, *spec.get("aliases", [])}
        if not name or any(not alias.strip() for alias in aliases):
            raise ValueError(f"Empty department alias: {code}")
        for alias in aliases:
            key = " ".join(alias.casefold().split())
            if owners.setdefault(key, code) != code:
                raise ValueError(f"Ambiguous department alias: {alias}")
        alternatives = sorted(aliases, key=lambda alias: (-len(alias), alias))
        pattern = "|".join(r"\s+".join(re.escape(word) for word in alias.split()) for alias in alternatives)
        prefixes = tuple(spec.get("course_prefixes", []))
        for prefix in prefixes:
            if not re.fullmatch(r"[A-Z]{1,3}[0-9]?", prefix) or prefix_owners.setdefault(prefix, code) != code:
                raise ValueError(f"Invalid or shared course prefix: {prefix}")
        result[code] = Department(name, re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I), prefixes)
    return result


def configured() -> dict[str, Department]:
    return registry(Path(os.environ.get("ASKIISC_DEPARTMENT_REGISTRY", REGISTRY)).expanduser().resolve())


def mentioned(text: str) -> tuple[str, ...]:
    matches = [(match.start(), -len(match.group()), code)
               for code, department in configured().items()
               for match in department.aliases.finditer(text)]
    result, end = [], -1
    for start, neg_length, code in sorted(matches):
        if start < end:
            continue
        end = start - neg_length
        if code not in result:
            result.append(code)
    return tuple(result)


def label(code: str) -> str:
    department = configured().get(code)
    return department.name if department else code


def course_department(codes: list[str]) -> str | None:
    """Route only when every code has the same explicitly configured owner."""
    prefixes = {prefix: code for code, department in configured().items() for prefix in department.course_prefixes}
    owners = {prefixes.get(code[:-3]) for code in codes}
    return next(iter(owners)) if len(owners) == 1 else None


def validate_codes(codes) -> None:
    unknown = set(codes) - configured().keys() - {SHARED}
    if unknown:
        raise ValueError("Unknown department codes: " + ", ".join(sorted(unknown)))


def source_departments(default=DEFAULT) -> tuple[str, ...]:
    codes = tuple(dict.fromkeys(code.strip() for code in
                               os.environ.get("ASKIISC_SOURCE_DEPARTMENTS", default).split(",")
                               if code.strip()))
    validate_codes(codes)
    if not codes:
        raise ValueError("Select at least one source department")
    return codes
