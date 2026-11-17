"""Write a serving key file with long random keys (one per role), readable only by you.

    python tools/make_keys.py serving_keys.yaml          # refuses to overwrite an existing file
    python tools/make_keys.py serving_keys.yaml --roles guest student

Keep the file out of git (serving_keys.yaml is git-ignored) and share keys privately.
"""
import argparse
import os
import secrets
import sys
from pathlib import Path

import yaml

TENANTS = {"guest": "applicants", "student": "students", "faculty": "faculty", "admin": "office"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--roles", nargs="+", default=list(TENANTS), choices=list(TENANTS))
    a = ap.parse_args()
    if a.path.exists():
        print(f"{a.path} already exists; not overwriting it", file=sys.stderr)
        return 1
    keys = {secrets.token_urlsafe(32): {"tenant": TENANTS[r], "role": r} for r in a.roles}
    a.path.write_text(yaml.safe_dump({"keys": keys}, sort_keys=False), encoding="utf-8")
    os.chmod(a.path, 0o600)
    print(f"Wrote {len(keys)} random keys ({', '.join(a.roles)}) to {a.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
