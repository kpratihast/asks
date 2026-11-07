"""HTML link identity and contact evidence. This module never opens a URL."""
from __future__ import annotations

import re
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

EMAIL = re.compile(r"(?<![\w.+-])[a-z0-9][a-z0-9._%+\-]{0,63}@[a-z0-9.-]+\.[a-z]{2,}(?![\w-])", re.I)


def page_key(url: str, base: str = "") -> str | None:
    if not url or url.startswith("#") or len(url) > 2048:
        return None
    try:
        parts = urlsplit(urljoin(base, url))
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return None
        # Treat trailing slash and fragment variants as the same document.
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/") or "/", parts.query, ""))
    except ValueError:
        return None


def anchor_emails(href: str) -> tuple[str, ...]:
    """Use published mailto recipients, including percent encoding and mailto query arguments."""
    if not href.lower().startswith("mailto:"):
        return ()
    recipients = unquote(href[7:].split("?", 1)[0])
    return tuple(email.lower() for email in map(str.strip, re.split(r"[,;]", recipients))
                 if EMAIL.fullmatch(email))
