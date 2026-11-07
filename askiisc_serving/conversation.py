"""Conservative chat routing. History resolves references, never invents a new intent."""
from __future__ import annotations

import re

HELP_REPLY = (
    "I can help with IISc admissions, courses, research and department information. "
    "Please ask a question about one of these topics."
)

# Full-message matches: a greeting before a real question must not swallow the question.
_SOCIAL = re.compile(
    r"(?:(?:hi|hii+|hello|hey|greetings|namaste|good (?:morning|afternoon|evening))"
    r"(?: (?:there|everyone|askiisc|ask iisc))?|(?:thank you|thanks)(?: (?:a lot|so much))?"
    r"|(?:ok|okay|cool|great|bye|goodbye)|how are you(?: doing)?|what(?:'s| is) up"
    r"|who are you|what can you do|help(?: me)?|start (?:over|again)|new chat)", re.I
)
_DOMAIN = re.compile(r"\b(?:iisc|cds|ds\s*\d{3}|university|institute|campus|admissions?|professors?|faculty|courses?|research)\b", re.I)
# Only unmistakable requests outside the assistant's remit are short-circuited.
# Unrecognised academic questions still reach retrieval; this is not a domain allowlist.
_OFF_TOPIC = re.compile(
    r"\b(?:weather|horoscope|stock price|bitcoin price|capital of|tell (?:me )?a joke"
    r"|(?:write|compose) (?:me )?(?:a |an )?(?:poem|song|story)|(?:recipe|recipes) for)\b", re.I
)
_REFERENCE = re.compile(
    r"\b(?:it|its|they|them|their|he|his|she|her|those|these)\b"
    r"|\b(?:that|this|same) (?:course|programme|program|professor|instructor|lab|department|degree|exam|policy)\b", re.I
)
_ELLIPSIS = re.compile(r"^(?:and|also|what about|how about)\b", re.I)
_COURSE = re.compile(r"\b[A-Z]{2,3}\s*\d{3}\b", re.I)
_FILLER = set("a an the is are was were be been do does did can could would should will may might please tell me about more and also it its they them their he his she her that this those these same".split())


def guidance_reason(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", question.strip()).strip(" .!?…,;:").lower()
    if _SOCIAL.fullmatch(normalized):
        return "conversation_help"
    if _OFF_TOPIC.search(normalized) and not _DOMAIN.search(normalized):
        return "out_of_scope"
    return None


def context_question(question: str, history) -> str | None:
    """Use the latest user topic only for a referential or elliptical follow-up.

    Assistant answers are deliberately excluded: generated facts are not evidence.
    Explicit course identifiers usually make a short question self-contained.
    """
    if guidance_reason(question):
        return None
    reference = bool(_REFERENCE.search(question))
    elliptical = bool(_ELLIPSIS.search(question)) and len(question.split()) <= 12
    if not reference and (not elliptical or _COURSE.search(question) or _DOMAIN.search(question)):
        return None
    for turn in reversed(history):
        if turn.role == "user":
            return None if guidance_reason(turn.content) else turn.content
    return None


def valid_rewrite(question: str, rewritten: str, max_chars: int) -> bool:
    """Reject rewrites which drop the latest question's intent or substitute numbers."""
    if not rewritten.strip() or len(rewritten.strip()) > max_chars or guidance_reason(rewritten):
        return False
    tokens = set(re.findall(r"[\w]+", question.casefold())) - _FILLER
    rewritten_tokens = set(re.findall(r"[\w]+", rewritten.casefold()))
    return bool(tokens) and tokens <= rewritten_tokens
