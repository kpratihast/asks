"""Serving configuration. Everything comes from environment variables; secrets never live in code.

Defaults follow the PRD's example ceiling (C=20 concurrent per tenant, Q=10 queued, 30 requests/min
per key) and a Llama-3.2-3B context window. Calibrate them on the RTX 4090 (NFR-9) and record the
values used with every benchmark run.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    value = os.environ.get(name, '1' if default else '0').strip().lower()
    if value not in {'1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'}:
        raise ValueError(f'{name} must be true/false or 1/0')
    return value in {'1', 'true', 'yes', 'on'}


def _model_base(value: str) -> str:
    base = value.rstrip("/").removesuffix("/v1")
    url = urlsplit(base)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username is not None or url.query or url.fragment
            or any(char.isspace() for char in value)):
        raise ValueError("Model endpoints must be HTTP(S) base URLs without credentials, queries or fragments")
    if url.port is not None and url.port < 1:
        raise ValueError("Invalid model endpoint port")
    return base


@dataclass
class Settings:
    # model endpoint (vLLM on the GPU; Ollama on a laptop: same OpenAI-compatible API)
    vllm_url: str = field(default_factory=lambda: os.environ.get("VLLM_URL", "http://127.0.0.1:8000"))
    served_model: str = field(default_factory=lambda: os.environ.get("SERVED_MODEL", "meta-llama/Llama-3.2-3B-Instruct"))
    model_api_key: str | None = field(default_factory=lambda: os.environ.get("MODEL_API_KEY"), repr=False)
    trusted_model_url: str = field(default_factory=lambda: os.environ.get("ASKIISC_TRUSTED_MODEL_URL", ""))
    restricted_generation: bool = field(init=False)
    model_reasoning: dict = field(default_factory=lambda: json.loads(os.environ.get("MODEL_REASONING", "{}")))
    model_provider_preferences: dict = field(default_factory=lambda: json.loads(os.environ.get("MODEL_PROVIDER_PREFERENCES", "{}")))
    max_model_len: int = field(default_factory=lambda: _i("MAX_MODEL_LEN", 8192))
    max_output_tokens: int = field(default_factory=lambda: _i("MAX_OUTPUT_TOKENS", 384))
    prompt_safety_tokens: int = field(default_factory=lambda: _i("PROMPT_SAFETY_TOKENS", 128))
    tokenizer: str | None = field(default_factory=lambda: os.environ.get("SERVING_TOKENIZER"))  # HF id/path
    gen_connect_timeout_s: float = field(default_factory=lambda: _f("GEN_CONNECT_TIMEOUT_S", 5))
    gen_timeout_s: float = field(default_factory=lambda: _f("GEN_TIMEOUT_S", 120))
    max_question_chars: int = field(default_factory=lambda: _i("MAX_QUESTION_CHARS", 1000))
    max_evidence: int = field(default_factory=lambda: _i("MAX_EVIDENCE", 6))
    regenerate_on_bad_citations: bool = field(default_factory=lambda: os.environ.get("REGENERATE", "1") == "1")

    # Optional public-document chat; disabled unless explicitly enabled by the local launcher.
    public_chat: bool = field(default_factory=lambda: os.environ.get("PUBLIC_CHAT", "0") == "1")

    # Off = public-only mode, with email sessions and privileged API keys ignored.
    email_auth_enabled: bool = field(default_factory=lambda: _b("EMAIL_AUTH_ENABLED", True))

    # Optional template controls for compatible servers (for example Qwen thinking mode).
    chat_template_kwargs: dict = field(default_factory=lambda: json.loads(os.environ.get("CHAT_TEMPLATE_KWARGS", "{}")))

    # access control (FR-6)
    keys_file: Path | None = field(default_factory=lambda: Path(p) if (p := os.environ.get("SERVING_KEYS_FILE")) else None)

    # admission control (FR-7)
    rate_per_min: int = field(default_factory=lambda: _i("RATE_PER_MIN", 30))          # per API key
    tenant_concurrency: int = field(default_factory=lambda: _i("TENANT_CONCURRENCY", 20))  # C
    tenant_queue: int = field(default_factory=lambda: _i("TENANT_QUEUE", 10))           # Q
    queue_wait_s: float = field(default_factory=lambda: _f("QUEUE_WAIT_S", 20))
    global_concurrency: int = field(default_factory=lambda: _i("GLOBAL_CONCURRENCY", 32))

    # retrieval: "local" (in-process) or "remote" (the retrieval service's POST /retrieve)
    retrieval: str = field(default_factory=lambda: os.environ.get("SERVING_RETRIEVAL", "local"))
    retrieval_url: str = field(default_factory=lambda: os.environ.get("RETRIEVAL_URL", "http://127.0.0.1:8090"))

    def __post_init__(self):
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("CHAT_TEMPLATE_KWARGS must be a JSON object")
        if not isinstance(self.model_reasoning, dict):
            raise ValueError("MODEL_REASONING must be a JSON object")
        if not isinstance(self.model_provider_preferences, dict):
            raise ValueError("MODEL_PROVIDER_PREFERENCES must be a JSON object")
        self.vllm_url = _model_base(self.vllm_url)
        url = urlsplit(self.vllm_url)
        if self.model_api_key is None and (url.scheme, url.hostname, url.port or 443) == ("https", "openrouter.ai", 443):
            self.model_api_key = os.environ.get("OPENROUTER_API_KEY")
        try:
            loopback = ip_address(url.hostname).is_loopback
        except ValueError:
            loopback = url.hostname == "localhost"
        trusted_base = _model_base(self.trusted_model_url) if self.trusted_model_url else ""
        self.restricted_generation = (url.hostname.rstrip(".") != "openrouter.ai"
                                      and (loopback or trusted_base == self.vllm_url))
        if self.max_output_tokens < 1 or self.prompt_safety_tokens < 0:
            raise ValueError("Invalid output or safety token budget")
        if self.max_model_len <= self.max_output_tokens + self.prompt_safety_tokens:
            raise ValueError("MAX_MODEL_LEN must leave room for the prompt")
