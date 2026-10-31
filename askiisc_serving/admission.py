"""Admission control and load shedding (FR-7, NFR-9).

Three checks, cheapest first; any failure is an immediate HTTP 429 with Retry-After:
  1. per-key rate limit      token bucket, RATE_PER_MIN requests per minute per API key
  2. per-tenant concurrency  at most C requests in flight per tenant; up to Q more may wait
                             QUEUE_WAIT_S for a slot, the rest are rejected at once (bounded queue)
  3. global concurrency      at most GLOBAL_CONCURRENCY in flight in this process
An admitted request keeps its slot until its response (including a stream) has finished, so admitted
requests are never cut off by later arrivals.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass


class Rejected(Exception):
    def __init__(self, reason: str, retry_after_s: float):
        super().__init__(reason)
        self.reason, self.retry_after_s = reason, max(1, math.ceil(retry_after_s))


class TokenBucket:
    def __init__(self, per_min: int, clock=time.monotonic):
        self.capacity, self.rate, self.clock = float(per_min), per_min / 60.0, clock
        self.tokens, self.at = float(per_min), clock()

    def take(self) -> float:
        """0 if allowed, else seconds until the next token."""
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.rate


@dataclass
class _Tenant:
    active: int = 0
    waiting: int = 0
    cond: asyncio.Condition | None = None


class Admission:
    def __init__(self, rate_per_min: int, tenant_concurrency: int, tenant_queue: int, queue_wait_s: float,
                 global_concurrency: int, clock=time.monotonic):
        self.rate, self.c, self.q, self.wait = rate_per_min, tenant_concurrency, tenant_queue, queue_wait_s
        self.global_limit, self.clock = global_concurrency, clock
        self.buckets: dict[str, TokenBucket] = {}
        self.tenants: dict[str, _Tenant] = {}
        self.global_active = 0
        self.stats = {"admitted": 0, "rejected_rate": 0, "rejected_queue_full": 0, "rejected_queue_timeout": 0,
                      "rejected_global": 0}

    def _tenant(self, name: str) -> _Tenant:
        t = self.tenants.get(name)
        if t is None:
            t = self.tenants[name] = _Tenant(cond=asyncio.Condition())
        return t

    async def acquire(self, key_id: str, tenant: str) -> "Slot":
        b = self.buckets.setdefault(key_id, TokenBucket(self.rate, self.clock))
        wait = b.take()
        if wait:
            self.stats["rejected_rate"] += 1
            raise Rejected("rate_limited", wait)
        if self.global_active >= self.global_limit:
            self.stats["rejected_global"] += 1
            raise Rejected("server_busy", 2)
        t = self._tenant(tenant)
        async with t.cond:
            if t.active >= self.c:
                if t.waiting >= self.q:
                    self.stats["rejected_queue_full"] += 1
                    raise Rejected("tenant_queue_full", 2)
                t.waiting += 1
                try:
                    await asyncio.wait_for(t.cond.wait_for(lambda: t.active < self.c), self.wait)
                except asyncio.TimeoutError:
                    self.stats["rejected_queue_timeout"] += 1
                    raise Rejected("queue_timeout", 2)
                finally:
                    t.waiting -= 1
            t.active += 1
        self.global_active += 1
        self.stats["admitted"] += 1
        return Slot(self, t)

    def snapshot(self) -> dict:
        return {"global_active": self.global_active, **self.stats,
                "tenants": {k: {"active": v.active, "waiting": v.waiting} for k, v in self.tenants.items()}}


class Slot:
    def __init__(self, adm: Admission, t: _Tenant):
        self.adm, self.t, self.released = adm, t, False

    async def release(self):
        if self.released:
            return
        self.released = True
        self.adm.global_active -= 1
        async with self.t.cond:
            self.t.active -= 1
            self.t.cond.notify()
