"""Per-key token buckets and bounded queues sharing one process-wide capacity limit.

Requests wait in a FIFO queue for their tenant when either concurrency limit is full.
A single condition protects both limits, including the check after a waiter wakes.
Admitted requests keep their slot until the complete response/stream finishes.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field


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
    tickets: deque = field(default_factory=deque)

    @property
    def waiting(self):
        return len(self.tickets)


class Admission:
    def __init__(self, rate_per_min: int, tenant_concurrency: int, tenant_queue: int, queue_wait_s: float,
                 global_concurrency: int, clock=time.monotonic):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               [rate_per_min, tenant_concurrency, global_concurrency]):
            raise ValueError('Rate and concurrency limits must be positive integers')
        if isinstance(tenant_queue, bool) or not isinstance(tenant_queue, int) or tenant_queue < 0:
            raise ValueError('Tenant queue must be a nonnegative integer')
        if isinstance(queue_wait_s, bool) or not isinstance(queue_wait_s,(int,float)) or not math.isfinite(queue_wait_s) or queue_wait_s <= 0:
            raise ValueError('Queue wait must be a positive finite number')
        self.rate, self.c, self.q, self.wait = rate_per_min, tenant_concurrency, tenant_queue, queue_wait_s
        self.global_limit, self.clock = global_concurrency, clock
        self.buckets: dict[str, TokenBucket] = {}
        self.tenants: dict[str, _Tenant] = {}
        self.changed = asyncio.Condition()
        self.global_active = 0
        self.stats = {"admitted": 0, "rejected_rate": 0, "rejected_queue_full": 0, "rejected_queue_timeout": 0,
                      "rejected_global": 0}

    async def acquire(self, key_id: str, tenant: str) -> "Slot":
        b = self.buckets.setdefault(key_id, TokenBucket(self.rate, self.clock))
        wait = b.take()
        if wait:
            self.stats["rejected_rate"] += 1
            raise Rejected("rate_limited", wait)
        async with self.changed:
            t = self.tenants.setdefault(tenant, _Tenant())
            if t.active >= self.c or self.global_active >= self.global_limit or t.waiting:
                if t.waiting >= self.q:
                    if self.q == 0 and self.global_active >= self.global_limit:
                        self.stats["rejected_global"] += 1
                        raise Rejected("server_busy", 2)
                    self.stats["rejected_queue_full"] += 1
                    raise Rejected("tenant_queue_full", 2)
                ticket = object()
                t.tickets.append(ticket)
                try:
                    await asyncio.wait_for(self.changed.wait_for(
                        lambda: t.tickets[0] is ticket and t.active < self.c and self.global_active < self.global_limit
                    ), self.wait)
                except asyncio.TimeoutError:
                    self.stats["rejected_queue_timeout"] += 1
                    raise Rejected("queue_timeout", 2)
                finally:
                    # Timeouts and disconnected/cancelled clients must not strand queue entries.
                    t.tickets.remove(ticket)
                    self.changed.notify_all()
            t.active += 1
            self.global_active += 1
            self.stats["admitted"] += 1
            return Slot(self, t)

    def snapshot(self) -> dict:
        return {"global_active": self.global_active, **self.stats,
                "limits": {"rate_per_min": self.rate, "tenant_concurrency": self.c,
                           "tenant_queue": self.q, "queue_wait_s": self.wait,
                           "global_concurrency": self.global_limit},
                "tenants": {k: {"active": v.active, "waiting": v.waiting} for k, v in self.tenants.items()}}


class Slot:
    def __init__(self, adm: Admission, t: _Tenant):
        self.adm, self.t, self.released = adm, t, False

    async def release(self):
        async with self.adm.changed:
            if self.released:
                return
            self.released = True
            self.adm.global_active -= 1
            self.t.active -= 1
            self.adm.changed.notify_all()
