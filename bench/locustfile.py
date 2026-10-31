"""Load generator for the serving tier (PRD 7.1): Poisson arrivals, 3 tenants (separate API keys),
questions replayed from the held-out set.

    pip install locust
    $env:ASKIISC_KEYS="key-applicants,key-students,key-staff"     # one key per tenant
    locust -f bench/locustfile.py --host http://127.0.0.1:8100 --users 10 --spawn-rate 2 --run-time 10m --headless --csv results/c10

Sweep --users 1 5 10 20 30 (PRD) and keep each run's CSV. Each simulated user waits an exponential
(Poisson-process) gap with mean MEAN_GAP_S between requests. HTTP 429 is counted separately as "shed".
"""
import itertools
import os
import random
from pathlib import Path

import yaml
from locust import HttpUser, task

KEYS = [k.strip() for k in os.environ.get("ASKIISC_KEYS", "").split(",") if k.strip()]
if not KEYS:
    raise SystemExit("set ASKIISC_KEYS to a comma-separated list of API keys (one per tenant)")
QFILE = Path(os.environ.get("ASKIISC_QUESTIONS", Path(__file__).resolve().parent.parent / "test_cds120.yaml"))
QUESTIONS = [q["question"] for q in yaml.safe_load(QFILE.read_text(encoding="utf-8"))["questions"]]
MEAN_GAP_S = float(os.environ.get("MEAN_GAP_S", "2"))
MODE = os.environ.get("ASKIISC_MODE", "full")
_key_cycle = itertools.cycle(KEYS)


class Student(HttpUser):
    def wait_time(self):
        return random.expovariate(1.0 / MEAN_GAP_S)

    def on_start(self):
        self.key = next(_key_cycle)                      # spread users across tenants

    @task
    def ask(self):
        q = random.choice(QUESTIONS)
        with self.client.post("/v1/ask", json={"question": q, "mode": MODE},
                              headers={"Authorization": f"Bearer {self.key}"},
                              name="/v1/ask", catch_response=True) as r:
            if r.status_code == 429:
                r.failure("shed (429)")
            elif r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            else:
                status = r.json().get("status")
                r.success()
                self.environment.events.request.fire(request_type="STATUS", name=status, response_time=0,
                                                     response_length=0, exception=None, context={})
