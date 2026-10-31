"""End-to-end view of a test run at the FROZEN threshold (no threshold is fitted here).

    python tools/score_heldout.py test_cds120_standin.csv

From the eval CSV: an answerable question counts as correct only if retrieval found the gold page in
the top k (hit) AND the evidence was judged sufficient (the system would answer). A must-abstain
question counts as correct if the evidence was judged insufficient (the system would refuse).
"""
import csv
import sys
from collections import defaultdict


def b(x):
    return str(x).strip().lower() == "true"


rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
ans = [r for r in rows if r["category"] != "unanswerable"]
abst = [r for r in rows if r["category"] == "unanswerable"]
found = [r for r in ans if b(r["hit"])]
answered_ok = [r for r in found if b(r["sufficient"])]
refused_found = [r for r in found if not b(r["sufficient"])]
missed = [r for r in ans if not b(r["hit"])]
answered_missed = [r for r in missed if b(r["sufficient"])]
refused_ok = [r for r in abst if not b(r["sufficient"])]
print(f"Answerable questions: {len(ans)}")
print(f"  right page in top 5 (retrieval):        {len(found)}/{len(ans)} = {len(found)/max(len(ans),1):.1%}")
print(f"  found AND would answer:                 {len(answered_ok)}/{len(ans)} = {len(answered_ok)/max(len(ans),1):.1%}")
print(f"  found but would refuse (score < thr):   {len(refused_found)}")
print(f"  not found, would answer anyway (risky): {len(answered_missed)}")
print(f"Must-abstain questions: {len(abst)}")
print(f"  correctly refused:                      {len(refused_ok)}/{len(abst)} = {len(refused_ok)/max(len(abst),1):.1%}"
      + ("" if abst else ""))
wrong = [r["id"] for r in abst if b(r["sufficient"])]
if wrong:
    print(f"  WOULD ANSWER (violates 100% abstention): {', '.join(wrong)}")
print(f"Leaks: {sum(int(r.get('leaks') or 0) for r in rows)}")
by = defaultdict(lambda: [0, 0, 0])
for r in ans:
    t = by[r["category"]]; t[0] += 1; t[1] += b(r["hit"]); t[2] += b(r["hit"]) and b(r["sufficient"])
print("\nBy topic (found / found-and-answered / total):")
for k, (n, h, ok) in sorted(by.items()):
    print(f"  {k:<22} {h:>3} / {ok:>3} / {n}")
