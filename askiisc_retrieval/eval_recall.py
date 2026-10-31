"""Recall@k on the dev set, plus leak checks and latency. Run after every retrieval change, so
improvements are measured, not guessed.

Reports per category: hit@k (any gold matched; all golds for need_all), MRR, leaks (forbidden
markers that reached the results: must be 0), degraded runs, and retrieval latency p50/p95.
For unanswerable questions it records top_score, which is what the abstention threshold is
calibrated against later.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict

import yaml

from .config import ROOT
from .citations import result_to_dict
from .contract import Passage


def matches(spec: dict, p: Passage) -> bool:
    c = p.chunk
    if "url" in spec and c.source_url.rstrip("/") != spec["url"].rstrip("/"):
        return False
    norm = lambda s: s.upper().replace(" ", "")
    if "heading_contains" in spec and norm(spec["heading_contains"]) not in norm(c.section_heading):
        return False
    if "exclude_heading" in spec and norm(spec["exclude_heading"]) in norm(c.section_heading):
        return False
    if "text_contains" in spec and spec["text_contains"].lower() not in c.text.lower():
        return False
    if "term" in spec and c.academic_term != spec["term"]:
        return False
    return True


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


ABSTAIN_CATEGORIES = {"unanswerable", "scope_leak"}     # no evidence the user may see -> must abstain


def calibrate(answer: list[float], abstain: list[float]) -> dict:
    """How well a score separates 'should answer' from 'should abstain' (abstain when score < t).
    margin > 0 means a threshold exists that gets every dev question right."""
    if not answer or not abstain:
        return {"n_answer": len(answer), "n_abstain": len(abstain)}
    cands = sorted(set(answer + abstain))
    cuts = [cands[0] - 1e-6] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 1e-6]
    def acc(t):
        return (sum(x >= t for x in answer) + sum(x < t for x in abstain)) / (len(answer) + len(abstain))
    best = max(cuts, key=lambda t: (acc(t), -abs(t - 0.5)))
    return {"n_answer": len(answer), "n_abstain": len(abstain), "answer_min": round(min(answer), 3),
            "abstain_max": round(max(abstain), 3), "margin": round(min(answer) - max(abstain), 3),
            "best_threshold": round(best, 3), "best_accuracy": round(acc(best), 3)}


def run(retriever, k: int = 5, use_graph: bool = True, out: str | None = None,
        questions_path=ROOT / "dev_questions.yaml", quiet: bool = False, use_rerank: bool = True) -> dict:
    qs = yaml.safe_load(open(questions_path, encoding="utf-8"))["questions"]
    rows, lat, records = [], [], []
    for q in qs:
        res = retriever.retrieve(q["question"], q.get("scopes", ["public"]), use_graph=use_graph,
                                 use_rerank=use_rerank)
        top = res.passages[:k] + [p for p in res.passages[k:] if p.via == "graph"]   # graph extras count
        lat.append(res.timings_ms["total"])
        gold = q.get("gold", [])
        found = [next((i + 1 for i, p in enumerate(top) if matches(g, p)), None) for g in gold]
        if not gold:
            hit = None
        elif q.get("need_all"):
            hit = all(found)
        else:
            hit = any(found)
        first = min((f for f in found if f), default=None)
        gold_found = f"{sum(1 for f in found if f)}/{len(gold)}" if gold else ""
        # Citation precision: of the distinct sources cited for the top-k passages, how many are pages
        # the gold labels point to. Low values = the generator gets (and may cite) irrelevant pages.
        top_markers = {p.marker for p in top}
        cited = [c for c in res.citations if c.marker in top_markers]
        gold_urls = {g["url"].rstrip("/") for g in gold if "url" in g}
        cite_prec = (round(sum(c.source_url.rstrip("/") in gold_urls for c in cited) / len(cited), 3)
                     if gold_urls and cited else None)
        leaks = sum(m in p.chunk.text for p in res.passages for m in q.get("forbidden_markers", []))
        cited_str = " ; ".join(f"[{c.marker}] {c.source_url}#{c.section_heading}" for c in cited)
        records.append({"id": q["id"], "category": q["category"], "gold": gold, **result_to_dict(res)})
        rows.append({"id": q["id"], "category": q["category"], "hit": hit, "rank": first, "gold_found": gold_found,
                     "n_sources": len(cited), "cite_precision": cite_prec, "cited_sources": cited_str,
                     "rr": (1 / first) if first else 0.0, "leaks": leaks, "degraded": res.degraded,
                     "top_score": round(res.top_score, 3), "term": res.target_term,
                     "verified": q.get("verified", False), "_q": q, "ms": res.timings_ms["total"],
                     "embed_ms": res.timings_ms.get("embed"), "search_ms": res.timings_ms.get("search"),
                     "graph_ms": res.timings_ms.get("graph"), "rerank_ms": round((res.timings_ms.get("rerank") or 0) + (res.timings_ms.get("rerank_graph") or 0), 1) or None,
                     "rerank_top": None if res.rerank_top is None else round(res.rerank_top, 3),
                     "sufficient": res.evidence_sufficient, "degraded_reason": res.degraded_reason or "",
                     "question": q["question"]})

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    graded = [r for r in rows if r["hit"] is not None]
    summary = {
        "k": k, "graph": use_graph, "questions": len(rows),
        "backend": getattr(retriever.store, "backend", "?"),
        f"hit@{k}": round(sum(r["hit"] for r in graded) / max(len(graded), 1), 3),
        "mrr": round(statistics.mean(r["rr"] for r in graded), 3) if graded else 0.0,
        "cite_precision": round(statistics.mean(r["cite_precision"] for r in rows if r["cite_precision"] is not None), 3)
                          if any(r["cite_precision"] is not None for r in rows) else None,
        "leaks": sum(r["leaks"] for r in rows), "degraded": sum(r["degraded"] for r in rows),
        "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95),
        "p95_search_ms": pct([r["search_ms"] or 0 for r in rows], 0.95),
    }
    # Abstention calibration: questions the system answered correctly vs questions it must abstain on.
    # (Answerable questions that retrieval MISSED are left out: abstaining on those is acceptable.)
    ans_rows = [r for r in rows if r["hit"]]
    abs_rows = [r for r in rows if r["category"] in ABSTAIN_CATEGORIES]
    summary["calib_vector"] = calibrate([r["top_score"] for r in ans_rows], [r["top_score"] for r in abs_rows])
    if all(r["rerank_top"] is not None for r in ans_rows + abs_rows) and ans_rows:
        summary["calib_rerank"] = calibrate([r["rerank_top"] for r in ans_rows], [r["rerank_top"] for r in abs_rows])
    if not quiet:
        print(f"\n{'id':<5}{'category':<16}{'hit':<6}{'rank':<6}{'vec':<7}{'rerank':<8}{'leaks':<6}{'ms':<8}question")
        for r in rows:
            mark = "-" if r["hit"] is None else ("yes" if r["hit"] else "NO")
            ver = "" if r["verified"] or "verified" not in r["_q"] else "  (unverified gold)"
            rr = "-" if r["rerank_top"] is None else r["rerank_top"]
            gf = f" [{r['gold_found']} sources]" if r["gold_found"] and not r["gold_found"].endswith("/1") else ""
            print(f"{r['id']:<5}{r['category']:<16}{mark:<6}{str(r['rank'] or '-'):<6}{r['top_score']:<7}{rr:<8}"
                  f"{r['leaks']:<6}{r['ms']:<8}{r['question'][:55]}{gf}{ver}")
        print("\nby category:")
        for cat, rs in sorted(by_cat.items()):
            g = [r for r in rs if r["hit"] is not None]
            hr = f"{sum(r['hit'] for r in g)}/{len(g)}" if g else "n/a"
            print(f"  {cat:<16} hit {hr:<6} leaks {sum(r['leaks'] for r in rs)}")
        print("\nsummary:", {k: v for k, v in summary.items() if not k.startswith("calib")})
        print("\nabstention (abstain when score < threshold):")
        for name in ("calib_vector", "calib_rerank"):
            c = summary.get(name)
            if not c or "margin" not in c:
                continue
            sep = "SEPARABLE" if c["margin"] > 0 else "overlapping"
            print(f"  {name[6:]:<7} answered min {c['answer_min']:<6} | must-abstain max {c['abstain_max']:<6} | "
                  f"margin {c['margin']:+.3f} ({sep}) | best threshold {c['best_threshold']} "
                  f"-> {c['best_accuracy']:.0%} of {c['n_answer'] + c['n_abstain']} correct")
        if summary["leaks"]:
            print("!! LEAK: a superseded or out-of-scope passage was returned. This must be 0.")
    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            fields = [k for k in rows[0] if k != "_q"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        import json
        from pathlib import Path
        jpath = Path(out).with_suffix("").as_posix() + "_citations.jsonl"
        with open(jpath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not quiet:
            print(f"\nPer-question citations and passages: {jpath}")
    return {"summary": summary, "rows": rows}
