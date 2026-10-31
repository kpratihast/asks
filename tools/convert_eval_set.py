"""Convert the evaluation team's question file (JSON or CSV with question_id, query, citation_url,
expected_behavior, topic) into the retrieval eval format (YAML).

    python tools/convert_eval_set.py cds_iisc_rag_eval_120.json test_cds120.yaml

Gold = the cited PAGE (citation_url). The file's `citation_section` values are descriptive labels
("Aptitude-test format"), not the page's real headings, so retrieval is scored at page level.
expected_behavior "abstain" -> category "unanswerable" (must be refused); otherwise category = topic.
This script only converts; it never changes the questions or their labels.
"""
import csv
import json
import sys
from pathlib import Path

import yaml


def load(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def convert(rows: list[dict]) -> dict:
    out = []
    for r in rows:
        abstain = (r.get("expected_behavior") or "").strip().lower() == "abstain"
        url = (r.get("citation_url") or "").strip()
        out.append({
            "id": r["question_id"],
            "category": "unanswerable" if abstain else (r.get("topic") or "answerable"),
            "question": r["query"],
            "gold": [] if abstain or not url else [{"url": url}],
            "verified": True,
            "expected_answer": r.get("expected_answer", ""),
            "section_label": r.get("citation_section", ""),
        })
    return {"questions": out}


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    data = convert(load(src))
    dst.write_text("# Held-out test set from the evaluation team, converted unchanged. DO NOT tune on it.\n"
                   + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    n = len(data["questions"]); a = sum(q["category"] == "unanswerable" for q in data["questions"])
    print(f"{n} questions ({n - a} answerable, {a} must-abstain) -> {dst}")
