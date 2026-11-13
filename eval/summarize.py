"""Print the README result tables (markdown) from the saved result files. No model or API calls.

Usage: python -m eval.summarize
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from agent.graph import FALLBACK_ANSWER

RESULTS = Path(__file__).resolve().parent / "results"
EVAL_DIR = RESULTS.parent
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
AGENT_MODES = ["dense", "hybrid", "hybrid_rerank"]
SETS = [("testset", "easy"), ("testset_medium", "medium"), ("testset_hard", "hard"),
        ("testset_harder", "harder"), ("testset_paraphrase", "paraphrase")]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def overlap(name: str, chunks_by_paper: dict) -> float:
    """Mean share of a question's words that also appear in the chunk holding its evidence."""
    shares = []
    for q in json.loads((EVAL_DIR / f"{name}.json").read_text()):
        chunk = next(text for text in chunks_by_paper[q["paper_id"]] if q["evidence"] in text)
        q_words = words(q["question"])
        shares.append(len(q_words & words(chunk)) / len(q_words))
    return sum(shares) / len(shares)


def mrr(ranks: list) -> float:
    return sum(1 / r for r in ranks if r and r <= 10) / len(ranks)


def hit(ranks: list, k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000))
    return means[125], means[4874]


def retrieval_tables() -> None:
    from retrieval.chunking import load_chunks, normalize

    chunks_by_paper = defaultdict(list)
    for c in load_chunks():
        chunks_by_paper[c.paper_id].append(normalize(c.text))
    print("### Retrieval only: MRR@10 and hit@5 (right chunk in the top 5)\n")
    print("| Question set | Questions | Word overlap | " + " | ".join(MODES) + " |")
    print("|---|---|---|" + "---|" * len(MODES))
    pooled = {m: [] for m in MODES}
    for name, label in SETS:
        data = json.loads((RESULTS / f"modes{'' if name == 'testset' else '_' + name}.json").read_text())
        for m in MODES:
            pooled[m] += data[m]["ranks"]
        cells = " | ".join(f"{mrr(data[m]['ranks']):.2f} / {hit(data[m]['ranks'], 5):.2f}" for m in MODES)
        print(f"| {label} | {len(data['dense']['ranks'])} | {overlap(name, chunks_by_paper):.2f} | {cells} |")
    cells = " | ".join(f"{mrr(pooled[m]):.2f} / {hit(pooled[m], 5):.2f}" for m in MODES)
    print(f"| **all** | {len(pooled['dense'])} | | {cells} |\n")


def agent_table(runs: list[dict]) -> None:
    print("### End to end with the agent (50 hard questions, local 4B model)\n")
    print("| Mode | Gave up | Right evidence in final 5 passages | Avg seconds per question |")
    print("|---|---|---|---|")
    for m in AGENT_MODES:
        rs = [r for r in runs if r["mode"] == m]
        gave_up = sum(1 for r in rs if r["answer"] == FALLBACK_ANSWER)
        has = sum(1 for r in rs if any(r["evidence"] in c for c in r["contexts"]))
        print(f"| {m} | {gave_up}/{len(rs)} | {has}/{len(rs)} | {sum(r['seconds'] for r in rs) / len(rs):.1f} |")
    print()


def ragas_tables() -> None:
    scores = {}
    for line in (RESULTS / "ragas_scores.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not (r["note"] or "").startswith("error"):
            scores[(r["id"], r["mode"], r["metric"])] = r["value"]
    print("### RAGAS (judge: gemini-3.5-flash-lite), mean per mode\n")
    print("| Metric | " + " | ".join(AGENT_MODES) + " |")
    print("|---|---|---|---|")
    for metric in ("context_recall", "context_precision", "factual_correctness", "faithfulness"):
        cells = []
        for m in AGENT_MODES:
            vals = [v for (i, mm, k), v in scores.items() if k == metric and mm == m and v is not None]
            cells.append(f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "-")
        print(f"| {metric} | " + " | ".join(cells) + " |")
    print("\n### Paired differences (same questions in both modes), mean and 95% bootstrap interval\n")
    print("| Metric | Questions | rerank vs dense | hybrid vs dense | rerank vs hybrid |")
    print("|---|---|---|---|---|")
    for metric in ("context_recall", "context_precision", "factual_correctness", "faithfulness"):
        ids = sorted({i for (i, _, k), v in scores.items() if k == metric
                      and all(scores.get((i, m, metric)) is not None for m in AGENT_MODES)})
        cells = []
        for a, b in (("hybrid_rerank", "dense"), ("hybrid", "dense"), ("hybrid_rerank", "hybrid")):
            diffs = [scores[(i, a, metric)] - scores[(i, b, metric)] for i in ids]
            lo, hi = bootstrap_ci(diffs) if diffs else (0, 0)
            cells.append(f"{sum(diffs) / len(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]" if diffs else "-")
        print(f"| {metric} | {len(ids)} | " + " | ".join(cells) + " |")
    print()


def main() -> None:
    runs = [json.loads(line) for line in (RESULTS / "agent_runs.jsonl").read_text().splitlines()]
    retrieval_tables()
    agent_table(runs)
    ragas_tables()


if __name__ == "__main__":
    main()
