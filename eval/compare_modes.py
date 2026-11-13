"""Compare retrieval modes (dense, bm25, hybrid, hybrid_rerank) on eval test sets (no LLM judge).

Usage: python -m eval.compare_modes [testset ...]   (default: testset)
"""
import json
import sys
import time
from pathlib import Path

from eval.bakeoff_embedders import first_hit_rank
from eval.bakeoff_rerankers import summarize
from retrieval.retrievers import CANDIDATES, Retriever

EVAL_DIR = Path(__file__).resolve().parent
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def run(retriever, testset):
    questions = json.loads((EVAL_DIR / f"{testset}.json").read_text())
    results = {}
    print(f"--- {testset} ({len(questions)} questions)", flush=True)
    for mode in MODES:
        ranks = []
        started = time.time()
        for q in questions:
            found = retriever.retrieve(q["question"], mode, k=CANDIDATES)
            ranks.append(first_hit_rank(range(len(found)), [c.text for c in found], q["evidence"]))
        seconds_per_query = (time.time() - started) / len(questions)
        results[mode] = {
            **summarize(ranks),
            "hit@30": sum(r is not None for r in ranks) / len(ranks),
            "seconds_per_query": round(seconds_per_query, 3),
            "ranks": ranks,
        }
        r = results[mode]
        print(f"{mode:14s} hit@1={r['hit@1']:.3f} hit@5={r['hit@5']:.3f} hit@10={r['hit@10']:.3f} "
              f"hit@30={r['hit@30']:.3f} mrr@10={r['mrr@10']:.3f} sec/query={r['seconds_per_query']}", flush=True)
    suffix = "" if testset == "testset" else f"_{testset}"
    (EVAL_DIR / "results" / f"modes{suffix}.json").write_text(json.dumps(results, indent=2))


def main():
    retriever = Retriever()
    try:
        for testset in sys.argv[1:] or ["testset"]:
            run(retriever, testset)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
