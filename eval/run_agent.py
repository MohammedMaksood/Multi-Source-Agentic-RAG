"""Run the agent over the hardest question sets in each retrieval mode and save answers and contexts.

Uses the local Ollama model, so it costs nothing. Resumable: finished (question, mode) pairs are skipped.
Usage: python -m eval.run_agent
"""
import json
import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import ask, build_graph
from llm.client import LLM
from retrieval.retrievers import Retriever

EVAL_DIR = Path(__file__).resolve().parent
SETS = ["testset_hard", "testset_harder", "testset_paraphrase"]
MODES = ["dense", "hybrid", "hybrid_rerank"]
OUT = EVAL_DIR / "results" / "agent_runs.jsonl"


def finished_pairs() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    return {(r["id"], r["mode"]) for r in rows}


def main():
    questions = [q for name in SETS for q in json.loads((EVAL_DIR / f"{name}.json").read_text())]
    references = json.loads((EVAL_DIR / "references.json").read_text())
    done = finished_pairs()
    llm = LLM("ollama")
    retriever = Retriever()
    try:
        checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
        graphs = {mode: build_graph(llm, retriever, mode, checkpointer) for mode in MODES}
        with OUT.open("a") as out:
            for q in questions:
                for mode in MODES:
                    if (q["id"], mode) in done:
                        continue
                    started = time.time()
                    result = ask(graphs[mode], q["question"], f"{mode}-{q['id']}")
                    row = {
                        "id": q["id"],
                        "mode": mode,
                        "question": q["question"],
                        "reference": references[q["id"]],
                        "evidence": q["evidence"],
                        "answer": result["answer"],
                        "contexts": [d["text"] for d in result["documents"]],
                        "sources": result["sources"],
                        "query": result["query"],
                        "retries": result["retries"],
                        "sufficient": result["sufficient"],
                        "seconds": round(time.time() - started, 1),
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"{q['id']} {mode:14s} retries={row['retries']} sufficient={row['sufficient']} {row['seconds']}s", flush=True)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
