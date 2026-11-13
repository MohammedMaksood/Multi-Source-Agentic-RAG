"""Chat with the agent. One-shot: python -m agent.cli --thread demo "your question". Interactive: omit the question."""
import argparse
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import ask, build_graph
from llm.client import LLM
from retrieval.retrievers import Retriever

MEMORY_DB = Path(__file__).resolve().parent.parent / "data" / "memory.db"


def show(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result["sources"]:
        print("sources: " + ", ".join(result["sources"]))
    print(f"(query: {result['query']} | retries: {result['retries']})\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--thread", default="default")
    parser.add_argument("--backend", choices=["ollama", "gemini"])
    parser.add_argument("--mode", default="hybrid_rerank", choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()

    MEMORY_DB.parent.mkdir(exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(MEMORY_DB, check_same_thread=False))
    retriever = Retriever()
    try:
        graph = build_graph(LLM(args.backend), retriever, args.mode, checkpointer)
        if args.question:
            show(ask(graph, args.question, args.thread))
            return
        while question := input("question (empty to quit): ").strip():
            show(ask(graph, question, args.thread))
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
