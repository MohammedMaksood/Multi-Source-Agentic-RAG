"""Compare rerankers on the top-30 BGE-M3 candidates for each question in eval/testset.json (no LLM judge)."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from eval.bakeoff_embedders import CUTOFFS, first_hit_rank
from retrieval.chunking import load_chunks

EVAL_DIR = Path(__file__).resolve().parent
EMBEDDER = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")
RERANKERS = [
    ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
]
CANDIDATES = 30


def summarize(ranks):
    result = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in CUTOFFS}
    result["mrr@10"] = sum(1 / r for r in ranks if r is not None and r <= 10) / len(ranks)
    return result


def first_stage(chunk_texts, questions):
    name, revision = EMBEDDER
    model = SentenceTransformer(name, revision=revision, device="cuda")
    kwargs = {"batch_size": 16, "normalize_embeddings": True, "show_progress_bar": False}
    chunk_vecs = model.encode(chunk_texts, **kwargs)
    query_vecs = model.encode([q["question"] for q in questions], **kwargs)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [list(np.argsort(-(chunk_vecs @ qvec))[:CANDIDATES]) for qvec in query_vecs]


def rerank(name, revision, chunk_texts, questions, candidates):
    model = CrossEncoder(name, revision=revision, device="cuda")
    ranks = []
    started = time.time()
    for q, cand in zip(questions, candidates):
        scores = model.predict([(q["question"], chunk_texts[i]) for i in cand], batch_size=8, show_progress_bar=False)
        reranked = [cand[i] for i in np.argsort(-scores)]
        ranks.append(first_hit_rank(reranked, chunk_texts, q["evidence"]))
    seconds_per_query = (time.time() - started) / len(questions)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ranks, seconds_per_query


def main():
    chunk_texts = [c.text for c in load_chunks()]
    questions = json.loads((EVAL_DIR / "testset.json").read_text())
    candidates = first_stage(chunk_texts, questions)

    baseline_ranks = [first_hit_rank(cand, chunk_texts, q["evidence"]) for q, cand in zip(questions, candidates)]
    results = {"no rerank (bge-m3 top 30)": {**summarize(baseline_ranks), "ranks": baseline_ranks}}
    print(f"candidates contain the evidence for {sum(r is not None for r in baseline_ranks)}/{len(questions)} questions")
    print(f"{'no rerank':42s} " + " ".join(f"{k}={v:.3f}" for k, v in summarize(baseline_ranks).items()), flush=True)

    out = EVAL_DIR / "results" / "rerankers.json"
    for name, revision in RERANKERS:
        ranks, seconds = rerank(name, revision, chunk_texts, questions, candidates)
        results[name] = {**summarize(ranks), "seconds_per_query": round(seconds, 2), "ranks": ranks}
        print(f"{name:42s} " + " ".join(f"{k}={v:.3f}" for k, v in summarize(ranks).items())
              + f" sec/query={seconds:.2f}", flush=True)
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
