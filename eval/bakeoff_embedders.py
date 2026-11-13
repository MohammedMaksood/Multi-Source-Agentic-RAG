"""Compare embedding models on retrieval hit-rate and MRR over eval/testset.json (no LLM involved)."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from retrieval.chunking import load_chunks

EVAL_DIR = Path(__file__).resolve().parent
MODELS = [
    ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
]
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}
CUTOFFS = (1, 5, 10)


def first_hit_rank(ranked_chunk_ids, chunk_texts, evidence):
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if evidence in chunk_texts[chunk_id]:
            return rank
    return None


def evaluate(name, revision, chunk_texts, questions):
    model = SentenceTransformer(name, revision=revision, device="cuda")
    encode_kwargs = {"batch_size": 16, "normalize_embeddings": True, "show_progress_bar": False}
    started = time.time()
    chunk_vecs = model.encode(chunk_texts, **encode_kwargs)
    index_seconds = time.time() - started

    queries = [QUERY_PREFIX.get(name, "") + q["question"] for q in questions]
    query_kwargs = {"prompt_name": "query"} if "query" in (model.prompts or {}) else {}
    query_vecs = model.encode(queries, **encode_kwargs, **query_kwargs)

    ranks = []
    for q, qvec in zip(questions, query_vecs):
        ranked = np.argsort(-(chunk_vecs @ qvec))
        ranks.append(first_hit_rank(ranked, chunk_texts, q["evidence"]))

    result = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in CUTOFFS}
    result["mrr@10"] = sum(1 / r for r in ranks if r is not None and r <= 10) / len(ranks)
    result["index_seconds"] = round(index_seconds, 1)
    result["ranks"] = ranks

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    chunk_texts = [c.text for c in load_chunks()]
    questions = json.loads((EVAL_DIR / "testset.json").read_text())
    results = {}
    for name, revision in MODELS:
        results[name] = evaluate(name, revision, chunk_texts, questions)
        r = results[name]
        print(f"{name:42s} hit@1={r['hit@1']:.2f} hit@5={r['hit@5']:.2f} hit@10={r['hit@10']:.2f} "
              f"mrr@10={r['mrr@10']:.3f} index={r['index_seconds']}s", flush=True)
    out = EVAL_DIR / "results" / "embedders.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
