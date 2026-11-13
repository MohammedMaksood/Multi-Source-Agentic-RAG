# Multi-Source Agentic RAG

An agentic RAG system built with LangGraph. An LLM agent rewrites the user's question, retrieves passages, grades whether they are good enough, retries with a new query when they are not, and only then writes a cited answer. Dense vectors and paper chunks are stored in Weaviate (local Docker by default, with optional Weaviate Cloud); BM25, RRF, and cross-encoder reranking provide hybrid retrieval. The project compares retrieval strategies and measures them with RAGAS. Every run can be traced in LangSmith, and conversations persist in SQLite.

**Status:** Phase 1 (documents only) is built and its evaluation gate is met. All 150 context-precision scores are complete (see Results). Phases 2 and 3 are planned, not built.

## Phases

Each phase adds one source and moves on only when its evaluation gate is met.

| Phase | Scope | Status |
|---|---|---|
| 1 | Documents only: dense search, BM25, RRF fusion, cross-encoder rerank, grade and retry loop, SQLite memory, RAGAS comparison of retrieval modes | Built, evaluation gate passed |
| 2 | A second source (SQL or another vector store) and a router that picks between sources | Planned |
| 3 | Web search | Planned |

## How the agent works

```
question + thread id
  -> load history from SQLite
  -> rewrite into a standalone query (skipped when there is no history)
  -> retrieve top 5 passages (dense | hybrid | hybrid_rerank)
  -> grade: do the passages contain enough to answer?
       yes -> generate a cited answer
       no, retries left (max 2) -> reformulate the query and retrieve again
       no, retries used up -> reply "not enough evidence" (no LLM call)
  -> save to SQLite
```

A question costs 2 LLM calls at minimum and 7 at most. Every call is logged with model, prompt version, tokens, latency and cost. Prompts are versioned files in `prompts/`, and every LLM output is validated against a pydantic schema.

## Models (all pinned)

| Role | Model | Why |
|---|---|---|
| Embedder | `BAAI/bge-m3` (revision `5617a9f`) | Best of four in the bake-off below |
| Vector database | Weaviate (local Docker or Weaviate Cloud) | Persists chunk text, metadata, and BGE-M3 vectors |
| Reranker | `BAAI/bge-reranker-v2-m3` (revision `953dc6f`) | Best of three in the bake-off below |
| Local LLM | `qwen3:4b-instruct-2507-q4_K_M` (Ollama) | Fits a 6 GB GPU beside the retrieval models |
| API LLM and RAGAS judge | `gemini-3.5-flash-lite` | Stable, free tier available |

The LLM backend is switchable between local Ollama and Gemini with `LLM_BACKEND`. The Gemini path is used for judging; the agent runs reported below used the local model.

## Results

The corpus is 19 public arXiv papers on retrieval-augmented generation, split into 948 chunks. All numbers below are produced by `python -m eval.summarize` from the files in `eval/results/`.

### Model bake-off (40 questions)

| Embedder | hit@5 | MRR@10 |
|---|---|---|
| bge-m3 | 0.68 | 0.446 |
| Qwen3-Embedding-0.6B | 0.53 | 0.361 |
| bge-small-en-v1.5 | 0.42 | 0.324 |
| all-MiniLM-L6-v2 | 0.45 | 0.301 |

| Reranker (on the top 30 from bge-m3) | hit@5 | MRR@10 |
|---|---|---|
| none | 0.68 | 0.446 |
| bge-reranker-v2-m3 | 0.93 | 0.790 |
| Qwen3-Reranker-0.6B | 0.80 | 0.657 |
| ms-marco-MiniLM-L6-v2 | 0.83 | 0.650 |

The bake-off used the first question set only, which turned out to share many words with its answers (see Limitations).

### Retrieval only: MRR@10 / hit@5 across five question sets

"Word overlap" is the mean share of a question's words that also appear in the chunk holding its answer. A higher value favors keyword matching.

| Question set | Questions | Word overlap | dense | bm25 | hybrid | hybrid_rerank |
|---|---|---|---|---|---|---|
| easy | 40 | 0.58 | 0.45 / 0.68 | 0.59 / 0.78 | 0.55 / 0.78 | 0.77 / 0.93 |
| medium | 15 | 0.65 | 0.56 / 0.67 | 0.78 / 0.93 | 0.70 / 0.80 | 0.82 / 0.93 |
| hard | 15 | 0.47 | 0.39 / 0.53 | 0.38 / 0.60 | 0.43 / 0.60 | 0.68 / 0.80 |
| harder | 15 | 0.47 | 0.36 / 0.53 | 0.45 / 0.60 | 0.59 / 0.73 | 0.68 / 0.73 |
| paraphrase | 20 | 0.37 | 0.16 / 0.30 | 0.12 / 0.20 | 0.12 / 0.25 | 0.35 / 0.40 |
| **all** | 105 | | 0.39 / 0.56 | 0.48 / 0.64 | 0.48 / 0.65 | 0.67 / 0.78 |

Hybrid with reranking has the best MRR on every set. Hybrid scores above dense search on the four sets with higher word overlap (0.47 to 0.65) and below it on the paraphrase set (0.37), so the benefit of adding BM25 depends on how users phrase their questions.

### End to end with the agent (50 hard questions, local model)

| Mode | Gave up | Right evidence in final 5 passages | Avg seconds per question |
|---|---|---|---|
| dense | 26/50 | 21/50 | 6.7 |
| hybrid | 19/50 | 26/50 | 6.5 |
| hybrid_rerank | 15/50 | 36/50 | 6.1 |

### RAGAS (judge: `gemini-3.5-flash-lite`)

| Metric | dense | hybrid | hybrid_rerank |
|---|---|---|---|
| context_recall | 0.60 (n=50) | 0.66 (n=50) | 0.88 (n=50) |
| context_precision | 0.40 (n=50) | 0.41 (n=50) | 0.63 (n=50) |
| factual_correctness | 0.15 (n=50) | 0.20 (n=50) | 0.21 (n=50) |
| faithfulness | 0.88 (n=24) | 0.85 (n=31) | 0.83 (n=35) |

Faithfulness is computed only on answers the agent gave, and each mode answered a different number of questions, so the row above is not a fair comparison. The paired table below compares modes on the same questions.

Paired differences, mean and 95% bootstrap interval:

| Metric | Questions | rerank vs dense | hybrid vs dense | rerank vs hybrid |
|---|---|---|---|---|
| context_recall | 50 | +0.28 [+0.12, +0.42] | +0.06 [-0.06, +0.18] | +0.22 [+0.08, +0.36] |
| context_precision | 50 | +0.22 [+0.08, +0.35] | +0.01 [-0.09, +0.10] | +0.21 [+0.10, +0.34] |
| factual_correctness | 50 | +0.06 [-0.02, +0.14] | +0.06 [-0.01, +0.12] | +0.01 [-0.08, +0.09] |
| faithfulness | 20 | +0.02 [-0.13, +0.17] | +0.03 [-0.12, +0.15] | -0.01 [-0.09, +0.08] |

### What the results support

- **Reranking helps.** Context recall improves by +0.28 over dense search [+0.12, +0.42], and context precision improves by +0.22 [+0.08, +0.35]. Both paired intervals stay above zero. The retrieval-only tables show the same on all five sets.
- **Hybrid over dense is not shown.** The interval includes zero on every RAGAS metric, and hybrid lost to dense on the paraphrase set.
- **Answer-level gains are small and not significant.** The local 4B model writes the answers and also decides when to give up, which caps what better retrieval can achieve end to end.
- **Faithfulness shows no sign of dropping** with better retrieval, but only 20 questions were answered in all three modes.
- **Phase 1 gate passed.** Reranking beats dense on both context recall and context precision. Paired faithfulness has no observed drop (+0.02), but its interval is wide [-0.13, +0.17] and only 20 questions were answered in all three modes, so that check remains limited.

## Limitations

- **Small, author-built evaluation.** The 105 retrieval questions and the 50 reference answers were drafted with an AI assistant (Claude) and have not been independently reviewed. Most questions share many words with their answers, which favors keyword search. Treat absolute scores as indicative.
- **Small samples.** Several paired comparisons use 16 to 50 questions, and their intervals are wide.
- **Noisy judge.** RAGAS metrics are scored by an LLM. `factual_correctness` gave 0.0 to at least 7 answers that closely match their reference, so it is the least reliable metric here. It is reported but not used for conclusions.
- **Weak local grader.** On passages that did contain the labelled evidence, the 4B grader accepted them 75% of the time. Some "gave up" results are grader mistakes, not retrieval failures.
- **One small corpus.** 19 papers and 948 chunks. Reranking gains may be smaller on a larger corpus, where the first stage misses more.
- **Prompt injection.** A hijacked query rewrite only changes retrieval and cannot reach the answer step directly, and the local model resisted one instruction hidden in a retrieved passage. That is a single sample, not a guarantee. Retrieved text is untrusted input.

## Run it

Python 3.11, an NVIDIA GPU with about 6 GB of VRAM, and [Ollama](https://ollama.com) for the local model.

```bash
python3.11 -m venv ~/.venvs/msa && source ~/.venvs/msa/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in your keys
docker compose up -d                                   # local Weaviate on 127.0.0.1:8080 (skip when using Cloud)
python scripts/download_corpus.py                      # 19 arXiv papers into data/papers/
ollama pull qwen3:4b-instruct-2507-q4_K_M
python -m scripts.index_weaviate                      # index chunks without calling an LLM

python -m agent.cli --thread demo "How does HyDE retrieve documents?"
python -m pytest tests -q

python -m eval.compare_modes testset testset_hard      # retrieval-only comparison
python -m eval.run_agent                               # agent over 50 questions x 3 modes (local model)
python -m eval.run_ragas --max-calls 250 --rpm 10      # RAGAS judging, resumable
python -m eval.summarize                               # prints the tables above
```

By default, the app connects to local Weaviate using `WEAVIATE_HOST`, `WEAVIATE_HTTP_PORT`, and `WEAVIATE_GRPC_PORT`. To use Weaviate Cloud instead, set `WEAVIATE_URL` to your cluster's REST endpoint and `WEAVIATE_API_KEY` in your local `.env`; when both are set, the cloud connection takes precedence and Docker is not required. Keep the API key private. Cloud usage may incur charges, and indexing uploads your paper chunks and vectors to that cluster.

On first use with a database, the agent creates a corpus-versioned Weaviate collection and indexes the paper chunks using the existing BGE-M3 embeddings. Later runs reuse that collection. Local data persists in the named `weaviate_data` Docker volume. To view local data, connect a Weaviate browser console to the local instance; Weaviate's hosted Explorer is limited to Cloud collections.

On WSL, keep the virtual environment on the Linux filesystem, not under `/mnt/c`. Importing PyTorch from a Windows-mounted drive took 88 seconds here, against 12 seconds on the Linux disk.

## Layout

```
agent/       LangGraph agent, CLI, output schemas
retrieval/   chunking, dense, BM25, RRF fusion, reranking
llm/         one client for Ollama and Gemini, with call logging
prompts/     versioned prompt files
eval/        question sets, reference answers, comparison and RAGAS scripts, saved results
scripts/     corpus download
tests/       unit tests
```

## Privacy and cost notes

- **LangSmith tracing is off by default.** When enabled (`LANGSMITH_TRACING=true`), prompts and questions are sent to LangSmith.
- **Gemini free tier.** The judge ran on the free tier, which can use submitted content to improve Google products. Only public paper text and the evaluation questions were sent.
- **RAGAS analytics are disabled** in `eval/run_ragas.py` through `RAGAS_DO_NOT_TRACK`.
- Judging 150 samples across four metrics takes about 1,500 Gemini calls in total, about 1.4M input tokens for the first 955. The script paces requests, caps the calls per run, and resumes where it stopped.
