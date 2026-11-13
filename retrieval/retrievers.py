"""Dense, BM25, hybrid (RRF) and hybrid + rerank retrieval over local Weaviate."""
import hashlib
import os
import re
import uuid
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval.chunking import Chunk, load_chunks

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
EMBEDDER = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")
RERANKER = ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
CANDIDATES = 30
RRF_K = 60
HALF_PRECISION = {"dtype": torch.float16}  # halves GPU memory so a local LLM fits beside retrieval models
COLLECTION_PREFIX = "PaperChunk"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        if not self.chunks:
            raise RuntimeError("No paper chunks found. Download the corpus before starting retrieval.")

        texts = [c.text for c in self.chunks]
        corpus_digest = hashlib.sha256(
            "\n".join(f"{c.paper_id}:{c.index}:{c.text}" for c in self.chunks).encode("utf-8")
        ).hexdigest()[:12]
        self.collection_name = f"{COLLECTION_PREFIX}{corpus_digest}"
        self.chunk_lookup = {(c.paper_id, c.index): i for i, c in enumerate(self.chunks)}

        try:
            import weaviate
            from weaviate.classes.config import Configure, DataType, Property
        except ImportError as exc:
            raise RuntimeError(
                "Weaviate support is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        try:
            self.client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to local Weaviate at {host}:{http_port}. "
                "Start it with `docker compose up -d` and try again. "
                f"Details: {exc}"
            ) from exc

        try:
            if self.client.collections.exists(self.collection_name):
                self.collection = self.client.collections.get(self.collection_name)
            else:
                self.collection = self.client.collections.create(
                    name=self.collection_name,
                    vector_config=Configure.Vectors.self_provided(),
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="paper_id", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                    ],
                )

            count = self.collection.aggregate.over_all(total_count=True).total_count
            if count not in (0, len(self.chunks)):
                raise RuntimeError(
                    f"Weaviate collection {self.collection_name} contains {count} objects; "
                    f"expected {len(self.chunks)}. Inspect the collection before rebuilding it."
                )

            self.embedder = SentenceTransformer(
                EMBEDDER[0], revision=EMBEDDER[1], device="cuda", model_kwargs=HALF_PRECISION
            )
            self.bm25 = BM25Okapi([tokenize(t) for t in texts])
            if count == 0:
                self._index_chunks(texts)
            # Load the large reranker only when hybrid_rerank is requested.
            # Indexing and dense/BM25 retrieval do not need it.
            self.reranker = None
        except Exception:
            self.client.close()
            raise

    def _initial_vectors(self, texts: list[str]) -> np.ndarray:
        """Reuse the existing local embedding cache for first import when available."""
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        path = INDEX_DIR / f"bge-m3-fp16-{digest}.npy"
        if path.exists():
            try:
                vectors = np.load(path)
                if vectors.shape == (len(texts), 1024):
                    return vectors
            except (OSError, ValueError):
                pass
        return self.embedder.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )

    def _index_chunks(self, texts: list[str]) -> None:
        vectors = self._initial_vectors(texts)
        with self.collection.batch.fixed_size(batch_size=100) as batch:
            for chunk, vector in zip(self.chunks, vectors):
                stable_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.collection_name}:{chunk.paper_id}:{chunk.index}",
                )
                batch.add_object(
                    properties={
                        "text": chunk.text,
                        "paper_id": chunk.paper_id,
                        "chunk_index": chunk.index,
                    },
                    vector=vector.astype(np.float32, copy=False).tolist(),
                    uuid=stable_id,
                )
        failed = self.collection.batch.failed_objects
        if failed:
            raise RuntimeError(
                f"Weaviate indexed {len(self.chunks) - len(failed)} of {len(self.chunks)} chunks; "
                f"{len(failed)} objects failed. Inspect the Weaviate logs before retrying."
            )

    def _dense(self, query: str) -> list[int]:
        query_vec = self.embedder.encode(query, normalize_embeddings=True, show_progress_bar=False)
        result = self.collection.query.near_vector(
            near_vector=query_vec.tolist(),
            limit=CANDIDATES,
            return_properties=["paper_id", "chunk_index"],
        )
        ids = []
        for obj in result.objects:
            key = (obj.properties["paper_id"], int(obj.properties["chunk_index"]))
            if key in self.chunk_lookup:
                ids.append(self.chunk_lookup[key])
        return ids

    def _bm25(self, query: str) -> list[int]:
        return np.argsort(-self.bm25.get_scores(tokenize(query)))[:CANDIDATES].tolist()

    def _hybrid(self, query: str) -> list[int]:
        return reciprocal_rank_fusion([self._dense(query), self._bm25(query)])[:CANDIDATES]

    def _hybrid_rerank(self, query: str) -> list[int]:
        candidates = self._hybrid(query)
        if self.reranker is None:
            self.reranker = CrossEncoder(
                RERANKER[0], revision=RERANKER[1], device="cuda", model_kwargs=HALF_PRECISION
            )
        scores = self.reranker.predict(
            [(query, self.chunks[i].text) for i in candidates], batch_size=8, show_progress_bar=False
        )
        return [candidates[i] for i in np.argsort(-scores)]

    def retrieve(self, query: str, mode: str, k: int = 5) -> list[Chunk]:
        rankers = {
            "dense": self._dense,
            "bm25": self._bm25,
            "hybrid": self._hybrid,
            "hybrid_rerank": self._hybrid_rerank,
        }
        if mode not in rankers:
            raise ValueError(f"unknown mode {mode!r}, expected one of {list(rankers)}")
        return [self.chunks[i] for i in rankers[mode](query)[:k]]

    def close(self) -> None:
        self.client.close()

    def indexed_count(self) -> int:
        return self.collection.aggregate.over_all(total_count=True).total_count
