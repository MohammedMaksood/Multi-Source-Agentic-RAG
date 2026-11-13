from retrieval.retrievers import reciprocal_rank_fusion, tokenize


def test_rrf_rewards_chunks_found_by_both_rankers():
    # chunk 2 is in both rankings, so it beats chunk 1 (first in one only) and chunk 3 (second in one only)
    assert reciprocal_rank_fusion([[1, 2], [2, 3]]) == [2, 1, 3]


def test_rrf_keeps_single_ranking_order():
    assert reciprocal_rank_fusion([[5, 4, 3]]) == [5, 4, 3]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Self-RAG: BM25, EM@500") == ["self", "rag", "bm25", "em", "500"]
