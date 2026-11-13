"""Index the local paper corpus into Weaviate without making an LLM call."""
from retrieval.retrievers import Retriever


def main() -> None:
    retriever = Retriever()
    try:
        print(f"Weaviate collection: {retriever.collection_name}")
        print(f"Indexed chunks: {retriever.indexed_count()}")
        print("Local Weaviate API: http://127.0.0.1:8080")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
