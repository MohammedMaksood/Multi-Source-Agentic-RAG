"""Index the paper corpus into Weaviate without making an LLM call."""
import os

from retrieval.retrievers import Retriever


def main() -> None:
    retriever = Retriever()
    try:
        print(f"Weaviate collection: {retriever.collection_name}")
        print(f"Connected to: {retriever.database_location}")
        print(f"Indexed chunks: {retriever.indexed_count()}")
        if not os.getenv("WEAVIATE_URL", "").strip():
            host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
            port = os.getenv("WEAVIATE_HTTP_PORT", "8080")
            print(f"Local Weaviate API: http://{host}:{port}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
