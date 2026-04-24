"""Fully LangChain-based query path.

Uses the same Chroma vectorstore + HuggingFaceEmbeddings defined in
langchain_ingester.py to run similarity search — no dependency on ingester.py.
"""

from langchain_ingester import vectorstore


def langchain_query_docs(query: str, n_results: int = 2) -> list[dict]:
    """Semantic search via LangChain Chroma.similarity_search_with_score."""
    results = vectorstore.similarity_search_with_score(query, k=n_results)
    return [
        {
            "content": doc.page_content,
            "url": doc.metadata.get("url", ""),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score": round(1 - distance, 4),
        }
        for doc, distance in results
    ]
