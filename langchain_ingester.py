"""LangChain-based ingestion path — parallel to the hand-built ingester.py.

Uses LangChain Document Loaders and Text Splitters for modular document processing,
while sharing the same ChromaDB collection and embedding model.
"""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader

from ingester import collection, model  # reuse same vector store & embeddings


splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,       # characters (≈500 words)
    chunk_overlap=200,
    length_function=len,
)


def langchain_ingest_url(url: str) -> dict:
    """Ingest a web page using LangChain WebBaseLoader."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [hashlib.md5(f"lc_{url}_{i}".encode()).hexdigest() for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"url": url, "chunk_index": i, "loader": "langchain"} for i in range(len(chunks))],
    )
    return {"url": url, "chunks_stored": len(chunks), "loader": "langchain"}


def langchain_ingest_pdf(path: str, source: str = "") -> dict:
    """Ingest a PDF using LangChain PyMuPDFLoader."""
    source = source or path
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [hashlib.md5(f"lc_{source}_{i}".encode()).hexdigest() for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"url": source, "chunk_index": i, "loader": "langchain"} for i in range(len(chunks))],
    )
    return {"file": source, "chunks_stored": len(chunks), "loader": "langchain"}
