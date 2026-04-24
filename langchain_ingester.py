"""Fully LangChain-based ingestion.

Uses LangChain Document Loaders, Text Splitters, HuggingFace Embeddings,
and Chroma vector store — no dependency on ingester.py.
"""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vectorstore = Chroma(
    collection_name="webpages",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)


def langchain_ingest_url(url: str) -> dict:
    """Ingest a web page using LangChain WebBaseLoader."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.metadata = {"url": url, "chunk_index": i, "loader": "langchain"}

    ids = [hashlib.md5(f"lc_{url}_{i}".encode()).hexdigest() for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)
    return {"url": url, "chunks_stored": len(chunks), "loader": "langchain"}


def langchain_ingest_pdf(path: str, source: str = "") -> dict:
    """Ingest a PDF using LangChain PyMuPDFLoader."""
    source = source or path
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.metadata = {"url": source, "chunk_index": i, "loader": "langchain"}

    ids = [hashlib.md5(f"lc_{source}_{i}".encode()).hexdigest() for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)
    return {"file": source, "chunks_stored": len(chunks), "loader": "langchain"}
