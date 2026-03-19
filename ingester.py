import os
import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
import hashlib
import fitz  # pymupdf

# 載入 embedding model（第一次會下載，之後 cache）
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 初始化 ChromaDB（存在本地 ./chroma_db 資料夾）
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "webpages",
    metadata={"hnsw:space": "cosine"},
)


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def fetch_webpage(url: str) -> str:
    """用 Tavily extract 爬網頁，回傳純文字內容"""
    response = tavily_client.extract(urls=[url], format="text")
    if not response["results"]:
        raise ValueError(f"Failed to extract content from {url}")
    return response["results"][0]["raw_content"]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把長文字切成有 overlap 的小片段"""
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # overlap 讓相鄰 chunk 有重疊，避免切斷語意

    return chunks


def read_pdf(path: str) -> str:
    """讀取 PDF，回傳純文字內容"""
    doc = fitz.open(path)
    text = "\n".join(str(page.get_text()) for page in doc)
    doc.close()
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    return "\n".join(lines)


def ingest_file(path: str, source: str = "") -> dict:
    """讀取本地 PDF → 切chunks → 向量化 → 存DB"""
    source = source or path
    text = read_pdf(path)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    ids = [hashlib.md5(f"{source}_{i}".encode()).hexdigest() for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"url": source, "chunk_index": i} for i in range(len(chunks))],
    )
    return {"file": source, "chunks_stored": len(chunks)}


def ingest_url(url: str) -> dict:
    """完整流程：爬網頁 → 切chunks → 向量化 → 存DB"""
    # 1. 爬網頁
    text = fetch_webpage(url)

    # 2. 切 chunks
    chunks = chunk_text(text)

    # 3. 向量化（批次處理）
    embeddings = model.encode(chunks).tolist()

    # 4. 建立唯一 ID（用 url + chunk index）
    ids = [hashlib.md5(f"{url}_{i}".encode()).hexdigest() for i in range(len(chunks))]

    # 5. 存進 ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"url": url, "chunk_index": i} for i in range(len(chunks))],
    )

    return {"url": url, "chunks_stored": len(chunks)}
