import httpx
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

# 載入 embedding model（第一次會下載，之後 cache）
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 初始化 ChromaDB（存在本地 ./chroma_db 資料夾）
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "webpages",
    metadata={"hnsw:space": "cosine"},
)


def fetch_webpage(url: str) -> str:
    """爬網頁，回傳純文字內容"""
    headers = {"User-Agent": "rag-mcp/0.1 (educational project; httpx)"}
    response = httpx.get(url, timeout=15, follow_redirects=True, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # 移除雜訊標籤
    for tag in soup(["script", "style", "nav", "footer", "header", "table", "sup", "[edit]"]):
        tag.decompose()

    # 優先抓主要內容區塊（Wikipedia 用 #mw-content-text，一般網頁用 article/main）
    main = soup.find(id="mw-content-text") or soup.find("article") or soup.find("main") or soup

    text = main.get_text(separator="\n")

    # 過濾掉太短的行（導航按鈕、標籤等雜訊通常很短）
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    return "\n".join(lines)


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
