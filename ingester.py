import os
import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
import hashlib
import base64
import json
import fitz  # pymupdf
from openai import OpenAI

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


def _get_vision_client() -> OpenAI | None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def _image_to_data_url(image_bytes: bytes, ext: str) -> str:
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext.lower(), "image/png")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _summarize_page_visuals(
    client: OpenAI,
    image_bytes: bytes,
    page_num: int,
    model_name: str,
) -> list[dict]:
    prompt = (
        "You are analyzing a PDF page image. Focus only on non-body-text visual content.\n"
        "Extract charts, diagrams, images, and visible tables. Ignore normal paragraphs unless "
        "they are needed to explain the visual.\n"
        "Return ONLY JSON in this shape:\n"
        '[{"type":"chart|diagram|image|table","title":"short title","summary":"what it shows","key_points":["point 1","point 2"]}]\n'
        "If there is no meaningful visual content, return []."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\nPage: {page_num}"},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes, 'png')}},
                ],
            }
        ],
        max_tokens=800,
    )
    content = response.choices[0].message.content or "[]"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return [{
            "type": "image",
            "title": f"Page {page_num} visual summary",
            "summary": content.strip(),
            "key_points": [],
        }]
    return [item for item in parsed if isinstance(item, dict)]


def extract_pdf_tables(path: str) -> list[dict]:
    """Extract tables from a PDF using PyMuPDF's table detector."""
    doc = fitz.open(path)
    items: list[dict] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            finder = page.find_tables()
            for table_idx, table in enumerate(finder.tables, start=1):
                rows = table.extract()
                if not rows:
                    continue
                cleaned_rows = [
                    [("" if cell is None else str(cell).strip()) for cell in row]
                    for row in rows
                ]
                non_empty = sum(1 for row in cleaned_rows for cell in row if cell)
                if non_empty < 4:
                    continue
                text_rows = [" | ".join(cell or "-" for cell in row) for row in cleaned_rows]
                items.append({
                    "page": page_num,
                    "table_index": table_idx,
                    "content": "\n".join(text_rows),
                })
    finally:
        doc.close()
    return items


def extract_pdf_visual_summaries(
    path: str,
    vision_model: str = "google/gemini-2.5-flash",
    max_pages: int = 8,
) -> list[dict]:
    """Render PDF pages and let a multimodal model describe charts/figures/tables."""
    client = _get_vision_client()
    if client is None:
        return []

    doc = fitz.open(path)
    items: list[dict] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            if page_num > max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            page_png = pix.tobytes("png")
            summaries = _summarize_page_visuals(client, page_png, page_num, vision_model)
            for idx, summary in enumerate(summaries, start=1):
                title = summary.get("title") or f"Page {page_num} visual {idx}"
                key_points = summary.get("key_points") or []
                key_points_text = "\n".join(f"- {point}" for point in key_points if point)
                content = (
                    f"[{summary.get('type', 'visual').upper()}] {title}\n"
                    f"Page: {page_num}\n"
                    f"Summary: {summary.get('summary', '').strip()}"
                ).strip()
                if key_points_text:
                    content += f"\nKey points:\n{key_points_text}"
                items.append({
                    "page": page_num,
                    "visual_index": idx,
                    "content": content,
                    "visual_type": summary.get("type", "visual"),
                })
    finally:
        doc.close()
    return items


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
        metadatas=[{"url": source, "chunk_index": i, "content_type": "text"} for i in range(len(chunks))],
    )
    return {"file": source, "chunks_stored": len(chunks)}


def ingest_pdf_multimodal(
    path: str,
    source: str = "",
    vision_model: str = "google/gemini-2.5-flash",
) -> dict:
    """Ingest text + tables + visual summaries from a PDF into the vector DB."""
    source = source or path

    text = read_pdf(path)
    text_chunks = chunk_text(text)
    table_items = extract_pdf_tables(path)
    visual_items = extract_pdf_visual_summaries(path, vision_model=vision_model)

    documents: list[str] = []
    metadatas: list[dict] = []

    for idx, chunk in enumerate(text_chunks):
        documents.append(chunk)
        metadatas.append({"url": source, "chunk_index": idx, "content_type": "text"})

    for item in table_items:
        documents.append(f"[TABLE]\nSource: {source}\nPage: {item['page']}\n{item['content']}")
        metadatas.append({
            "url": source,
            "page": item["page"],
            "table_index": item["table_index"],
            "content_type": "table",
        })

    for item in visual_items:
        documents.append(f"[VISUAL]\nSource: {source}\n{item['content']}")
        metadatas.append({
            "url": source,
            "page": item["page"],
            "visual_index": item["visual_index"],
            "content_type": item["visual_type"],
        })

    if not documents:
        return {"file": source, "chunks_stored": 0, "tables_stored": 0, "visuals_stored": 0}

    embeddings = model.encode(documents).tolist()
    ids = [hashlib.md5(f"{source}_multimodal_{i}".encode()).hexdigest() for i in range(len(documents))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return {
        "file": source,
        "chunks_stored": len(text_chunks),
        "tables_stored": len(table_items),
        "visuals_stored": len(visual_items),
        "vision_model": vision_model,
    }


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
        metadatas=[{"url": url, "chunk_index": i, "content_type": "text"} for i in range(len(chunks))],
    )

    return {"url": url, "chunks_stored": len(chunks)}
