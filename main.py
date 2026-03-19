import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from ingester import ingest_url as _ingest_url, collection, model

openai_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

mcp = FastMCP("rag-mcp")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    n_results: int = 2


class IngestRequest(BaseModel):
    url: str


@mcp.tool()
def ingest_url(url: str) -> dict:
    """爬取指定網頁並將內容切段向量化後存入向量資料庫。

    Args:
        url: 要爬取的網頁網址

    Returns:
        包含網址與已儲存 chunk 數量的結果
    """
    return _ingest_url(url)


@mcp.tool()
def query_docs(query: str, n_results: int = 2) -> list[dict]:
    """根據語意查詢向量資料庫，回傳最相關的文件片段。

    Args:
        query: 搜尋問題或關鍵字
        n_results: 回傳結果數量（預設 2）

    Returns:
        相關文件片段列表，每個包含內容、來源網址與相似度分數
    """
    embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]
    distances = (results["distances"] or [[]])[0]

    return [
        {
            "content": doc,
            "url": meta.get("url", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "score": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/query")
def api_query(req: QueryRequest):
    return query_docs(req.query, req.n_results)


@app.post("/ingest")
def api_ingest(req: IngestRequest):
    return ingest_url(req.url)


@app.post("/answer")
def api_answer(req: QueryRequest):
    chunks = query_docs(req.query, req.n_results)
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    response = openai_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer the user's question based only on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
        ],
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        q = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        results = query_docs(q, n)
        for i, r in enumerate(results, 1):
            print(f"({i}) score: {r['score']}  url: {r['url']}  chunk: {r['chunk_index']}")
            print(r["content"])
            print()
    elif len(sys.argv) > 1 and sys.argv[1] == "ingest":
        url = sys.argv[2]
        print(json.dumps(ingest_url(url), indent=2, ensure_ascii=False))
    else:
        mcp.run()
