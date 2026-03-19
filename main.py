from mcp.server.fastmcp import FastMCP
from ingester import ingest_url as _ingest_url, collection, model

mcp = FastMCP("rag-mcp")


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
            "score": round(1 - dist, 4),  # 轉換為相似度（越高越相關）
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]


# 從cli傳入query
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        q = " ".join(sys.argv[2:])
        results = query_docs(q)
        for i, r in enumerate(results, 1):
            print(f"({i}) score: {r['score']}  url: {r['url']}  chunk: {r['chunk_index']}")
            print(r["content"])
            print()
    elif len(sys.argv) > 1 and sys.argv[1] == "ingest":
        url = sys.argv[2]
        print(json.dumps(ingest_url(url), indent=2, ensure_ascii=False))
    else:
        mcp.run()
