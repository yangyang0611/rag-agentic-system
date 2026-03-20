from openai.types.chat import ChatCompletionToolParam
from mcp.server.fastmcp import FastMCP
from ingester import ingest_url as _ingest_url, collection, model, tavily_client

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
            "score": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]


@mcp.tool()
def web_search(query: str, n_results: int = 3) -> list[dict]:
    """搜尋網路上的即時資訊，回傳相關結果。當向量資料庫沒有相關資料時使用。

    Args:
        query: 搜尋問題或關鍵字
        n_results: 回傳結果數量（預設 3）

    Returns:
        搜尋結果列表，每個包含內容、來源網址與相關度分數
    """
    results = tavily_client.search(query, max_results=n_results)
    return [
        {
            "content": r["content"],
            "url": r["url"],
            "score": round(r.get("score", 0), 4),
        }
        for r in results["results"]
    ]


AGENT_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "query_docs",
            "description": "Search the local vector database for relevant document chunks. Use this when the question is likely about content the user has previously ingested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {"type": "integer", "description": "Number of results", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for up-to-date information. Use this when the question is about recent events, general knowledge, or when the local database is unlikely to have the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {"type": "integer", "description": "Number of results", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "query_docs": lambda args: query_docs(args["query"], args.get("n_results", 3)),
    "web_search": lambda args: web_search(args["query"], args.get("n_results", 3)),
}
