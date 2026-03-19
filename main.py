import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from typing import cast
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCall
from mcp.server.fastmcp import FastMCP
from ingester import ingest_url as _ingest_url, ingest_file as _ingest_file, collection, model, tavily_client

openai_client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

mcp = FastMCP("rag-mcp")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


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


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/query")
def api_query(req: QueryRequest):
    return query_docs(req.query, req.n_results)


@app.post("/ingest")
def api_ingest(req: IngestRequest):
    return ingest_url(req.url)


@app.post("/upload")
def api_upload(file: UploadFile):
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    result = _ingest_file(tmp_path, source=file.filename or "unknown.pdf")
    os.unlink(tmp_path)
    return result


@app.post("/answer")
def api_answer(req: QueryRequest):
    chunks = query_docs(req.query, req.n_results)
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    response = openai_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context and your own knowledge to answer. Be direct and concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
        ],
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


@app.post("/search")
def api_search(req: QueryRequest):
    results = tavily_client.search(req.query, max_results=req.n_results)
    context = "\n\n".join(f"[{i+1}] {r['content']}" for i, r in enumerate(results["results"]))
    response = openai_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context and your own knowledge to answer. Be direct and concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
        ],
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [{"url": r["url"], "content": r["content"], "score": round(r.get("score", 0), 4)} for r in results["results"]],
    }


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


@app.post("/agent")
def api_agent(req: QueryRequest):
    import json as _json

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. ALWAYS search the local database (query_docs) first. Only use web_search if the database returns no relevant results. You may call multiple tools if needed. Be direct and concise in your final answer."},
        {"role": "user", "content": req.query},
    ]
    all_sources: list[dict] = []
    tools_used: list[str] = []

    print(f"\n{'='*60}")
    print(f"[Agent] New query: {req.query}")
    print(f"{'='*60}")

    # Agent loop: let LLM decide which tools to call (max 5 rounds)
    for round_num in range(5):
        print(f"[Agent] Round {round_num + 1}: Asking LLM...")
        try:
            response = openai_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                tools=AGENT_TOOLS,
            )
        except Exception as e:
            # Groq tool_use_failed: retry without tools
            print(f"[Agent] Round {round_num + 1}: Tool call failed ({e}), retrying without tools")
            response = openai_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
            )
            msg = response.choices[0].message
            break
        msg = response.choices[0].message

        # No tool calls → LLM is ready to answer
        if not msg.tool_calls:
            print(f"[Agent] Round {round_num + 1}: No tool calls, generating answer")
            break

        # Execute each tool the LLM chose
        messages.append(cast(ChatCompletionMessageParam, msg.to_dict()))
        for tc in msg.tool_calls:
            if not isinstance(tc, ChatCompletionMessageToolCall):
                continue
            args = _json.loads(tc.function.arguments)
            print(f"[Agent] Round {round_num + 1}: {tc.function.name}({args})")
            results = TOOL_DISPATCH[tc.function.name](args)
            print(f"[Agent] Round {round_num + 1}: → {len(results)} results")
            tools_used.append(tc.function.name)
            all_sources.extend(results)
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _json.dumps(results, ensure_ascii=False),
            }))
    else:
        print(f"[Agent] Max rounds reached, forcing final answer")
        messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": "Please provide your final answer now based on all the information gathered."}))
        response = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
        )
        msg = response.choices[0].message

    print(f"[Agent] Done\n{'='*60}\n")

    # Deduplicate sources by content
    seen = set()
    unique_sources = []
    for s in all_sources:
        key = s.get("content", "")
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    # Build source label from tools used
    if "query_docs" in tools_used and "web_search" in tools_used:
        source_label = "DB + Web"
    elif "query_docs" in tools_used:
        source_label = "DB"
    elif "web_search" in tools_used:
        source_label = "Web"
    else:
        source_label = "LLM"

    return {
        "answer": f"*[Source: {source_label}]*\n\n{msg.content}",
        "sources": unique_sources,
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
