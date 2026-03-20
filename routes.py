import json as _json
from typing import cast

from fastapi import UploadFile
from fastapi.responses import FileResponse
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from pydantic import BaseModel

from ingester import ingest_file as _ingest_file, tavily_client
from tools import ingest_url, query_docs, web_search, AGENT_TOOLS, TOOL_DISPATCH


class QueryRequest(BaseModel):
    query: str
    n_results: int = 2
    session_id: str = "default"


class IngestRequest(BaseModel):
    url: str


# ── Memory ──
# 用 dict 存每個 session 的對話記錄，key = session_id，value = messages list
# 這樣同一個 session 的多次請求可以記住之前聊過什麼
memory: dict[str, list] = {}

def register_routes(app, openai_client):
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

    @app.post("/ask-db")
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

    # ── Agent endpoint ──
    # 這是整個 RAG 系統的核心：一個 agentic tool-calling loop。
    # 跟上面的 /answer、/search 不同，Agent 不是寫死流程，
    # 而是讓 LLM 自己決定要呼叫哪些工具、呼叫幾次。
    #
    # 流程概覽：
    #   1. 把用戶問題 + system prompt（策略指令）丟給 LLM
    #   2. LLM 回傳要呼叫的工具（或直接回答）
    #   3. 我們執行工具，把結果塞回對話
    #   4. 重複 2-3，直到 LLM 不再呼叫工具（代表它覺得資訊夠了）
    #   5. 最多跑 5 輪，避免無限迴圈
    @app.delete("/memory/{session_id}")
    def api_clear_memory(session_id: str):
        """清除指定 session 的對話記憶"""
        memory.pop(session_id, None)
        return {"status": "cleared", "session_id": session_id}

    @app.post("/agent")
    def api_agent(req: QueryRequest):
        # 從 memory 取出這個 session 的歷史對話（如果有的話）
        history = memory.get(req.session_id, [])
        messages: list[ChatCompletionMessageParam] = [
            # ↓ 這就是「策略指令」：用自然語言告訴 LLM 行為優先順序
            #   改這句話就能改變 Agent 的決策邏輯，不用動任何程式碼
            {"role": "system", "content": "You are a helpful assistant with access to tools. ALWAYS search the local database (query_docs) first. Only use web_search if the database returns no relevant results. You may call multiple tools if needed. Be direct and concise in your final answer."},
            # ↓ 把歷史對話塞進來，讓 LLM 知道之前聊過什麼
            *history,
            {"role": "user", "content": req.query},
        ]
        all_sources: list[dict] = []
        tools_used: list[str] = []

        print(f"\n{'='*60}")
        print(f"[Agent] New query: {req.query}")
        print(f"{'='*60}")

        # ── Agentic Loop ──
        # 每一輪：問 LLM → 它決定要不要用工具 → 執行工具 → 結果餵回去
        # LLM 不呼叫工具時 = 它準備好回答了 → break
        for round_num in range(5):
            print(f"[Agent] Round {round_num + 1}: Asking LLM...")
            try:
                # 把可用工具列表 (AGENT_TOOLS) 傳給 LLM，讓它自己選
                response = openai_client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages,
                    tools=AGENT_TOOLS,
                )
            except Exception as e:
                # fallback：工具呼叫失敗就讓 LLM 直接回答
                print(f"[Agent] Round {round_num + 1}: Tool call failed ({e}), retrying without tools")
                response = openai_client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages,
                )
                msg = response.choices[0].message
                break
            msg = response.choices[0].message

            # LLM 沒有呼叫任何工具 → 代表它覺得資訊夠了，準備回答
            if not msg.tool_calls:
                print(f"[Agent] Round {round_num + 1}: No tool calls, generating answer")
                break

            # ── 執行 LLM 選擇的工具 ──
            # LLM 回傳的 tool_calls 裡包含：工具名稱 + 參數（都是 LLM 自己決定的）
            # 我們只負責「照做」，然後把結果塞回 messages 讓 LLM 繼續判斷
            messages.append(cast(ChatCompletionMessageParam, msg.to_dict()))
            for tc in msg.tool_calls:
                if not isinstance(tc, ChatCompletionMessageToolCall):
                    continue
                args = _json.loads(tc.function.arguments)
                print(f"[Agent] Round {round_num + 1}: {tc.function.name}({args})")
                # TOOL_DISPATCH 是一個 dict，把工具名稱對應到實際函式
                results = TOOL_DISPATCH[tc.function.name](args)
                print(f"[Agent] Round {round_num + 1}: → {len(results)} results")
                tools_used.append(tc.function.name)
                all_sources.extend(results)
                # 把工具執行結果以 "tool" role 塞回對話，LLM 下一輪會看到
                messages.append(cast(ChatCompletionMessageParam, {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _json.dumps(results, ensure_ascii=False),
                }))
        else:
            # for-else：迴圈跑完 5 輪都沒 break → 強制要求 LLM 給最終答案
            print(f"[Agent] Max rounds reached, forcing final answer")
            messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": "Please provide your final answer now based on all the information gathered."}))
            response = openai_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
            )
            msg = response.choices[0].message

        print(f"[Agent] Done\n{'='*60}\n")

        # 去重：同一段內容可能被不同工具重複回傳
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = s.get("content", "")
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # 標記這次回答的資料來源，讓前端顯示
        if "query_docs" in tools_used and "web_search" in tools_used:
            source_label = "DB + Web"
        elif "query_docs" in tools_used:
            source_label = "DB"
        elif "web_search" in tools_used:
            source_label = "Web"
        else:
            source_label = "LLM"

        # ── 儲存對話記憶 ──
        # 只保留 user 和 assistant 的對話（不存 tool calls，太雜了）
        # 這樣下次同一個 session 問問題時，LLM 會知道之前聊過什麼
        history.append({"role": "user", "content": req.query})
        history.append({"role": "assistant", "content": msg.content})
        memory[req.session_id] = history

        return {
            "answer": f"*[Source: {source_label}]*\n\n{msg.content}",
            "sources": unique_sources,
        }
