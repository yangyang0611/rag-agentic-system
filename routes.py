import json as _json
from typing import cast

from fastapi import UploadFile
from fastapi.responses import FileResponse
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from pydantic import BaseModel

from ingester import ingest_file as _ingest_file, ingest_pdf_multimodal as _ingest_pdf_multimodal, tavily_client
from tools import ingest_url, query_docs, web_search, AGENT_TOOLS, TOOL_DISPATCH
from langchain_ingester import langchain_ingest_url, langchain_ingest_pdf
from langgraph_agent import build_agent_graph
from structured_ingester import (
    ingest_pdf_structured, ingest_url_structured,
    get_all_toc, find_section,
)


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

# ── Agent State ──
# Human-in-the-loop：Agent 每執行完一輪工具就暫停，等用戶決定要不要繼續
# 這個 dict 存放每個 session 暫停時的完整狀態，讓 /agent/continue 可以接著跑
agent_state: dict[str, dict] = {}

# ── LangGraph Agent State ──
langgraph_state: dict[str, dict] = {}

def register_routes(app, openai_client):
    MODEL = "google/gemini-2.5-flash"

    def chat(messages, **kwargs):
        """共用的 LLM 呼叫，省掉重複的 model + client 設定
        **kwargs 會把額外的具名參數（如 tools=AGENT_TOOLS）原封不動傳給 create()
        這樣不用為每個參數都寫一個形參，未來要傳 temperature 等也不用改這裡
        """
        # **kwargs 展開 dict：chat(messages, tools=X) → create(model=..., messages=..., tools=X)
        return openai_client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=4096, **kwargs
        )

    @app.get("/")
    def index():
        return FileResponse("index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
        result = _ingest_pdf_multimodal(tmp_path, source=file.filename or "unknown.pdf")
        os.unlink(tmp_path)
        return result

    @app.post("/upload-multimodal")
    def api_upload_multimodal(file: UploadFile):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        result = _ingest_pdf_multimodal(tmp_path, source=file.filename or "unknown.pdf")
        os.unlink(tmp_path)
        return result

    @app.post("/ask-db")
    def api_answer(req: QueryRequest):
        chunks = query_docs(req.query, req.n_results)
        context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
        response = chat([
            {"role": "system", "content": "You are a helpful assistant. Use the provided context and your own knowledge to answer. Be direct and concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
        ])
        return {
            "answer": response.choices[0].message.content,
            "sources": chunks,
        }

    @app.post("/search")
    def api_search(req: QueryRequest):
        results = tavily_client.search(req.query, max_results=req.n_results)
        context = "\n\n".join(f"[{i+1}] {r['content']}" for i, r in enumerate(results["results"]))
        response = chat([
            {"role": "system", "content": "You are a helpful assistant. Use the provided context and your own knowledge to answer. Be direct and concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
        ])
        return {
            "answer": response.choices[0].message.content,
            "sources": [{"url": r["url"], "content": r["content"], "score": round(r.get("score", 0), 4)} for r in results["results"]],
        }

    @app.delete("/memory/{session_id}")
    def api_clear_memory(session_id: str):
        memory.pop(session_id, None)
        agent_state.pop(session_id, None)
        return {"status": "cleared", "session_id": session_id}

    # ── 共用：跑一輪 agent（問 LLM → 執行工具 → 回傳結果或暫停） ──
    def run_agent_round(state: dict) -> dict:
        """執行一輪 agent loop。
        回傳 status="thinking" 表示暫停等用戶確認，status="done" 表示最終回答。
        """
        messages = state["messages"]
        all_sources = state["all_sources"]
        tools_used = state["tools_used"]
        round_num = state["round"]

        if round_num >= 5:
            # 已達最大輪數，強制生成最終回答
            print(f"[Agent] Max rounds reached, forcing final answer")
            messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": "Please provide your final answer now based on all the information gathered."}))
            response = chat(messages)
            return finish_agent(state, response.choices[0].message)

        print(f"[Agent] Round {round_num + 1}: Asking LLM...")
        try:
            response = chat(messages, tools=AGENT_TOOLS)
        except Exception as e:
            print(f"[Agent] Round {round_num + 1}: Tool call failed ({e}), retrying without tools")
            response = chat(messages)
            return finish_agent(state, response.choices[0].message)

        msg = response.choices[0].message

        # LLM 沒呼叫工具 → 準備好回答了
        if not msg.tool_calls:
            print(f"[Agent] Round {round_num + 1}: No tool calls, generating answer")
            return finish_agent(state, msg)

        # ── 執行工具，但執行完後暫停，等用戶確認 ──
        round_tools = []
        round_sources = []
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
            round_tools.append({"name": tc.function.name, "args": args})
            round_sources.extend(results)
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _json.dumps(results, ensure_ascii=False),
            }))

        state["round"] = round_num + 1

        # ── Reflection：LLM 自我評估搜尋結果 ──
        print(f"[Agent] Round {round_num + 1}: Reflecting...")
        reflection_response = chat([
            {"role": "system", "content": (
                "You are a critical evaluator. Given the search results so far, assess:\n"
                "1. Is the information SUFFICIENT to answer the original question?\n"
                "2. Are there any CONTRADICTIONS in the results?\n"
                "3. Should we search from a DIFFERENT ANGLE or with different keywords?\n\n"
                "Respond in this JSON format:\n"
                '{"sufficient": true/false, "reason": "brief explanation", "next_action": "answer" or "search_again", "suggested_query": "new query if searching again"}'
            )},
            {"role": "user", "content": (
                f"Original question: {state['original_query']}\n\n"
                f"Results gathered so far ({len(all_sources)} chunks from {round_num + 1} rounds):\n"
                + "\n".join(f"- {s.get('content', '')[:200]}" for s in all_sources[-6:])
            )},
        ])
        reflection_text = reflection_response.choices[0].message.content.strip()
        print(f"[Agent] Reflection: {reflection_text[:200]}")

        # 如果 reflection 建議換角度搜，把建議注入 messages
        try:
            reflection = _json.loads(reflection_text)
            if reflection.get("next_action") == "search_again" and reflection.get("suggested_query"):
                messages.append(cast(ChatCompletionMessageParam, {
                    "role": "user",
                    "content": f"The previous results were insufficient. Please search again with a different angle: {reflection['suggested_query']}",
                }))
        except _json.JSONDecodeError:
            reflection_text = '{"sufficient": true, "reason": "could not parse", "next_action": "answer"}'

        # ── 暫停！回傳中間結果 + reflection，等用戶按 Continue 或 Stop ──
        print(f"[Agent] Round {round_num + 1}: Paused, waiting for user decision")
        return {
            "status": "thinking",
            "round": round_num + 1,
            "tools_called": round_tools,
            "sources": round_sources,
            "reflection": reflection_text,
        }

    def finish_agent(state: dict, msg) -> dict:
        """Agent 結束：去重 sources、標記來源、儲存記憶、回傳最終答案"""
        all_sources = state["all_sources"]
        tools_used = state["tools_used"]
        session_id = state["session_id"]
        original_query = state["original_query"]

        print(f"[Agent] Done\n{'='*60}\n")

        # 去重
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = s.get("content", "")
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # 來源標籤
        has_db = "query_docs" in tools_used
        has_web = "web_search" in tools_used
        has_struct = "query_structured" in tools_used
        labels = []
        if has_db:
            labels.append("DB")
        if has_struct:
            labels.append("Structured")
        if has_web:
            labels.append("Web")
        source_label = " + ".join(labels) if labels else "LLM"

        # 儲存對話記憶
        history = memory.get(session_id, [])
        history.append({"role": "user", "content": original_query})
        history.append({"role": "assistant", "content": msg.content})
        memory[session_id] = history

        # 清除暫存的 agent state
        agent_state.pop(session_id, None)

        return {
            "status": "done",
            "answer": f"*[Source: {source_label}]*\n\n{msg.content}",
            "sources": unique_sources,
        }

    # ── Agent endpoint（Human-in-the-loop 版）──
    # 流程：
    #   POST /agent       → 第一輪：query rewrite + 執行工具 → 暫停，回傳中間結果
    #   POST /agent/continue → 用戶按 Continue → 繼續下一輪
    #   POST /agent/stop     → 用戶按 Stop → 強制生成最終答案
    #
    # 每一輪執行完工具後暫停，讓用戶看到「Agent 用了什麼工具、搜到什麼結果」
    # 用戶可以決定要不要讓 Agent 繼續思考，這就是 human-in-the-loop
    @app.post("/agent")
    def api_agent(req: QueryRequest):
        # ── Step 0: Query Rewrite ──
        history = memory.get(req.session_id, [])
        rewrite_response = chat([
            {"role": "system", "content": (
                "You are a query rewriter. Rewrite the user's question into a better search query "
                "optimized for both vector database semantic search and web search. "
                "Keep the original language if it's not English. "
                "If there is conversation history, resolve pronouns and references (e.g. 'it', 'that', 'this') using context. "
                "Output ONLY the rewritten query, nothing else."
            )},
            *history,
            {"role": "user", "content": req.query},
        ])
        rewritten_query = rewrite_response.choices[0].message.content.strip()
        print(f"\n{'='*60}")
        print(f"[Agent] Original query: {req.query}")
        print(f"[Agent] Rewritten query: {rewritten_query}")
        print(f"{'='*60}")

        # 初始化 agent state，存起來讓 /continue 和 /stop 可以接著用
        state = {
            "session_id": req.session_id,
            "original_query": req.query,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to tools. ALWAYS search the local database (query_docs) first. If the question targets a specific section or topic, also try query_structured for precise section retrieval. Only use web_search if local sources return no relevant results. You may call multiple tools if needed. Be direct and concise in your final answer."},
                *history,
                {"role": "user", "content": f"Original question: {req.query}\n\nOptimized search query: {rewritten_query}"},
            ],
            "all_sources": [],
            "tools_used": [],
            "round": 0,
        }
        agent_state[req.session_id] = state

        return run_agent_round(state)

    class SessionRequest(BaseModel):
        session_id: str = "default"

    @app.post("/agent/continue")
    def api_agent_continue(req: SessionRequest):
        """用戶按 Continue → Agent 繼續下一輪工具呼叫"""
        state = agent_state.get(req.session_id)
        if not state:
            return {"status": "error", "message": "No active agent session"}
        return run_agent_round(state)

    @app.post("/agent/stop")
    def api_agent_stop(req: SessionRequest):
        """用戶按 Stop → 強制讓 Agent 用目前收集到的資訊生成最終答案"""
        state = agent_state.get(req.session_id)
        if not state:
            return {"status": "error", "message": "No active agent session"}
        print(f"[Agent] User stopped, forcing final answer")
        messages = state["messages"]
        messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": "Please provide your final answer now based on all the information gathered."}))
        response = chat(messages)
        return finish_agent(state, response.choices[0].message)

    # ── LangChain Ingestion Endpoints ──

    @app.post("/ingest/langchain")
    def api_ingest_langchain(req: IngestRequest):
        return langchain_ingest_url(req.url)

    @app.post("/upload/langchain")
    def api_upload_langchain(file: UploadFile):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        result = langchain_ingest_pdf(tmp_path, source=file.filename or "unknown.pdf")
        os.unlink(tmp_path)
        return result

    # ── LangGraph Agent Endpoints (with Reflection) ──

    run_initial, run_continue, run_stop = build_agent_graph(openai_client)

    @app.post("/agent-v2")
    def api_agent_v2(req: QueryRequest):
        """LangGraph agent with reflection and HITL."""
        history = memory.get(req.session_id, [])
        state, response_data = run_initial(req.session_id, req.query, history)

        if state["status"] == "paused":
            langgraph_state[req.session_id] = state
        elif state["status"] == "done":
            # Save to conversation memory
            hist = memory.get(req.session_id, [])
            hist.append({"role": "user", "content": req.query})
            hist.append({"role": "assistant", "content": response_data.get("answer", "")})
            memory[req.session_id] = hist
            langgraph_state.pop(req.session_id, None)

        return response_data

    @app.post("/agent-v2/continue")
    def api_agent_v2_continue(req: SessionRequest):
        state = langgraph_state.get(req.session_id)
        if not state:
            return {"status": "error", "message": "No active LangGraph agent session"}

        state, response_data = run_continue(state)

        if state["status"] == "paused":
            langgraph_state[req.session_id] = state
        elif state["status"] == "done":
            hist = memory.get(req.session_id, [])
            hist.append({"role": "user", "content": state["original_query"]})
            hist.append({"role": "assistant", "content": response_data.get("answer", "")})
            memory[req.session_id] = hist
            langgraph_state.pop(req.session_id, None)

        return response_data

    # ── Structured (Vectorless) Retrieval Endpoints ──

    @app.post("/ingest-structured")
    def api_ingest_structured(req: IngestRequest):
        """Ingest a URL with structural parsing (heading-based tree)."""
        return ingest_url_structured(req.url)

    @app.post("/upload-structured")
    def api_upload_structured(file: UploadFile):
        """Upload PDF with structural parsing."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        result = ingest_pdf_structured(tmp_path, source=file.filename or "unknown.pdf")
        os.unlink(tmp_path)
        return result

    @app.post("/query-structured")
    def api_query_structured(req: QueryRequest):
        """Query using LLM-based routing over document structure (TOC)."""
        toc = get_all_toc()
        if not toc:
            return {"sections": [], "message": "No structured documents indexed."}

        # Format TOC for LLM
        toc_text = "\n".join(
            f"{'  ' * (t['level'] - 1)}{i+1}. {t['title']}  [path: {t['path']}]"
            for i, t in enumerate(toc)
        )

        # LLM picks the most relevant sections
        routing_response = chat([
            {"role": "system", "content": (
                "You are a document section router. Given a document's table of contents "
                "and a user query, select the 1-3 most relevant sections.\n\n"
                "Return ONLY a JSON array of the section paths, e.g.:\n"
                '["Experience > NVIDIA", "Skills"]\n\n'
                "If no section is relevant, return an empty array []."
            )},
            {"role": "user", "content": f"Table of Contents:\n{toc_text}\n\nQuery: {req.query}"},
        ])

        # Parse LLM response
        import json as _json
        try:
            selected_paths = _json.loads(routing_response.choices[0].message.content.strip())
        except _json.JSONDecodeError:
            selected_paths = []

        # Retrieve section content
        sections = []
        for path in selected_paths[:req.n_results]:
            node = find_section(path)
            if node:
                sections.append({
                    "content": node.get_section_content(),
                    "source": node.source,
                    "section_path": node.path,
                    "page": node.page,
                })

        return {"sections": sections, "toc": [t["title"] for t in toc]}

    @app.post("/agent-v2/stop")
    def api_agent_v2_stop(req: SessionRequest):
        state = langgraph_state.get(req.session_id)
        if not state:
            return {"status": "error", "message": "No active LangGraph agent session"}

        state, response_data = run_stop(state)

        hist = memory.get(req.session_id, [])
        hist.append({"role": "user", "content": state["original_query"]})
        hist.append({"role": "assistant", "content": response_data.get("answer", "")})
        memory[req.session_id] = hist
        langgraph_state.pop(req.session_id, None)

        return response_data
