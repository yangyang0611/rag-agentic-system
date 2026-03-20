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


class IngestRequest(BaseModel):
    url: str


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

    @app.post("/agent")
    def api_agent(req: QueryRequest):
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant with access to tools. ALWAYS search the local database (query_docs) first. Only use web_search if the database returns no relevant results. You may call multiple tools if needed. Be direct and concise in your final answer."},
            {"role": "user", "content": req.query},
        ]
        all_sources: list[dict] = []
        tools_used: list[str] = []

        print(f"\n{'='*60}")
        print(f"[Agent] New query: {req.query}")
        print(f"{'='*60}")

        for round_num in range(5):
            print(f"[Agent] Round {round_num + 1}: Asking LLM...")
            try:
                response = openai_client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages,
                    tools=AGENT_TOOLS,
                )
            except Exception as e:
                print(f"[Agent] Round {round_num + 1}: Tool call failed ({e}), retrying without tools")
                response = openai_client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages,
                )
                msg = response.choices[0].message
                break
            msg = response.choices[0].message

            if not msg.tool_calls:
                print(f"[Agent] Round {round_num + 1}: No tool calls, generating answer")
                break

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

        seen = set()
        unique_sources = []
        for s in all_sources:
            key = s.get("content", "")
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

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
