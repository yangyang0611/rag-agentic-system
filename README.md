# RAG Agentic System

A retrieval-augmented generation system that lets users ingest documents (web pages, PDFs) into a vector database (ChromaDB), perform semantic search over them, and get LLM-generated answers grounded in retrieved context. The system exposes two interfaces from a single backend: a REST API with a web UI for end users, and an MCP (Model Context Protocol) server for AI agents like Claude.

## Architecture

```
  Web UI                       AI Agent (Claude, Cursor, etc.)
    │                                │
    │ REST API                       │ MCP (stdio)
    ▼                                ▼
┌────────────────────────────────────────────────────┐
│                  FastAPI Server                     │
│                                                    │
│  ┌──────────────┐    ┌───────────────────────────┐ │
│  │ REST Routes  │    │ MCP Tool Server            │ │
│  │              │    │                            │ │
│  │ /ingest      │───▶│ ingest_url()              │ │
│  │ /upload      │    │ query_docs()              │ │
│  │ /query       │───▶│ web_search()              │ │
│  │ /ask-db      │    └───────────────────────────┘ │
│  │ /search      │                                  │
│  │ /agent       │──▶ Agent Loop (hand-rolled)      │
│  │ /agent-v2    │──▶ Agent Loop (LangGraph)        │
│  └──────────────┘                                  │
└──────┬──────────────────┬──────────────────┬───────┘
       ▼                  ▼                  ▼
    Tavily API        ChromaDB           OpenRouter
    (web scraping     (vector store,     (LLM inference via
     + web search)    HNSW + cosine)     OpenAI-compatible API)
```

Both interfaces share the same ingestion pipeline, vector store, and tool implementations — no logic is duplicated.


## Key Engineering Decisions

### Hybrid Retrieval with Reflection

The agent queries the local vector DB first. If results are insufficient, it falls back to live web search. After each tool call, a **reflection step** has the LLM evaluate:
- Is the gathered information sufficient?
- Are there contradictions across sources?
- Should the next search use a different angle or keywords?

This turns a simple retrieve-and-answer pipeline into a multi-step reasoning loop (up to 5 rounds).

### Two Agent Implementations

| | Hand-rolled (`/agent`) | LangGraph (`/agent-v2`) |
|---|---|---|
| Control flow | Manual loop with `if/else` | State graph with conditional edges |
| State | Python dict, managed in-memory | `StateGraph` with typed nodes |
| Resume | Dict lookup by session ID | Graph re-invocation from checkpoint |

Both implement the same behavior: **query rewrite → tool selection → execution → reflection → human checkpoint**. Building both let me compare the trade-offs between explicit control flow and graph-based orchestration.

### Human-in-the-Loop Agent Control

The agent pauses after each tool execution round and returns intermediate results (tool calls, retrieved sources, reflection analysis) to the frontend. The user decides whether to **continue** (let the agent search more) or **stop** (force a final answer from what's been gathered). Session state is preserved server-side so the agent can resume exactly where it paused.

### Document Ingestion Pipeline

```
URL → Tavily extract → chunk (500 words, 50-word overlap) → sentence-transformers encode → ChromaDB upsert
PDF → PyMuPDF extract → same chunking/embedding pipeline
```

- **Overlapping chunks** prevent loss of context at chunk boundaries
- **Content-addressed IDs** (MD5 of source + chunk index) enable idempotent upserts — re-ingesting the same document updates rather than duplicates
- Embedding model: `BAAI/bge-large-en-v1.5` (top-ranked on MTEB at time of selection)

### Query Rewriting

Before searching, the agent rewrites the user's raw question into a search-optimized query. For multi-turn conversations, it resolves pronouns and references using session history (e.g., "tell me more about it" → "tell me more about [specific topic from previous turn]").

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Server | FastAPI, Uvicorn |
| Vector DB | ChromaDB (HNSW index, cosine similarity, local persistence) |
| Embeddings | sentence-transformers (`BAAI/bge-large-en-v1.5`) |
| LLM | OpenRouter API via OpenAI SDK (swappable — any OpenAI-compatible endpoint) |
| Web data | Tavily API (page extraction + web search) |
| PDF parsing | PyMuPDF |
| Agent framework | LangGraph (state graph with conditional edges) |
| Agent protocol | MCP (Model Context Protocol) |
| Frontend | Vanilla JS, marked.js |

## Project Structure

```
main.py                # FastAPI + MCP server init, LLM client config
routes.py              # REST endpoints, hand-rolled agent loop, session memory
tools.py               # MCP tool definitions + agent tool dispatch table
ingester.py            # Document pipeline: fetch → chunk → embed → store
langgraph_agent.py     # LangGraph agent: state graph with reflection nodes
langchain_ingester.py  # Alternative ingestion via LangChain loaders
index.html             # Web UI (single page)
static/app.js          # Frontend: multi-mode query, agent controls
```

## Getting Started

```bash
uv sync
# Set OPENROUTER_API_KEY and TAVILY_API_KEY in .env
uv run python main.py      # Web UI at http://localhost:8000
```

To use as an MCP server (e.g., in Cline), add to your MCP config:

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "/absolute/path/to/rag-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/rag-mcp/main.py"]
    }
  }
}
```

Use absolute paths — MCP clients spawn the server as a subprocess, so relative paths won't resolve correctly.