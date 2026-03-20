# RAG MCP Architecture

## System Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Users                                   │
├────────────────────────────┬────────────────────────────────────┤
│      Web UI (Browser)      │      AI Agent (Claude etc.)       │
│      index.html            │      via MCP Protocol             │
└─────────────┬──────────────┴──────────────────┬────────────────┘
              │ HTTP REST                       │ MCP (stdio)
              ▼                                 ▼
┌─────────────────────────────┐   ┌──────────────────────────────┐
│   FastAPI REST Endpoints    │   │      MCP Tools               │
│                             │   │                              │
│  POST /ingest ─────────────────→│  ingest_url()                │
│  POST /query  ─────────────────→│  query_docs()                │
│  POST /search              │    │  web_search()                │
│  POST /answer              │    │                              │
│  POST /upload              │    └──────────────────────────────┘
│  POST /agent               │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Services                           │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Tavily API     │   ChromaDB       │   Gemini API              │
│   (External)     │   (Local DB)     │   (LLM Inference)         │
│                  │                  │                           │
│  - extract()     │  - upsert()      │  - chat.completions       │
│    Fetch webpage │    Store vectors  │    .create()              │
│  - search()      │  - query()       │    Generate answers       │
│    Web search    │    Semantic search│    (gemini-2.5-flash)     │
└──────────────────┴──────────────────┴───────────────────────────┘
```

## Call Chains

```
Web UI Button              REST Endpoint        Backend Calls
─────────────              ─────────────        ────────────────────

[Ingest URL]  ──→ POST /ingest  ──→ ingest_url()
                                        ├── Tavily extract (fetch page)
                                        ├── SentenceTransformer (embed)
                                        └── ChromaDB upsert (store)

[Upload PDF]  ──→ POST /upload  ──→ ingest_file()
                                        ├── PyMuPDF (read PDF)
                                        ├── SentenceTransformer (embed)
                                        └── ChromaDB upsert (store)

[Raw Chunks]  ──→ POST /query   ──→ query_docs()
                                        ├── SentenceTransformer (embed query)
                                        └── ChromaDB query (search DB)

[AI Answer    ──→ POST /answer  ──→ query_docs()  → ChromaDB
 (DB)]                              └── Gemini LLM (generate answer)

[AI Answer    ──→ POST /search  ──→ Tavily search (web search)
 (Web)]                             └── Gemini LLM (generate answer)

[Agent]       ──→ POST /agent   ──→ Agent Loop (see below)


AI Agent (Claude)                MCP Tool          Backend Calls
─────────────────                ────────           ────────────────

Claude decides to ingest  ──→ ingest_url()  ──→ Tavily extract + ChromaDB
Claude decides to query   ──→ query_docs()  ──→ ChromaDB query
Claude decides to search  ──→ web_search()  ──→ Tavily search
```

## Agent Loop Detail

```
User clicks [Agent] → POST /agent → Agent Loop starts

┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Agent Loop (max 5 rounds)                  │  │
│  │                                                        │  │
│  │  Round 1:                                              │  │
│  │  Server → Gemini:  "What tool should I use?"           │  │
│  │  Gemini → Server:  "Use query_docs(conan)"             │  │
│  │  Server → ChromaDB: query → 3 results                  │  │
│  │                                                        │  │
│  │  Round 2:                                              │  │
│  │  Server → Gemini:  "Here are DB results. Enough?"      │  │
│  │  Gemini → Server:  "Yes, here is my answer"            │  │
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Return { answer, sources, source_label: "DB" }              │
└──────────────────────────────────────────────────────────────┘


If DB has no relevant data, more rounds:

  Round 1:
  Server → Gemini:  "What tool?"
  Gemini → Server:  "Use query_docs"
  Server → ChromaDB: query → 0 results / irrelevant

  Round 2:
  Server → Gemini:  "DB had nothing. Now what?"
  Gemini → Server:  "Use web_search"
  Server → Tavily:   search → 3 results

  Round 3:
  Server → Gemini:  "Here are web results. Enough?"
  Gemini → Server:  "Yes, here is my answer"


Who does what:

┌─────────────┬──────────────────────────────┐
│   Role       │  Responsibility              │
├─────────────┼──────────────────────────────┤
│ Browser UI   │  Send question, show results │
│ FastAPI      │  Run Agent Loop              │
│ Server       │  Execute tools (ChromaDB/    │
│ (your code)  │  Tavily), feed results to LLM│
│ Gemini API   │  Decide which tool to use    │
│ (LLM)       │  Generate final answer       │
│              │  Brain only, no searching    │
│ ChromaDB     │  Local vector search         │
│ Tavily API   │  Web search / page extract   │
└─────────────┴──────────────────────────────┘
```

## Key Point

Tavily is the single external data source engine, handling both "fetch webpage content" (`extract`) and "search the internet" (`search`). The difference is the entry point: Web UI calls via REST endpoints, AI Agents call via MCP tools.
