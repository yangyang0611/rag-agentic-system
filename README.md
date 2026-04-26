# Stockbot — Multimodal RAG for Equity Research

A retrieval-augmented research workspace that ingests **earnings call audio, financial PDFs, web pages, and tables** into a single hybrid retrieval layer, then lets an agent answer equity-research questions across all sources at once. Built on top of a generic RAG / agent foundation so the same backend powers both a browser UI for analysts and an MCP (Model Context Protocol) server for AI agents like Cline, Claude, or Cursor.

> The "Stockbot" framing is the demo use case — the underlying engine is a general-purpose multimodal RAG with two retrieval paths (vector + structure-aware) and two agent runtimes (hand-rolled + LangGraph).

## Architecture

```
  Web UI                       AI Agent (Claude, Cursor, etc.)
    │                                │
    │ REST API                       │ MCP (stdio)
    ▼                                ▼
┌────────────────────────────────────────────────────────┐
│                  FastAPI Server                         │
│                                                         │
│  ┌──────────────────┐    ┌───────────────────────────┐ │
│  │ REST Routes      │    │ MCP Tool Server            │ │
│  │                  │    │                            │ │
│  │ /ingest          │───▶│ ingest_url()              │ │
│  │ /upload (PDF)    │    │ query_docs()              │ │
│  │ /ingest-audio    │    │ web_search()              │ │
│  │ /query           │───▶│ query_structured()        │ │
│  │ /ask-db          │    └───────────────────────────┘ │
│  │ /search          │                                  │
│  │ /query-structured│                                  │
│  │ /agent           │──▶ Agent Loop (hand-rolled)      │
│  │ /agent-v2        │──▶ Agent Loop (LangGraph)        │
│  └──────────────────┘                                  │
└──┬──────────┬───────────┬──────────┬───────────┬───────┘
   ▼          ▼           ▼          ▼           ▼
 Tavily   ChromaDB    Struct      Whisper     OpenRouter
 (web)    (vectors)   Index       (audio)     (LLM + VLM)
```

Both interfaces share the same ingestion pipeline, vector store, and tool implementations — no logic is duplicated.

## What's in for Equity Research

| Source | Pipeline | Result in retrieval |
|---|---|---|
| Earnings call MP3 / WAV | `faster-whisper large-v3` (GPU, float16, VAD, financial-domain prompt) | Time-stamped transcript chunks (`[AUDIO 743s-910s] …`) |
| Financial PDF | PyMuPDF text + table detector + Gemini-2.5-Flash visual summarizer | Text chunks + extracted tables + chart/diagram descriptions |
| News / IR web pages | Tavily extract → 500-word overlapping chunks | Standard text chunks |
| Heading-rich docs (10-K, slide decks) | Structure-aware parser → JSON heading tree | Whole sections, routed by LLM via TOC |

A single query goes through the same agent and surfaces hits across all of these — analysts can ask "*What did C.C. Wei say about CoWoS demand and how does that line up with the Q4 capex chart?*" and the agent will pull the audio segment, the table, and the visual summary together.

## Key Engineering Decisions

### Audio Ingestion via Whisper

Earnings calls are speech, not text — but they're the highest-signal source in equity research. The pipeline keeps them first-class in the retrieval layer:

- **`faster-whisper large-v3` on GPU (float16)** — RTF ≈ 0.26 on RTX 4060, so a 60-min call transcribes in ~15 min
- **Domain-biased decoding** — `initial_prompt` seeds Whisper with finance vocab (CoWoS, agentic AI, C.C. Wei, etc.) so it doesn't mishear ticker-specific terms
- **VAD filter** — drops silent stretches (typical earnings-call lobby music) so the model doesn't hallucinate over them
- **Time-aware chunking** — Whisper segments are packed to ~500 words while preserving `start_time` / `end_time` per chunk, so retrieved hits know exactly where in the audio they came from
- **Idempotent upserts** — content-addressed IDs (`md5(source + chunk_idx)`) mean re-ingesting the same MP3 overwrites instead of duplicating

### PDF Multimodal Extraction

Financial PDFs are not just text — three streams get extracted in one pass:

- **Body text** via PyMuPDF
- **Tables** via PyMuPDF's table detector → `cell | cell | cell` rows fed to retrieval as `[TABLE]` blocks
- **Charts / diagrams** via a Gemini-2.5-Flash VLM call that produces JSON summaries (`type`, `title`, `summary`, `key_points`)

All three end up in the same ChromaDB collection, so a query like "*revenue by platform Q1*" can return either the text paragraph, the underlying table, or the chart describing it — whichever ranks higher.

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

### Structure-Aware Retrieval (Vectorless)

A second retrieval path that doesn't use vector embeddings at all. Instead of chunking text blindly, it parses documents into a **hierarchical tree (DocNode)** based on headings:

```
Document
├── Experience          (heading, level 1)
│   ├── NVIDIA          (heading, level 2)
│   │   └── paragraph: "Developed test automation..."
│   └── Previous Co.    (heading, level 2)
│       └── paragraph: "Built data processing..."
└── Education           (heading, level 1)
    └── paragraph: "M.S. in ..."
```

**Why this matters for filings:** vector search on flat chunks can mix content from different sections of a 10-K (e.g., a chunk containing text from both "Risk Factors" and "MD&A"). Structure-aware retrieval preserves document boundaries — asking "*what risks did they flag around China exposure?*" returns only content under that specific heading.

**How it works:**
1. **Parsing** — PDF: PyMuPDF `get_text("dict")` detects headings by font size relative to body text. HTML: BeautifulSoup parses `<h1>`-`<h6>` tags.
2. **Indexing** — The tree is stored as a JSON file (`./struct_index/`), no embedding computation needed.
3. **Query routing** — The LLM receives only the document's **table of contents** (heading titles, ~1KB) and selects the most relevant sections. This costs minimal tokens while providing accurate routing.
4. **Retrieval** — The selected heading node's full content is recursively collected and returned.

| | Vector-based (`query_docs`) | Structure-aware (`query_structured`) |
|---|---|---|
| Index | Embedding vectors in ChromaDB | Heading tree as JSON |
| Search | Cosine similarity | LLM routes via TOC |
| Returns | Arbitrary text chunks | Complete sections with exact boundaries |
| Best for | Semantic similarity across unstructured text | Precise section lookup in filings, decks |

### Query Rewriting

Before searching, the agent rewrites the user's raw question into a search-optimized query. For multi-turn conversations, it resolves pronouns and references using session history (e.g., "*tell me more about it*" → "*tell me more about CoWoS capacity expansion*").

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Server | FastAPI, Uvicorn |
| Vector DB | ChromaDB (HNSW index, cosine similarity, local persistence) |
| Embeddings | sentence-transformers (`BAAI/bge-large-en-v1.5`) |
| LLM | OpenRouter API via OpenAI SDK (swappable — any OpenAI-compatible endpoint) |
| VLM (PDF visuals) | Google Gemini 2.5 Flash via OpenRouter |
| Speech-to-text | `faster-whisper` (`large-v3`, GPU/float16) |
| Audio decode | ffmpeg (system dependency) |
| Web data | Tavily API (page extraction + web search) |
| PDF parsing | PyMuPDF (text + tables + page rendering for VLM) |
| HTML parsing | BeautifulSoup (structural heading extraction) |
| Agent framework | LangGraph (state graph with conditional edges) |
| Agent protocol | MCP (Model Context Protocol) |
| Frontend | Vanilla JS, marked.js |

## Project Structure

```
main.py                # FastAPI + MCP server init, LLM client config
routes.py              # REST endpoints, hand-rolled agent loop, session memory
tools.py               # MCP tool definitions + agent tool dispatch table
ingester.py            # Vector pipeline: fetch → chunk → embed → store
                       # + PDF multimodal: text + tables + VLM visual summaries
audio_ingester.py      # Whisper transcription → time-aware chunks → ChromaDB
structured_ingester.py # Structural pipeline: parse headings → DocNode tree → JSON index
langgraph_agent.py     # LangGraph agent: state graph with reflection nodes
langchain_ingester.py  # Alternative ingestion via LangChain loaders
index.html             # Web UI (single page)
static/app.js          # Frontend: multi-mode query, agent controls, audio upload
scripts/               # POC scripts (Whisper benchmark, ingestion smoke tests)
```

## Getting Started

```bash
# System deps (Whisper needs ffmpeg to decode MP3 / WAV / M4A / WebM)
sudo apt install ffmpeg

# Python deps
uv sync

# Set OPENROUTER_API_KEY and TAVILY_API_KEY in .env
uv run python main.py      # Web UI at http://localhost:8000
```

First run downloads ~3 GB of models (BGE embeddings + Whisper large-v3) into `~/.cache/huggingface/`. Subsequent starts are fast.

### Equity-research demo flow

1. Open <http://localhost:8000>
2. **Ingest** an earnings call MP3 (e.g., TSMC Q1 2026, English) → wait for transcription
3. **Upload** the matching earnings PDF — text + tables + chart summaries all index in one shot
4. Run **Agent v2** with a mixed-source question: *"Based on the call transcript and the revenue-by-platform table, how dependent is TSMC on HPC for Q1?"*
5. The agent returns timestamps for the audio source, page numbers for the PDF chunks, and the reflection trace it used to combine them.

### MCP server usage

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

## Future Work

- **Image / chart upload** — let users drop K-line screenshots or chart images directly (currently only PDF-embedded visuals are described)
- **Real-time financial data tools** — `yfinance` / FinMind tools so the agent can pair RAG context with current price / PE / technical indicators
- **Speaker diarization** — `pyannote` to label CEO vs analyst turns in earnings transcripts
- **Video** — `yt-dlp` + Whisper to ingest financial-news videos with frame-level chart description
- **Scheduled ingestion** — daily watcher for IR pages / SEC filings of subscribed tickers
