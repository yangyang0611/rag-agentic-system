import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from tools import mcp, ingest_url, query_docs
from routes import register_routes

openai_client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

register_routes(app, openai_client)


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
