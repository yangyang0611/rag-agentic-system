import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from tools import mcp
from routes import register_routes

# ── LLM Client ──
# 使用 OpenAI SDK 作為通用介面，透過 base_url 切換不同的 LLM backend
# 只要 server 實作了 OpenAI 相容的 API（/v1/chat/completions），就能直接接
# 範例：
#   OpenRouter:    base_url="https://openrouter.ai/api/v1"
#   Gemini:        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
#   公司內部 LLM:  base_url="http://internal-llm.company.com/v1"
openai_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

register_routes(app, openai_client)


if __name__ == "__main__":
    mcp.run()
