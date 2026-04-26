"""End-to-end test for audio_ingester.

驗證：
1. ingest_audio 跑得通
2. ChromaDB 多了 N 個 chunks，content_type=audio，metadata 帶 timestamp
3. 同一檔案 ingest 兩次 collection 不會增量（idempotent upsert）
4. 用「AI demand / capex」這類關鍵字能檢索到音訊內容
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# 必須在 import ingester 前載入 .env，因為 ingester.py 在 module level 讀 TAVILY_API_KEY
load_dotenv()

from audio_ingester import ingest_audio  # noqa: E402
from ingester import collection, model as embedding_model  # noqa: E402


def count_audio_chunks() -> int:
    return len(collection.get(where={"content_type": "audio"})["ids"])


def main() -> None:
    audio = Path("data/audio/tsmc_q1_content.mp3")
    if not audio.exists():
        raise SystemExit(f"audio file not found: {audio}")

    print(f"\n--- Step 1: baseline audio chunk count ---")
    before = count_audio_chunks()
    print(f"audio chunks in DB: {before}")

    print(f"\n--- Step 2: first ingest ---")
    result1 = ingest_audio(audio, source="tsmc_q1_2026_earnings", language="en")
    print(f"result: {result1}")
    after1 = count_audio_chunks()
    print(f"audio chunks in DB: {after1} (delta={after1 - before})")

    print(f"\n--- Step 3: re-ingest same file (must be idempotent) ---")
    result2 = ingest_audio(audio, source="tsmc_q1_2026_earnings", language="en")
    print(f"result: {result2}")
    after2 = count_audio_chunks()
    print(f"audio chunks in DB: {after2} (delta vs first ingest={after2 - after1})")
    assert after2 == after1, "idempotency broken — chunk count grew on re-ingest"
    print("✅ idempotent")

    print(f"\n--- Step 4: retrieval check ---")
    queries = ["AI demand outlook", "2026 capital expenditure", "leading-edge process"]
    query_embeddings = embedding_model.encode(queries).tolist()
    for query, emb in zip(queries, query_embeddings):
        results = collection.query(
            query_embeddings=[emb],
            n_results=2,
            where={"content_type": "audio"},
        )
        print(f"\nQ: {query!r}")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            preview = doc[:160].replace("\n", " ")
            print(f"  [{meta['start_time']:.0f}s-{meta['end_time']:.0f}s] {preview}...")


if __name__ == "__main__":
    main()
