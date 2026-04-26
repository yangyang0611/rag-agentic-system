"""Audio ingestion: MP3 → Whisper transcript → time-aware chunks → ChromaDB.

設計重點：
- 復用 ingester.py 的 BGE embedding model 與 ChromaDB collection（同一向量庫）
- 每個 chunk 保留 start_time / end_time（將來可做影片跳轉）
- Whisper model 採 lazy-load：避免 import 時就吃 3GB VRAM
- 同檔案重複 ingest 會 overwrite（content-addressed ID）
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from ingester import collection, model as embedding_model

FINANCIAL_PROMPT_EN = (
    "TSMC, NVIDIA, Jensen Huang, C.C. Wei, Morris Chang, "
    "CoWoS, 3nm, 2nm, advanced packaging, gross margin, revenue guidance, "
    "AI accelerator, agentic AI, Blackwell, Hopper, data center, foundry."
)
FINANCIAL_PROMPT_ZH = (
    "台積電 魏哲家 張忠謀 輝達 黃仁勳 "
    "先進製程 3奈米 2奈米 CoWoS 先進封裝 "
    "毛利率 營收 展望 人工智慧 加速器 晶圓代工"
)

# Whisper 模型在第一次使用時才載入（~3GB VRAM）。
# 模組層級 singleton 讓多次 ingest 共用同一份模型。
_whisper_model: WhisperModel | None = None


def _get_whisper(model_size: str = "large-v3") -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return _whisper_model


@dataclass
class AudioChunk:
    text: str
    start: float
    end: float


def transcribe_audio(audio_path: Path, language: str = "en") -> list[AudioChunk]:
    """Run Whisper, return raw segments with timing preserved."""
    prompt = FINANCIAL_PROMPT_ZH if language == "zh" else FINANCIAL_PROMPT_EN
    whisper = _get_whisper()
    segments, _info = whisper.transcribe(
        str(audio_path),
        language=language,
        initial_prompt=prompt,
        vad_filter=True,
        beam_size=5,
    )
    return [AudioChunk(text=s.text.strip(), start=s.start, end=s.end) for s in segments]


def group_segments_into_chunks(
    segments: list[AudioChunk],
    target_words: int = 500,
    overlap_segments: int = 1,
) -> list[AudioChunk]:
    """Pack consecutive Whisper segments into ~500-word chunks, keeping timing.

    為什麼用 segment 級重疊（而非詞級）：Whisper 已天然在語意停頓切，
    從上一 chunk 借 1 個尾段當下一 chunk 開頭，足以避免邊界斷句問題。
    """
    if not segments:
        return []

    chunks: list[AudioChunk] = []
    i = 0
    while i < len(segments):
        word_count = 0
        j = i
        while j < len(segments) and word_count < target_words:
            word_count += len(segments[j].text.split())
            j += 1
        merged_text = " ".join(s.text for s in segments[i:j])
        chunks.append(AudioChunk(
            text=merged_text,
            start=segments[i].start,
            end=segments[j - 1].end,
        ))
        # 從 j 往回借 overlap_segments 個段落讓相鄰 chunk 有重疊；首段不需借
        next_i = j - overlap_segments if j > overlap_segments else j
        if next_i <= i:  # 防呆：避免 target_words 過小時死循環
            next_i = i + 1
        i = next_i
    return chunks


def ingest_audio(
    audio_path: str | Path,
    source: str = "",
    language: str = "en",
) -> dict:
    """Full pipeline: audio file → transcript → chunked + embedded into ChromaDB."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    source = source or audio_path.name

    print(f"[audio_ingester] transcribing {audio_path.name} (lang={language})...")
    segments = transcribe_audio(audio_path, language=language)
    print(f"[audio_ingester] got {len(segments)} segments, packing chunks...")

    chunks = group_segments_into_chunks(segments)
    if not chunks:
        return {"source": source, "chunks_stored": 0, "audio_duration": 0.0}

    documents = [
        f"[AUDIO {chunk.start:.0f}s-{chunk.end:.0f}s] {chunk.text}"
        for chunk in chunks
    ]
    embeddings = embedding_model.encode(documents).tolist()
    # 與 ingester.py 用 "_audio_" 區隔命名空間：
    # 同個 source 之後若再 ingest PDF/網頁不會撞 ID
    ids = [
        hashlib.md5(f"{source}_audio_{i}".encode()).hexdigest()
        for i in range(len(chunks))
    ]
    metadatas = [
        {
            "url": source,
            "chunk_index": i,
            "content_type": "audio",
            "start_time": chunk.start,
            "end_time": chunk.end,
            "language": language,
        }
        for i, chunk in enumerate(chunks)
    ]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    full_transcript = " ".join(s.text for s in segments)
    preview = full_transcript[:500] + ("..." if len(full_transcript) > 500 else "")
    print(f"[audio_ingester] stored {len(chunks)} chunks for '{source}'")
    return {
        "source": source,
        "chunks_stored": len(chunks),
        "audio_duration": segments[-1].end if segments else 0.0,
        "language": language,
        "preview": preview,
    }
