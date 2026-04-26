"""Whisper POC for multimodal RAG.

POC 目的：
- 驗證 RTX 4060 + faster-whisper large-v3 跑得動
- 量測速度（RTF = transcribe_time / audio_duration，越低越快）
- 確認金融領域專有名詞辨識品質（搭配 initial_prompt）

Usage:
    python scripts/test_whisper.py <audio_path> [--model large-v3] [--lang en]

輸出：stdout 顯示帶時間戳的逐字稿 + 完整檔存到 scripts/out/<stem>.<model>.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel

# initial_prompt 是 Whisper 的「領域偏置」機制：模型在解碼前先看到這段文字，
# 之後遇到發音相近的詞會傾向選擇 prompt 裡出現過的拼法。
# 例如沒給 prompt 時 "CoWoS" 容易被聽成 "co-walls"、"魏哲家" 容易出錯。
# 限制：最多 ~224 tokens，超過會被截斷。
FINANCIAL_PROMPT_EN = (
    "TSMC, NVIDIA, Jensen Huang, C.C. Wei, Morris Chang, "
    "CoWoS, 3nm, 2nm, advanced packaging, gross margin, revenue guidance, "
    "AI accelerator, Blackwell, Hopper, data center, foundry."
)

FINANCIAL_PROMPT_ZH = (
    "台積電 魏哲家 張忠謀 輝達 黃仁勳 "
    "先進製程 3奈米 2奈米 CoWoS 先進封裝 "
    "毛利率 營收 展望 人工智慧 加速器 晶圓代工"
)


def transcribe(audio_path: Path, model_size: str, language: str) -> None:
    prompt = FINANCIAL_PROMPT_ZH if language == "zh" else FINANCIAL_PROMPT_EN

    # compute_type="float16" 是 VRAM / 速度的關鍵權衡：
    #   - float16 → ~3GB VRAM，速度快，品質幾乎無損（4060 8GB 適用）
    #   - int8_float16 → ~2GB，速度更快但品質下降一點
    #   - float32 → ~6GB，速度慢，4060 容易爆
    # 首次跑會從 HuggingFace 下載模型 (~3GB)，之後 cache 在 ~/.cache/huggingface/
    print(f"[load] model={model_size} device=cuda compute_type=float16")
    t0 = time.time()
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    load_time = time.time() - t0
    print(f"[load] done in {load_time:.1f}s\n")

    print(f"[transcribe] file={audio_path.name} lang={language} vad=True")
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        # 顯式指定語言比 auto-detect 快也更穩。
        # 法說會混合中英時，建議分段或用 "en"（Whisper 對英文中混雜中文更寬容，反之較弱）。
        language=language,
        initial_prompt=prompt,
        # VAD (Voice Activity Detection) 過濾無聲段。
        # 法說會前常有等待室靜音，若不開 VAD 模型會在靜音段「幻想」出文字。
        vad_filter=True,
        # beam_size=5 是品質/速度甜蜜點。
        # 1 = greedy（最快但品質差），5 = 平衡，10+ = 邊際效益遞減但更慢。
        beam_size=5,
        # word_timestamps=False：只要 segment 級時間戳。
        # 開 True 會多花 ~30% 時間，給每個字一個 timestamp（影片跳轉用，目前不需要）。
        word_timestamps=False,
    )
    # 注意：segments 是 generator，遍歷時才實際做運算 → t0 必須包住 for 迴圈
    lines: list[str] = []
    for seg in segments:
        line = f"[{seg.start:6.1f} -> {seg.end:6.1f}] {seg.text.strip()}"
        print(line)
        lines.append(line)
    elapsed = time.time() - t0

    # RTF (Real-Time Factor) = 處理時間 / 音訊長度
    #   < 1 → 比即時還快（GPU 通常在這）
    #   = 1 → 剛好等於播放速度
    #   > 1 → 比即時慢（CPU 跑大模型常見）
    audio_dur = info.duration
    rtf = elapsed / audio_dur if audio_dur else 0
    print(
        f"\n[done] audio={audio_dur:.1f}s transcribe={elapsed:.1f}s "
        f"rtf={rtf:.2f}x (lower is better)"
    )
    print(f"[info] detected_lang={info.language} prob={info.language_probability:.2f}")

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{audio_path.stem}.{model_size}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="path to audio file")
    parser.add_argument(
        "--model",
        default="large-v3",
        # 模型尺寸 / VRAM / 中英品質：
        #   tiny   ~75MB  普通對話可，金融術語易錯
        #   base   ~145MB 同上
        #   small  ~480MB 英文堪用，中文吃力
        #   medium ~1.5GB 英文不錯，中文中等
        #   large-v3       ~3GB  最佳品質，4060 推薦
        #   distil-large-v3 ~1.5GB large-v3 的蒸餾版，速度 6x 但長音訊易漏字
        choices=["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"],
    )
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"audio file not found: {args.audio}")

    transcribe(args.audio, args.model, args.lang)


if __name__ == "__main__":
    main()
