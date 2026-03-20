# Fine-Tune LLM for Tool Calling — Qualcomm 場景

## 目標

讓公司內部的 LLM（例如在 Qualcomm GPU server 上跑的 Llama）學會 agent 能力：
- 判斷什麼時候該用工具、什麼時候直接回答
- 選擇正確的工具
- 組出正確的 JSON 參數格式

---

## 訓練資料格式

每一筆訓練資料 = 一段完整的對話（OpenAI Chat Format）。

### 情境 1：需要查內部資料庫

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant with access to tools: query_docs, web_search."},
    {"role": "user", "content": "Snapdragon 8 Gen 3 的 power consumption spec 是多少？"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "query_docs", "arguments": "{\"query\": \"Snapdragon 8 Gen 3 power consumption specification\"}"}}
    ]},
    {"role": "tool", "content": "[{\"content\": \"Snapdragon 8 Gen 3 TDP: 12.5W typical, 15W peak...\", \"url\": \"internal://specs/sd8gen3\"}]"},
    {"role": "assistant", "content": "根據內部規格文件，Snapdragon 8 Gen 3 的功耗為 typical 12.5W，peak 15W。"}
  ]
}
```

### 情境 2：不需要工具，直接回答

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant with access to tools: query_docs, web_search."},
    {"role": "user", "content": "什麼是 TDP？"},
    {"role": "assistant", "content": "TDP (Thermal Design Power) 是指晶片在典型工作負載下的最大散熱功率，用來設計散熱方案。"}
  ]
}
```

### 情境 3：需要搜尋外部資訊

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant with access to tools: query_docs, web_search."},
    {"role": "user", "content": "聯發科最新的天璣晶片跟我們的 Snapdragon 比較如何？"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "query_docs", "arguments": "{\"query\": \"Snapdragon latest benchmark performance\"}"}},
      {"function": {"name": "web_search", "arguments": "{\"query\": \"MediaTek Dimensity latest chip benchmark 2024\"}"}}
    ]},
    {"role": "tool", "content": "[{\"content\": \"Snapdragon 8 Gen 3 AnTuTu: 2,050,000...\"}]"},
    {"role": "tool", "content": "[{\"content\": \"Dimensity 9300 AnTuTu: 1,980,000...\"}]"},
    {"role": "assistant", "content": "根據內部資料，Snapdragon 8 Gen 3 AnTuTu 跑分約 205 萬，而公開資料顯示聯發科天璣 9300 約 198 萬，我們在性能上略有領先。"}
  ]
}
```

### 情境 4：先查 DB 沒結果，再搜 Web（多輪工具呼叫）

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant with access to tools: query_docs, web_search. ALWAYS search query_docs first."},
    {"role": "user", "content": "TSMC 3nm 良率目前多少？"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "query_docs", "arguments": "{\"query\": \"TSMC 3nm yield rate\"}"}}
    ]},
    {"role": "tool", "content": "[]"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "web_search", "arguments": "{\"query\": \"TSMC 3nm yield rate 2024\"}"}}
    ]},
    {"role": "tool", "content": "[{\"content\": \"TSMC N3 process yield has reached approximately 80%...\"}]"},
    {"role": "assistant", "content": "內部資料庫沒有相關資訊。根據公開報導，TSMC 3nm 製程良率目前約 80%。"}
  ]
}
```

### 情境 5：生成測試程式（Qualcomm 特定場景）

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant that helps generate semiconductor test programs. Tools: query_docs, generate_test."},
    {"role": "user", "content": "幫我生成 Snapdragon 8 Gen 3 的 DDR5 memory test"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "query_docs", "arguments": "{\"query\": \"Snapdragon 8 Gen 3 DDR5 test template specification\"}"}}
    ]},
    {"role": "tool", "content": "[{\"content\": \"DDR5 test spec: freq 4800-8400 MHz, voltage 1.1V...\"}]"},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "generate_test", "arguments": "{\"chip\": \"SD8Gen3\", \"interface\": \"DDR5\", \"freq_range\": \"4800-8400\", \"voltage\": \"1.1V\"}"}}
    ]},
    {"role": "tool", "content": "{\"test_file\": \"sd8gen3_ddr5_test.py\", \"status\": \"generated\"}"},
    {"role": "assistant", "content": "已根據內部規格生成 DDR5 測試程式 sd8gen3_ddr5_test.py，涵蓋頻率 4800-8400 MHz、電壓 1.1V 的測試範圍。"}
  ]
}
```

---

## 資料需求

| 項目 | 建議 |
|---|---|
| 總資料量 | 至少 1,000 ~ 10,000 筆對話 |
| 情境分佈 | 40% 用工具、30% 不用工具、20% 多工具、10% 多輪 |
| 語言 | 看實際使用，中英混合就準備中英混合資料 |
| 工具覆蓋 | 每個工具至少 200+ 筆範例 |
| 邊界情況 | 工具回傳空結果、錯誤處理、拒絕不合理請求 |

---

## 資料來源（Qualcomm 場景）

1. **內部文件** — chip spec、test plan、design doc → 做成「用戶問 spec 問題 → 查 DB → 回答」的對話
2. **歷史 ticket** — JIRA/Confluence 上的 Q&A → 轉換成對話格式
3. **現有測試程式** — 把人工寫測試的過程記錄成「查規格 → 生成測試」的對話
4. **人工標註** — 工程師手動寫幾百筆高品質範例，再用 GPT-4 擴增
5. **合成資料** — 用大模型（GPT-4/Claude）根據模板批量生成訓練資料，人工審核

---

## Fine-tune 流程

```
1. 準備資料（上面的 JSON 格式）
     ↓
2. 選基礎模型（Llama 3.1 8B / 70B）
     ↓
3. Fine-tune（用 LoRA / QLoRA 省 GPU 記憶體）
     ↓
4. 部署（用 vLLM 起 OpenAI 相容 API server）
     ↓
5. 接上你的系統（改 base_url 就好）
```

### Fine-tune 指令範例（用 Hugging Face TRL）

```bash
# LoRA fine-tune Llama 3.1 8B
python -m trl sft \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name ./training_data.jsonl \
  --output_dir ./finetuned-model \
  --use_peft \
  --lora_r 16 \
  --lora_alpha 32 \
  --per_device_train_batch_size 4 \
  --num_train_epochs 3
```

### 部署

```bash
# 用 vLLM 部署 fine-tuned 模型
vllm serve ./finetuned-model --port 8080

# 你的 code 只要改 base_url
openai_client = OpenAI(
    api_key="not-needed",
    base_url="http://gpu-server.qualcomm.com:8080/v1",
)
```

---

## LLM 直接寫測試程式（進階）

除了用 function + 模板，也可以訓練 LLM 直接輸出測試程式碼。

### 兩種方式比較

| 方式 | 適合場景 | 訓練資料 |
|---|---|---|
| Function + 模板 | 標準化測試（格式固定、換參數） | tool calling 對話資料 |
| LLM 直接寫 code | 客製化測試（每次邏輯不同） | 「需求 → 程式碼」配對資料 |

### LLM 直接寫 code 的訓練資料

```json
{
  "messages": [
    {"role": "user", "content": "生成 SD8Gen3 的 DDR5 memory test，頻率 4800-8400MHz"},
    {"role": "assistant", "content": "```python\nimport test_framework\n\ndef test_ddr5_sd8gen3():\n    dut = Device('SD8Gen3')\n    for freq in range(4800, 8401, 400):\n        dut.set_ddr5_freq(freq)\n        assert dut.memory_read_write_check()\n```"}
  ]
}
```

### 建議策略

先用 function + 模板快速上線，同時收集「工程師的需求描述 → 實際寫出的測試程式」資料，
累積夠多之後再訓練 LLM 直接寫 code。

---

## LLM 訓練的三個階段（以 Claude/GPT 為例）

了解大模型怎麼學會寫程式，有助於理解 fine-tune 在做什麼。

### 階段 1：Pre-training（大量閱讀）— Unsupervised

- 讀大量網路文本 + GitHub code + 書籍 + 論文
- **沒有人標註**「這是好程式」或「這是壞程式」
- 模型只是在學「下一個 token 最可能是什麼」
- 讀完後已經「會寫 code」，但不太會聽指令
- 成本極高：需要數千張 GPU 跑數週

### 階段 2：Supervised Fine-tuning（學會聽指令）— Supervised

- 人工標註幾萬筆高品質的 (需求, 回答) 配對資料
- 讓模型從「自動補全文字」變成「聽懂你要什麼，給出回答」
- **這就是公司 fine-tune 在做的事**

```json
{"instruction": "寫一個 DDR5 測試", "response": "```python\ndef test_ddr5():\n    ...\n```"}
```

### 階段 3：RLHF（學會品質判斷）— Reinforcement Learning

- 同一個問題讓模型生成多個回答
- 人類標註哪個回答比較好
- 用 reinforcement learning 讓模型偏好「更好的回答」

### Qualcomm 實務上只需要做階段 2

```
拿已經訓練好的 Llama（已完成階段 1 + 3）
  ↓
用公司的「需求 → 測試程式」資料做 fine-tune（階段 2）
  ↓
模型就學會寫 Qualcomm 格式的測試程式
```

不需要從頭訓練（階段 1 成本太高），也不需要做 RLHF（階段 3），
直接站在開源模型的肩膀上，用公司資料教它特定領域知識即可。

---

## 評估指標

| 指標 | 說明 | 目標 |
|---|---|---|
| Tool Selection Accuracy | 模型選對工具的比例 | > 90% |
| Argument Accuracy | 參數格式正確的比例 | > 85% |
| No-Tool Accuracy | 不該用工具時正確直接回答的比例 | > 95% |
| End-to-End Success | 完整對話流程成功率 | > 80% |
