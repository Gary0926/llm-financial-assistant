## 目前待辦

- 討論並定出這個 repo 要做出什麼有價值的差異化功能——目前只有問題診斷（三家主流 AI 已能直接查到目前功能等級的資訊），方向尚未拍板
- main.py 的全域 `history` / `stock_database` 等全域變數在多使用者情境下會互相污染對話紀錄，尚未修復
- `langchain-community` 已被官方標記 sunset，長期應遷移到獨立 integration package（如 FAISS 專屬套件），尚未動手
- 缺乏測試／CI，相依套件升級的相容性問題目前只能靠手動事後驗證才會被發現
- README 操作步驟裡的 notebook 檔名寫錯（寫成 `llm_financial_assistant.ipynb`，實際檔名是 `llm_financal_assistant.ipynb`），尚未修正

## 2026-09-09

- 誰跑的：Claude（Sonnet 5），部分程式修正委派 Codex 執行
- 跑了什麼：
  - 因 GitHub 安全性警報（Flask CVE-2026-27205）觸發，用 pip-audit 對 requirements.txt 做完整相依鏈掃描（含間接相依，共 100+ 個套件）
  - 升級 Flask、requests、yfinance、faiss-cpu、ipython、python-dotenv、line-bot-sdk 至各自最新版；為 langchain 系列補上版本釘選；移除多餘且過時的 `typing` 相依套件
  - 委派 Codex 修正因 langchain 1.x 重構而失效的兩處 import（`langchain.prompts`→`langchain_core.prompts`、`langchain.schema`→`langchain_core.documents`），並親自複查 diff 與重新驗證
  - 判定 repo 為公開專案，設定 `.claude/project-type = personal-open`，新增 `.gitignore`
  - 評估整體 repo 現況：功能開發自 2025-05 起停滯，模型寫死用 gpt-4、LangSmith 環境變數命名過時、無測試/CI
- 結果：
  - `chore: 新增 .gitignore 排除本機機器狀態檔案`（commit d30415e）
  - `fix: 升級相依套件並修復 langchain 1.x import 相容性`（commit 7c4e7d7）
  - pip-audit 確認升級後 0 個已知漏洞
- 接手前先看：
  - 這次最重要的結論是方向性的，不是技術性的：套件過時只是表面問題，真正的癥結是這個 repo 目前提供的功能（LLM+RAG+股價/新聞查詢的 LINE Bot）現在主流 AI 聊天工具本身就做得到，repo 沒有不可取代的價值。下次接手應優先討論產品方向，而不是繼續補技術債。
