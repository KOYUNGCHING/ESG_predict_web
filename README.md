# 地球股東會 ESG 分析平台

這是一個以 Flask 建立的 ESG 分析網站，整合永續報告書漂綠偵測、企業機器學習分析與模型成果儀表板。網站風格以主頁地球插圖為基礎，採用粗線框、亮綠與藍色的手繪感介面。

## 功能說明

### 1. 首頁

網址：

```text
http://127.0.0.1:5000/
```

首頁提供四個主要入口：

- 永續報告書漂綠偵測
- 機器學習企業分析
- 模型成果儀表板

### 2. 永續報告書漂綠偵測

網址：

```text
http://127.0.0.1:5000/greenwashing
```

功能：

- 從 `data_2/` 選擇已批次分析完成的公司年度報告
- 直接輸出整份報告漂綠風險分數
- 顯示高風險段落與既有分析原因
- 網站端不重新上傳 PDF，也不重新執行 NLP 模型

### 3. 機器學習企業分析

網址：

```text
http://127.0.0.1:5000/ml-prediction
```

輸入資料：

- TESG 分數
- 環境構面分數 E
- 社會構面分數 S
- 公司治理構面分數 G
- 公司規模 / 平均市值
- 前一期 ROA
- 目前 ROA

輸出結果：

- 未來 ROA 預測
- 財務風險分類
- 財務風險分類機率
- 公司 ESG / 獲利分群
- 表格型漂綠風險提示
- ROA 預測依據
- 財務風險依據
- 公司分群依據
- 漂綠警訊依據
- ROA 模型特徵重要度

### 4. 模型成果儀表板

網址：

```text
http://127.0.0.1:5000/dashboard
```

功能：

- 顯示訓練出的模型比較
- 顯示最佳 ROA 預測模型
- 顯示 Random Forest 特徵重要度
- 顯示描述統計
- 顯示線性模型係數

## 專案結構

```text
ESG_predict_web/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── descriptive_statistics_table.csv
│   ├── linear_coefficients.csv
│   ├── master_full_merged.csv
│   ├── model_base_merged.csv
│   ├── model_results_summary.csv
│   └── rf_feature_importance.csv
├── data_2/
│   ├── 台新_2013_2024_batch_summary.csv
│   ├── 國泰_2011_2024_batch_summary.csv
│   ├── 富邦_2006_2024_batch_summary.csv
│   ├── 彰銀_2014_2024_batch_summary.csv
│   └── 統一證_2009_2024_batch_summary.csv
├── models/
│   ├── roa_prediction_model.joblib
│   ├── model_features.pkl
│   ├── model_features.joblib
│   ├── financial_risk_classifier.pkl
│   ├── financial_risk_features.pkl
│   ├── kmeans_cluster_model.pkl
│   ├── cluster_scaler.pkl
│   ├── cluster_features.pkl
│   ├── cluster_names.pkl
│   └── greenwashing_reference.pkl
├── static/
│   ├── css/
│   ├── images/
│   └── js/
├── templates/
│   ├── index.html
│   ├── greenwashing.html
│   ├── ml_prediction.html
│   └── dashboard.html
└── uploads/
```

## 環境需求

建議使用 Python 3.10 以上版本。

主要套件包含：

- Flask
- pandas
- joblib
- scikit-learn
- xgboost

完整套件列在：

```text
requirements.txt
```

## 如何啟動專案

### 1. 進入專案資料夾

macOS / Linux：

```bash
cd /Users/你的使用者名稱/Desktop/esg/ESG_predict_web
```

Windows：

```powershell
cd C:\Users\你的使用者名稱\Desktop\esg\ESG_predict_web
```

### 2. 建立虛擬環境

macOS / Linux：

```bash
python3 -m venv .venv
```

Windows：

```powershell
py -m venv .venv
```

### 3. 啟動虛擬環境

macOS / Linux：

```bash
source .venv/bin/activate
```

Windows：

```powershell
.venv\Scripts\activate
```

如果 Windows PowerShell 顯示不能執行 scripts，可以先執行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

再重新啟動虛擬環境：

```powershell
.venv\Scripts\activate
```

### 4. 安裝套件

macOS / Linux：

```bash
pip install -r requirements.txt
```

Windows：

```powershell
python -m pip install -r requirements.txt
```

### 5. 確認模型與資料檔案

請確認以下資料夾存在：

```text
models/
data/
```

`models/` 需要放置訓練好的模型檔，例如：

```text
roa_prediction_model.joblib
model_features.pkl
financial_risk_classifier.pkl
kmeans_cluster_model.pkl
greenwashing_reference.pkl
```

`data/` 需要放置 Colab 匯出的 CSV，例如：

```text
model_results_summary.csv
rf_feature_importance.csv
descriptive_statistics_table.csv
linear_coefficients.csv
model_base_merged.csv
```

### 6. 啟動 Flask 網站

macOS / Linux：

```bash
python3 app.py
```

Windows：

```powershell
python app.py
```

啟動成功後，終端機會顯示：

```text
Running on http://127.0.0.1:5000
```

打開瀏覽器進入：

```text
http://127.0.0.1:5000
```

## 常見問題

### 1. 找不到模型檔

如果出現類似訊息：

```text
找不到 models/roa_prediction_model.joblib
```

請確認 Colab 匯出的模型檔是否已放入 `models/`。

### 2. 永續頁沒有報告選項

請確認 `data_2/` 內有批次分析完成的 CSV，且欄位包含 `company` 可由檔名推得、`year`、整體分數與 top 1 至 top 5 段落欄位。

### 3. 查不到某份永續報告

永續頁使用「公司名稱｜年度」查詢固定資料。如果 CSV 檔名或 `year` 欄位有誤，頁面會顯示找不到固定分析結果。

### 4. model_features.pkl 和模型欄位不一致

程式會優先使用模型內部的 `feature_names_in_`，避免欄位版本不同造成預測失敗。

## 資料來源與模型來源

本專案的模型與資料主要來自 Colab 分析流程，包含：

- ROA_next 預測模型
- 財務風險分類模型
- K-Means 公司分群模型
- Greenwashing Risk 參考值
- 模型比較與特徵重要度表格

## 備註

本專案目前為課程期末展示用途，模型結果僅供研究與展示參考，不應作為正式投資建議或企業評等結論。
