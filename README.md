# 地球股東會 ESG 分析平台

這是一個以 Flask 建立的 ESG 分析網站，整合永續報告書漂綠偵測、企業機器學習分析、線性回歸頁面與模型成果儀表板。網站風格以主頁地球插圖為基礎，採用粗線框、亮綠與藍色的手繪感介面。

## 功能說明

### 1. 首頁

網址：

```text
http://127.0.0.1:5000/
```

首頁提供四個主要入口：

- 永續報告書漂綠偵測
- 機器學習企業分析
- 線性回歸分析
- 模型成果儀表板

### 2. 永續報告書漂綠偵測

網址：

```text
http://127.0.0.1:5000/greenwashing
```

功能：

- 上傳企業永續報告書 PDF
- 使用 zero-shot NLP 模型分析段落
- 輸出整份報告漂綠風險分數
- 顯示高風險段落與模型判斷原因

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

### 4. 線性回歸分析

網址：

```text
http://127.0.0.1:5000/linear-esg
```

目前此頁保留為空白頁面，作為後續線性回歸模型展示的擴充位置。

### 5. 模型成果儀表板

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
│   ├── greenwashing_reference.pkl
│   └── zero_shot_model/
├── static/
│   ├── css/
│   ├── images/
│   └── js/
├── templates/
│   ├── index.html
│   ├── greenwashing.html
│   ├── ml_prediction.html
│   ├── linear_esg.html
│   └── dashboard.html
└── uploads/
```

## 環境需求

建議使用 Python 3.10 以上版本。

主要套件包含：

- Flask
- pandas
- PyMuPDF
- transformers
- torch
- joblib
- scikit-learn
- xgboost

完整套件列在：

```text
requirements.txt
```

## 如何啟動專案

### 1. 進入專案資料夾

```bash
cd /Users/yungching/Desktop/esg/ESG_predict_web
```

### 2. 建立虛擬環境

```bash
python3 -m venv .venv
```

### 3. 啟動虛擬環境

macOS / Linux：

```bash
source .venv/bin/activate
```

Windows：

```bash
.venv\Scripts\activate
```

### 4. 安裝套件

```bash
pip install -r requirements.txt
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

```bash
python3 app.py
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

### 2. PDF 無法分析

如果 PDF 是掃描檔或圖片型 PDF，PyMuPDF 可能無法抽取文字。此時系統會提示 PDF 文字太短或無法讀取。

### 3. 第一次分析報告書很慢

漂綠偵測會載入 NLP 模型，第一次執行通常較慢。模型載入後，同一次 server 執行期間會重複使用模型。

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
