from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
from transformers import pipeline


# =========================
# 0. Flask 基本設定
# =========================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"pdf"}

# 自動建立資料夾
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ML 模型路徑
ROA_MODEL_PATH = os.path.join(MODEL_FOLDER, "roa_prediction_model.joblib")
RISK_MODEL_PATH = os.path.join(MODEL_FOLDER, "financial_risk_classifier.pkl")
RISK_FEATURES_PATH = os.path.join(MODEL_FOLDER, "financial_risk_features.pkl")
KMEANS_MODEL_PATH = os.path.join(MODEL_FOLDER, "kmeans_cluster_model.pkl")
CLUSTER_SCALER_PATH = os.path.join(MODEL_FOLDER, "cluster_scaler.pkl")
CLUSTER_FEATURES_PATH = os.path.join(MODEL_FOLDER, "cluster_features.pkl")
CLUSTER_NAMES_PATH = os.path.join(MODEL_FOLDER, "cluster_names.pkl")
GREEN_REF_PATH = os.path.join(MODEL_FOLDER, "greenwashing_reference.pkl")
FEATURES_PATHS = [
    os.path.join(MODEL_FOLDER, "model_features.pkl"),
    os.path.join(MODEL_FOLDER, "model_features.joblib"),
]
DATA_FILES = {
    "model_results": os.path.join(DATA_FOLDER, "model_results_summary.csv"),
    "feature_importance": os.path.join(DATA_FOLDER, "rf_feature_importance.csv"),
    "descriptive_stats": os.path.join(DATA_FOLDER, "descriptive_statistics_table.csv"),
    "linear_coefficients": os.path.join(DATA_FOLDER, "linear_coefficients.csv"),
    "model_base": os.path.join(DATA_FOLDER, "model_base_merged.csv"),
}
ZERO_SHOT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ZERO_SHOT_MODEL_DIR = os.path.join(MODEL_FOLDER, "zero_shot_model")
ZERO_SHOT_LABELS = [
    "strong sustainability commitment",
    "specific quantitative evidence",
    "third-party verification or credible disclosure",
    "vague promotional sustainability claim"
]


# =========================
# 1. NLP 模型載入：漂綠偵測
# =========================

zero_shot_clf = None


def load_zero_shot_pipeline():
    """
    載入 zero-shot NLP 模型。
    若 models/zero_shot_model 存在，就優先載入本機匯出的模型。
    """
    model_source = (
        ZERO_SHOT_MODEL_DIR
        if os.path.isdir(ZERO_SHOT_MODEL_DIR)
        else ZERO_SHOT_MODEL_NAME
    )

    clf = pipeline(
        "zero-shot-classification",
        model=model_source,
        tokenizer=model_source,
        local_files_only=os.path.isdir(ZERO_SHOT_MODEL_DIR)
    )
    return clf


def get_zero_shot_model():
    """
    讓 NLP 模型只載入一次。
    """
    global zero_shot_clf

    if zero_shot_clf is None:
        print("Loading zero-shot NLP model...")
        zero_shot_clf = load_zero_shot_pipeline()
        print("NLP model loaded.")

    return zero_shot_clf


# =========================
# 2. ML 模型載入：未來 ROA 預測
# =========================

roa_model = None
roa_features = None
risk_model = None
risk_features = None
kmeans_model = None
cluster_scaler = None
cluster_features = None
cluster_names = None
green_ref = None


def load_roa_prediction_model():
    """
    載入已訓練好的機器學習模型與特徵欄位。
    用來預測未來 ROA。
    """
    global roa_model, roa_features

    if roa_model is None or roa_features is None:

        if not os.path.exists(ROA_MODEL_PATH):
            raise FileNotFoundError(
                "找不到 models/roa_prediction_model.joblib。"
                "請先把 Colab 下載的模型檔放到 models 資料夾。"
            )

        features_path = next(
            (path for path in FEATURES_PATHS if os.path.exists(path)),
            None
        )

        if features_path is None:
            raise FileNotFoundError(
                "找不到 models/model_features.pkl 或 models/model_features.joblib。"
                "請先把特徵列表檔放到 models 資料夾。"
            )

        roa_model = joblib.load(ROA_MODEL_PATH)
        roa_features = joblib.load(features_path)

        if not hasattr(roa_model, "predict"):
            raise TypeError(
                "models/roa_prediction_model.joblib 不是可預測的模型物件。"
                "目前讀到的內容沒有 predict() 方法，請確認這個檔案是訓練好的模型，"
                "不是特徵列表。"
            )

        print("ROA prediction model loaded.")
        print("Model features:", roa_features)

    return roa_model, roa_features


def load_joblib_if_exists(path):
    """
    安全讀取模型檔。
    如果某個延伸模型還沒放進 models/，頁面仍會保留可用的 ROA 預測。
    """
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_enterprise_models():
    """
    載入機器學習企業分析頁需要的延伸模型。
    包含財務風險分類、K-Means 分群，以及表格型漂綠風險參考值。
    """
    global risk_model, risk_features
    global kmeans_model, cluster_scaler, cluster_features, cluster_names, green_ref

    if risk_model is None:
        risk_model = load_joblib_if_exists(RISK_MODEL_PATH)
        risk_features = load_joblib_if_exists(RISK_FEATURES_PATH)

    if kmeans_model is None:
        kmeans_model = load_joblib_if_exists(KMEANS_MODEL_PATH)
        cluster_scaler = load_joblib_if_exists(CLUSTER_SCALER_PATH)
        cluster_features = load_joblib_if_exists(CLUSTER_FEATURES_PATH)
        cluster_names = load_joblib_if_exists(CLUSTER_NAMES_PATH)

    if green_ref is None:
        green_ref = load_joblib_if_exists(GREEN_REF_PATH) or build_greenwashing_reference_from_csv()

    return {
        "risk_ready": risk_model is not None and risk_features is not None,
        "cluster_ready": (
            kmeans_model is not None
            and cluster_scaler is not None
            and cluster_features is not None
            and cluster_names is not None
        ),
        "green_ready": green_ref is not None,
    }


def read_csv_if_exists(path):
    """
    統一讀取 Colab 匯出的 CSV，處理 utf-8-sig 的 BOM。
    """
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def build_greenwashing_reference_from_csv():
    """
    若沒有 greenwashing_reference.pkl，就用 Colab 匯出的資料表建立平均參考值。
    """
    ref = {}
    stats_df = read_csv_if_exists(DATA_FILES["descriptive_stats"])

    if stats_df is not None and {"變數", "均值"}.issubset(stats_df.columns):
        aliases = {
            "ESG總分": "esg_mean",
            "E（環境）": "e_mean",
            "ROA": "roa_mean",
        }
        for source_name, target_name in aliases.items():
            row = stats_df.loc[stats_df["變數"] == source_name]
            if not row.empty:
                ref[target_name] = float(row.iloc[0]["均值"])

    base_df = read_csv_if_exists(DATA_FILES["model_base"])
    if base_df is not None:
        ref.setdefault("esg_mean", float(base_df["TESG分數"].mean()))
        ref.setdefault("e_mean", float(base_df["環境構面分數"].mean()))
        ref.setdefault("roa_mean", float(base_df["ROA_avg"].mean()))

    if {"esg_mean", "e_mean", "roa_mean"}.issubset(ref.keys()):
        ref["verify_col"] = None
        return ref
    return None


# =========================
# 3. PDF 基本工具
# =========================

def allowed_file(filename):
    """
    檢查檔案是否為允許格式。
    目前只允許 PDF。
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """
    使用 PyMuPDF 從 PDF 抽取文字。
    如果 PDF 是掃描檔，這個方法可能抽不到文字。
    """
    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:
        page_text = page.get_text()
        text += page_text + "\n"

    doc.close()

    return text


def split_text_into_chunks(text, max_chars=900):
    """
    將報告文字切成 chunks。
    """
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= max_chars:
            current += " " + p
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p

    if current.strip():
        chunks.append(current.strip())

    return chunks


def truncate_text(text, max_chars=900):
    """
    避免送進 NLP model 的文字過長。
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# =========================
# 4. NLP 漂綠偵測核心
# =========================

def build_greenwashing_scores(result):
    scores = dict(zip(result["labels"], result["scores"]))

    commitment_score = scores.get("strong sustainability commitment", 0) * 100
    evidence_score = scores.get("specific quantitative evidence", 0) * 100
    credibility_score = scores.get("third-party verification or credible disclosure", 0) * 100
    vague_claim_score = scores.get("vague promotional sustainability claim", 0) * 100

    commitment_evidence_gap = max(0, commitment_score - evidence_score)

    greenwashing_risk = (
        0.35 * commitment_evidence_gap +
        0.30 * vague_claim_score +
        0.20 * (100 - credibility_score) +
        0.15 * (100 - evidence_score)
    )

    greenwashing_risk = max(0, min(100, greenwashing_risk))

    if greenwashing_risk >= 60:
        risk_level = "High"
    elif greenwashing_risk >= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    explanation = generate_nlp_explanation(
        commitment_score,
        evidence_score,
        credibility_score,
        vague_claim_score,
        greenwashing_risk
    )

    return {
        "commitment_score": round(commitment_score, 2),
        "evidence_score": round(evidence_score, 2),
        "credibility_score": round(credibility_score, 2),
        "vague_claim_score": round(vague_claim_score, 2),
        "greenwashing_risk": round(greenwashing_risk, 2),
        "risk_level": risk_level,
        "explanation": explanation
    }


def score_chunk_with_nlp(text, clf):
    """
    使用 zero-shot NLP model 分析單一段落。
    分析概念：
    承諾高 + 證據低 + 可信度低 + 空泛宣傳高
    => 漂綠風險高
    """

    text_for_model = truncate_text(text, max_chars=900)

    result = clf(
        text_for_model,
        candidate_labels=ZERO_SHOT_LABELS,
        multi_label=True
    )

    return build_greenwashing_scores(result)


def score_chunks_with_nlp(chunks, clf):
    """
    批次分析多個段落，減少反覆呼叫 pipeline 的等待時間。
    """
    texts_for_model = [truncate_text(chunk, max_chars=900) for chunk in chunks]
    results = clf(
        texts_for_model,
        candidate_labels=ZERO_SHOT_LABELS,
        multi_label=True,
        batch_size=4
    )
    return [build_greenwashing_scores(result) for result in results]


def generate_nlp_explanation(
    commitment_score,
    evidence_score,
    credibility_score,
    vague_claim_score,
    greenwashing_risk
):
    """
    根據 NLP 分數產生原因。
    """
    reasons = []

    if commitment_score >= 60:
        reasons.append("此段落被 NLP 模型判斷為具有較強的永續承諾、目標或願景敘述。")

    if evidence_score < 40:
        reasons.append("具體量化證據分數偏低，代表段落中可能缺少明確數據、百分比、年份或成果指標。")

    if vague_claim_score >= 50:
        reasons.append("模型偵測到較高比例的空泛或宣傳式永續敘述。")

    if credibility_score < 40:
        reasons.append("可信揭露或第三方驗證訊號較弱，因此降低整體可信度。")

    if greenwashing_risk >= 60:
        reasons.append("綜合以上因素，此段落被歸類為高漂綠風險。")
    elif greenwashing_risk >= 30:
        reasons.append("綜合以上因素，此段落被歸類為中等漂綠風險。")
    else:
        reasons.append("綜合以上因素，此段落的漂綠風險較低。")

    return " ".join(reasons)


# =========================
# 5. 漂綠整份報告彙整
# =========================

def summarize_report(results_df):
    """
    將所有 chunk 的結果彙整成整份報告結果。
    """

    avg_commitment = results_df["commitment_score"].mean()
    avg_evidence = results_df["evidence_score"].mean()
    avg_credibility = results_df["credibility_score"].mean()
    avg_vague_claim = results_df["vague_claim_score"].mean()
    avg_greenwashing = results_df["greenwashing_risk"].mean()

    max_greenwashing = results_df["greenwashing_risk"].max()

    high_risk_ratio = (results_df["risk_level"] == "High").mean()
    medium_risk_ratio = (results_df["risk_level"] == "Medium").mean()
    low_risk_ratio = (results_df["risk_level"] == "Low").mean()

    report_score = (
        0.5 * avg_greenwashing +
        0.3 * max_greenwashing +
        0.2 * high_risk_ratio * 100
    )

    report_score = max(0, min(100, report_score))

    if report_score >= 60:
        report_level = "High Greenwashing Risk"
    elif report_score >= 30:
        report_level = "Medium Greenwashing Risk"
    else:
        report_level = "Low Greenwashing Risk"

    top_row = results_df.sort_values("greenwashing_risk", ascending=False).iloc[0]

    summary_reason = generate_report_reason(
        avg_commitment=avg_commitment,
        avg_evidence=avg_evidence,
        avg_credibility=avg_credibility,
        avg_vague_claim=avg_vague_claim,
        max_greenwashing=max_greenwashing,
        high_risk_ratio=high_risk_ratio,
        report_level=report_level
    )

    return {
        "n_chunks": len(results_df),

        "avg_commitment_score": round(avg_commitment, 2),
        "avg_evidence_score": round(avg_evidence, 2),
        "avg_credibility_score": round(avg_credibility, 2),
        "avg_vague_claim_score": round(avg_vague_claim, 2),

        "avg_greenwashing_risk": round(avg_greenwashing, 2),
        "max_greenwashing_risk": round(max_greenwashing, 2),

        "high_risk_ratio": round(high_risk_ratio, 2),
        "medium_risk_ratio": round(medium_risk_ratio, 2),
        "low_risk_ratio": round(low_risk_ratio, 2),

        "report_greenwashing_score": round(report_score, 2),
        "report_level": report_level,
        "summary_reason": summary_reason,

        "top_risk_text": top_row["text"],
        "top_risk_score": round(top_row["greenwashing_risk"], 2),
        "top_risk_level": top_row["risk_level"],
        "top_risk_explanation": top_row["explanation"]
    }


def generate_report_reason(
    avg_commitment,
    avg_evidence,
    avg_credibility,
    avg_vague_claim,
    max_greenwashing,
    high_risk_ratio,
    report_level
):
    """
    產生整份報告的原因解釋。
    """

    reasons = []

    if avg_commitment > avg_evidence + 20:
        reasons.append("整體而言，報告中的永續承諾分數明顯高於具體證據分數。")

    if avg_evidence < 40:
        reasons.append("整體量化證據分數偏低，代表部分內容可能缺少明確數據、指標或成果佐證。")

    if avg_credibility < 40:
        reasons.append("可信揭露與第三方驗證訊號偏弱，可能降低報告內容的可信度。")

    if avg_vague_claim >= 50:
        reasons.append("報告中有較多空泛或宣傳式的永續敘述。")

    if max_greenwashing >= 60:
        reasons.append("報告中至少有一個段落被判定為高漂綠風險。")

    if high_risk_ratio >= 0.1:
        reasons.append("高風險段落占比達到需要注意的程度。")

    if not reasons:
        reasons.append("大多數段落的承諾、證據與可信度相對平衡，沒有明顯漂綠訊號。")

    if "High" in report_level:
        conclusion = "整體而言，這份報告具有較高的潛在漂綠風險。"
    elif "Medium" in report_level:
        conclusion = "整體而言，這份報告具有中等程度的潛在漂綠風險。"
    else:
        conclusion = "整體而言，這份報告的潛在漂綠風險較低。"

    return conclusion + " " + " ".join(reasons)


# =========================
# 6. ML 預測解釋
# =========================

def get_reference_metrics():
    """
    取得展示用參考基準。
    優先使用 data/model_base_merged.csv 的實際樣本平均，缺資料時再使用 green_ref。
    """
    base_df = read_csv_if_exists(DATA_FILES["model_base"])
    reference = {}

    if base_df is not None and not base_df.empty:
        column_map = {
            "TESG分數": "esg_mean",
            "環境構面分數": "e_mean",
            "社會構面分數": "s_mean",
            "公司治理構面分數": "g_mean",
            "市值_avg": "market_value_mean",
            "ROA_avg": "roa_mean",
        }
        for column, key in column_map.items():
            if column in base_df.columns:
                reference[key] = float(base_df[column].mean())
        reference["sample_count"] = int(len(base_df))

    if green_ref is not None:
        reference.setdefault("esg_mean", float(green_ref["esg_mean"]))
        reference.setdefault("e_mean", float(green_ref["e_mean"]))
        reference.setdefault("roa_mean", float(green_ref["roa_mean"]))

    return reference


def get_feature_importance_rows(limit=5):
    """
    讀取 Random Forest 特徵重要度，提供 ROA 預測理由使用。
    """
    importance_df = read_csv_if_exists(DATA_FILES["feature_importance"])
    if importance_df is None or importance_df.empty:
        return []

    rows = importance_df.head(limit).to_dict(orient="records")
    return [
        {
            "feature": row["Feature"],
            "importance": round(float(row["Importance"]), 3),
        }
        for row in rows
    ]


def describe_against_average(label, value, average, unit=""):
    """
    產生「高於 / 低於樣本平均」的中文描述。
    """
    diff = value - average
    direction = "高於" if diff >= 0 else "低於"
    return f"{label}為 {value:.2f}{unit}，{direction}樣本平均 {average:.2f}{unit}（差距 {diff:+.2f}{unit}）。"


def explain_roa_prediction(predicted_roa, user_input, active_features=None):
    """
    根據預測 ROA、樣本平均與特徵重要度產生較有依據的解釋。
    """

    reasons = []
    evidence = []
    reference = get_reference_metrics()
    importances = get_feature_importance_rows()

    esg = user_input.get("TESG分數", 0)
    e_score = user_input.get("環境構面分數", 0)
    s_score = user_input.get("社會構面分數", 0)
    g_score = user_input.get("公司治理構面分數", 0)
    firm_size = user_input.get("市值_avg", 0)
    roa_lag = user_input.get("ROA_lag", 0)

    if predicted_roa >= 2:
        level = "High Performance Potential"
        conclusion = "模型預測該公司的未來 ROA 表現相對較佳。"
    elif predicted_roa >= 0.5:
        level = "Medium Performance Potential"
        conclusion = "模型預測該公司的未來 ROA 表現屬於中等水準。"
    else:
        level = "Low Performance Potential"
        conclusion = "模型預測該公司的未來 ROA 表現可能偏低，需要注意財務績效風險。"

    roa_delta = predicted_roa - roa_lag
    if roa_delta >= 0:
        reasons.append(f"預測 ROA 較前一期 ROA 高 {roa_delta:.2f}，模型判斷獲利能力有改善訊號。")
    else:
        reasons.append(f"預測 ROA 較前一期 ROA 低 {abs(roa_delta):.2f}，模型判斷獲利能力可能轉弱。")

    if "roa_mean" in reference:
        reasons.append(describe_against_average("預測 ROA", predicted_roa, reference["roa_mean"]))

    comparisons = [
        ("ESG 總分", esg, "esg_mean"),
        ("環境 E", e_score, "e_mean"),
        ("社會 S", s_score, "s_mean"),
        ("治理 G", g_score, "g_mean"),
        ("平均市值", firm_size, "market_value_mean"),
    ]
    for label, value, ref_key in comparisons:
        if ref_key in reference:
            evidence.append(describe_against_average(label, value, reference[ref_key]))

    if importances:
        top_features = "、".join(
            f"{row['feature']}（重要度 {row['importance']}）"
            for row in importances[:3]
        )
        reasons.append(f"依照目前 Random Forest 特徵重要度，主要參考變數為：{top_features}。")

    if active_features:
        evidence.append("本次 ROA 模型實際使用欄位：" + "、".join(active_features) + "。")

    if "sample_count" in reference:
        evidence.append(f"參考樣本數：{reference['sample_count']} 筆公司年度資料。")

    if not reasons:
        reasons.append("模型主要根據 ESG 分數、E/S/G 構面、公司規模與前一期 ROA 綜合判斷。")

    return {
        "level": level,
        "conclusion": conclusion,
        "reasons": reasons,
        "evidence": evidence,
        "feature_importance": importances,
        "reference": reference,
    }


# =========================
# 7. 網站資料整理工具
# =========================


def ensure_model_columns(input_df, features):
    """
    補齊模型需要但使用者沒有輸入的欄位。
    這能避免 Colab 後續若多加欄位，網頁因欄位缺漏直接報錯。
    """
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    return input_df[features]


def get_model_features(model, fallback_features):
    """
    優先使用模型本身記錄的 feature_names_in_。
    這可以處理模型檔與 model_features.pkl 不小心版本不一致的情況。
    """
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        return list(model_features)
    return list(fallback_features)


def build_enterprise_input(form):
    """
    將機器學習表單轉成模型欄位。
    欄位名稱沿用 Colab 訓練資料，避免模型預測時欄位對不上。
    """
    return {
        "TESG分數": float(form.get("tesg_score", 0)),
        "環境構面分數": float(form.get("env_score", 0)),
        "社會構面分數": float(form.get("social_score", 0)),
        "公司治理構面分數": float(form.get("gov_score", 0)),
        "市值_avg": float(form.get("market_value", 0)),
        "ROA_lag": float(form.get("roa_lag", 0)),
        "ROA_avg": float(form.get("roa_avg", 0)),
    }


def predict_financial_risk(input_df):
    """
    使用財務風險分類模型輸出 High / Medium / Low Risk。
    如果模型缺檔，改用 ROA 與 ESG 的規則備援，讓展示不中斷。
    """
    if risk_model is not None and risk_features is not None:
        risk_input = ensure_model_columns(input_df.copy(), risk_features)
        prediction = risk_model.predict(risk_input)[0]
        probabilities = []

        if hasattr(risk_model, "predict_proba"):
            proba = risk_model.predict_proba(risk_input)[0]
            probabilities = [
                {
                    "label": str(label),
                    "probability": round(float(prob) * 100, 1),
                }
                for label, prob in zip(risk_model.classes_, proba)
            ]
            probabilities = sorted(
                probabilities,
                key=lambda row: row["probability"],
                reverse=True
            )

        explanation = [
            "財務風險由 Random Forest 分類模型判斷，輸入欄位包含 ESG、E/S/G、公司規模與前一期 ROA。",
            "模型會根據訓練資料中 High / Medium / Low Risk 的型態，判斷目前輸入較接近哪一類公司。",
        ]

        if probabilities:
            top = probabilities[0]
            explanation.append(f"最高分類信心為 {top['label']}：{top['probability']:.1f}%。")

        return {
            "label": str(prediction),
            "source": "模型分類",
            "probabilities": probabilities,
            "explanation": explanation,
        }

    roa_avg = float(input_df.iloc[0].get("ROA_avg", 0))
    roa_lag = float(input_df.iloc[0].get("ROA_lag", 0))
    esg = float(input_df.iloc[0].get("TESG分數", 0))

    if roa_avg < 0.5 or roa_lag < 0.5:
        label = "High Risk"
    elif roa_avg < 1.5 or esg < 50:
        label = "Medium Risk"
    else:
        label = "Low Risk"

    return {
        "label": label,
        "source": "規則備援",
        "probabilities": [],
        "explanation": [
            "財務風險模型檔尚未完整載入，因此使用 ROA 與 ESG 門檻做備援判斷。",
            f"目前 ROA 為 {roa_avg:.2f}，前一期 ROA 為 {roa_lag:.2f}，ESG 總分為 {esg:.2f}。",
        ],
    }


def predict_company_cluster(input_df):
    """
    使用 K-Means 模型判斷公司 ESG / 獲利型態。
    若分群模型缺檔，使用 ESG 與 ROA 平均值做簡易分型。
    """
    if (
        kmeans_model is not None
        and cluster_scaler is not None
        and cluster_features is not None
        and cluster_names is not None
    ):
        cluster_input = ensure_model_columns(input_df.copy(), cluster_features)
        cluster_input_scaled = cluster_scaler.transform(cluster_input)
        cluster_id = int(kmeans_model.predict(cluster_input_scaled)[0])
        distances = kmeans_model.transform(cluster_input_scaled)[0]
        nearest_distance = float(distances[cluster_id])
        sorted_distances = sorted(float(distance) for distance in distances)

        confidence_note = "分群邊界接近"
        if len(sorted_distances) > 1 and sorted_distances[1] > 0:
            separation = sorted_distances[1] - sorted_distances[0]
            if separation >= 0.8:
                confidence_note = "分群結果明確"
            elif separation >= 0.3:
                confidence_note = "分群結果中等明確"

        explanation = [
            "K-Means 分群使用 ESG、E/S/G、平均市值與目前 ROA，先經標準化後再分群。",
            f"本次公司最接近 Cluster {cluster_id}，與該群中心的標準化距離為 {nearest_distance:.2f}。",
            confidence_note,
        ]

        return {
            "name": cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            "id": f"Cluster {cluster_id}",
            "source": "K-Means 模型",
            "explanation": explanation,
        }

    esg = float(input_df.iloc[0].get("TESG分數", 0))
    roa_avg = float(input_df.iloc[0].get("ROA_avg", 0))
    esg_mean = green_ref["esg_mean"] if green_ref else 58.6
    roa_mean = green_ref["roa_mean"] if green_ref else 1.27

    if esg >= esg_mean and roa_avg >= roa_mean:
        name = "High ESG / High Profitability"
    elif esg >= esg_mean and roa_avg < roa_mean:
        name = "High ESG / Low Profitability"
    elif esg < esg_mean and roa_avg >= roa_mean:
        name = "Low ESG / High Profitability"
    else:
        name = "Low ESG / Low Profitability"

    return {
        "name": name,
        "id": "Rule Cluster",
        "source": "規則備援",
        "explanation": [
            "K-Means 模型檔尚未完整載入，因此使用 ESG 與 ROA 是否高於樣本平均做備援分型。",
            f"ESG 總分 {esg:.2f}，樣本平均 {esg_mean:.2f}；目前 ROA {roa_avg:.2f}，樣本平均 {roa_mean:.2f}。",
        ],
    }


def calculate_tabular_greenwashing_risk(user_input):
    """
    用 ESG、環境分數、ROA 與 E/S/G 平衡度估計企業層級漂綠警訊。
    這和 PDF NLP 偵測不同，適合在企業資料輸入頁做快速提醒。
    """
    if green_ref is None:
        return {
            "score": None,
            "level": "參考值尚未建立",
            "reasons": ["請放入 greenwashing_reference.pkl，或保留 data/ 描述統計表。"],
        }

    score = 0
    reasons = []
    esg = user_input["TESG分數"]
    e_score = user_input["環境構面分數"]
    s_score = user_input["社會構面分數"]
    g_score = user_input["公司治理構面分數"]
    roa_avg = user_input["ROA_avg"]

    if esg < green_ref["esg_mean"]:
        score += 25
        reasons.append(f"ESG 總分低於樣本平均 {green_ref['esg_mean']:.2f}。")

    if e_score < green_ref["e_mean"]:
        score += 20
        reasons.append(f"環境構面分數低於樣本平均 {green_ref['e_mean']:.2f}。")

    if esg >= green_ref["esg_mean"] and roa_avg < green_ref["roa_mean"]:
        score += 20
        reasons.append(f"ESG 高於平均，但目前 ROA 低於樣本平均 {green_ref['roa_mean']:.2f}。")

    if np.std([e_score, s_score, g_score]) > 15:
        score += 20
        reasons.append("E/S/G 三構面落差較大，永續表現可能不均衡。")

    if roa_avg < 0:
        score += 15
        reasons.append("目前 ROA 為負值，需留意永續敘事與財務表現落差。")

    score = min(score, 100)

    if score >= 70:
        level = "高漂綠風險"
    elif score >= 40:
        level = "中等漂綠風險"
    else:
        level = "低漂綠風險"

    if not reasons:
        reasons.append("ESG、環境分數、ROA 與構面平衡度沒有出現明顯警訊。")

    return {
        "score": int(score),
        "level": level,
        "reasons": reasons,
    }


def build_dashboard_data():
    """
    整理 Colab 匯出的 CSV，提供 dashboard.html 直接渲染。
    """
    model_results = read_csv_if_exists(DATA_FILES["model_results"])
    feature_importance = read_csv_if_exists(DATA_FILES["feature_importance"])
    descriptive_stats = read_csv_if_exists(DATA_FILES["descriptive_stats"])
    linear_coefficients = read_csv_if_exists(DATA_FILES["linear_coefficients"])
    model_base = read_csv_if_exists(DATA_FILES["model_base"])

    best_model = None
    if model_results is not None and not model_results.empty and "R2" in model_results.columns:
        best_row = model_results.sort_values("R2", ascending=False).iloc[0]
        best_model = {
            "name": best_row["Model"],
            "r2": round(float(best_row["R2"]), 3),
            "rmse": round(float(best_row["RMSE"]), 3),
            "mae": round(float(best_row["MAE"]), 3),
        }

    overview = []
    if model_base is not None:
        overview = [
            {"label": "公司年度樣本", "value": f"{len(model_base):,}"},
            {"label": "平均 ESG", "value": f"{model_base['TESG分數'].mean():.2f}"},
            {"label": "平均 ROA", "value": f"{model_base['ROA_avg'].mean():.2f}"},
            {"label": "平均市值", "value": f"{model_base['市值_avg'].mean():,.0f}"},
        ]

    return {
        "overview": overview,
        "best_model": best_model,
        "model_results": model_results.to_dict(orient="records") if model_results is not None else [],
        "feature_importance": feature_importance.head(8).to_dict(orient="records") if feature_importance is not None else [],
        "descriptive_stats": descriptive_stats.to_dict(orient="records") if descriptive_stats is not None else [],
        "linear_coefficients": linear_coefficients.to_dict(orient="records") if linear_coefficients is not None else [],
    }


# =========================
# 8. Flask Routes
# =========================

@app.route("/")
def index():
    """
    首頁：
    主視覺 + 三個模型選擇區。
    """
    return render_template("index.html")


@app.route("/greenwashing", methods=["GET", "POST"])
def greenwashing():
    """
    永續報告書漂綠偵測頁面。
    GET：顯示上傳頁面
    POST：接收 PDF，進行 NLP 分析，輸出結果
    """

    result = None
    error = None

    if request.method == "POST":

        if "report_file" not in request.files:
            error = "沒有收到檔案，請重新上傳。"
            return render_template("greenwashing.html", result=result, error=error)

        file = request.files["report_file"]

        if file.filename == "":
            error = "請先選擇一份 PDF 永續報告書。"
            return render_template("greenwashing.html", result=result, error=error)

        if not allowed_file(file.filename):
            error = "目前只支援 PDF 檔案。"
            return render_template("greenwashing.html", result=result, error=error)

        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(pdf_path)

        try:
            text = extract_text_from_pdf(pdf_path)

            if len(text.strip()) < 500:
                error = "PDF 文字太短，可能是掃描檔、圖片型 PDF，或無法直接讀取文字。"
                return render_template("greenwashing.html", result=result, error=error)

            chunks = split_text_into_chunks(text, max_chars=900)

            if len(chunks) == 0:
                error = "無法從 PDF 中切出可分析的文字段落。"
                return render_template("greenwashing.html", result=result, error=error)

            # demo 先限制前 15 段，讓網頁分析時間比較可控。
            chunks = chunks[:15]

            clf = get_zero_shot_model()

            rows = []
            chunk_scores = score_chunks_with_nlp(chunks, clf)

            for i, (chunk, scores) in enumerate(zip(chunks, chunk_scores)):
                rows.append({
                    "chunk_id": i,
                    "text": chunk,
                    **scores
                })

            results_df = pd.DataFrame(rows)

            if results_df.empty:
                error = "分析結果為空，請確認 PDF 是否含有可讀文字。"
                return render_template("greenwashing.html", result=result, error=error)

            summary = summarize_report(results_df)

            top_chunks = results_df.sort_values(
                "greenwashing_risk",
                ascending=False
            ).head(5)

            result = {
                "filename": filename,
                "summary": summary,
                "top_chunks": top_chunks.to_dict(orient="records")
            }

        except Exception as e:
            error = f"分析失敗：{str(e)}"

    return render_template("greenwashing.html", result=result, error=error)


@app.route("/dashboard")
def dashboard():
    """
    Colab 成果儀表板。
    直接讀取匯出的 CSV，展示模型表現、特徵重要度與描述統計。
    """
    dashboard_data = build_dashboard_data()
    return render_template("dashboard.html", dashboard=dashboard_data)


@app.route("/linear-esg")
def linear_esg():
    """
    線性回歸分析頁面。
    目前先保留乾淨空白頁，方便後續擴充線性模型展示。
    """
    return render_template("linear_esg.html")


@app.route("/ml-prediction", methods=["GET", "POST"])
def ml_prediction():
    """
    機器學習企業分析頁面。
    同一份企業資料會輸出 ROA 預測、財務風險、公司分群與漂綠警訊。
    """

    result = None
    error = None
    model_status = load_enterprise_models()

    if request.method == "POST":
        try:
            model, features = load_roa_prediction_model()
            user_input = build_enterprise_input(request.form)
            input_df = pd.DataFrame([user_input])

            # 1. ROA 預測：欄位優先採用模型實際記錄的 feature_names_in_。
            active_features = get_model_features(model, features)
            roa_input = ensure_model_columns(input_df.copy(), active_features)
            predicted_roa = float(model.predict(roa_input)[0])
            roa_explanation = explain_roa_prediction(
                predicted_roa,
                user_input,
                active_features=active_features
            )

            # 2. 財務風險、公司分群、漂綠警訊：使用 Colab 匯出的延伸模型與參考值。
            financial_risk = predict_financial_risk(input_df)
            cluster = predict_company_cluster(input_df)
            greenwashing = calculate_tabular_greenwashing_risk(user_input)

            if predicted_roa >= user_input["ROA_lag"]:
                trend = "改善"
            else:
                trend = "下降"

            result = {
                "predicted_roa": round(predicted_roa, 4),
                "level": roa_explanation["level"],
                "conclusion": roa_explanation["conclusion"],
                "reasons": roa_explanation["reasons"],
                "roa_evidence": roa_explanation["evidence"],
                "feature_importance": roa_explanation["feature_importance"],
                "reference": roa_explanation["reference"],
                "financial_risk": financial_risk["label"],
                "financial_risk_source": financial_risk["source"],
                "financial_risk_probabilities": financial_risk["probabilities"],
                "financial_risk_explanation": financial_risk["explanation"],
                "cluster_name": cluster["name"],
                "cluster_id": cluster["id"],
                "cluster_source": cluster["source"],
                "cluster_explanation": cluster["explanation"],
                "greenwashing_score": greenwashing["score"],
                "greenwashing_level": greenwashing["level"],
                "greenwashing_reasons": greenwashing["reasons"],
                "trend": trend,
                "features": user_input,
                "model_features": active_features
            }

        except Exception as e:
            error = f"預測失敗：{str(e)}"

    return render_template(
        "ml_prediction.html",
        result=result,
        error=error,
        model_status=model_status
    )


# =========================
# 9. 啟動 Flask
# =========================

if __name__ == "__main__":
    app.run(debug=True)
