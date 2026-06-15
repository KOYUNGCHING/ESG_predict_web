from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib


# =========================
# 0. Flask 基本設定
# =========================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
GREENWASHING_DATA_FOLDER = os.path.join(BASE_DIR, "data_2")

# 自動建立資料夾
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
GREENWASHING_DATA_FILES = [
    os.path.join(GREENWASHING_DATA_FOLDER, filename)
    for filename in os.listdir(GREENWASHING_DATA_FOLDER)
    if filename.endswith(".csv")
] if os.path.isdir(GREENWASHING_DATA_FOLDER) else []


# =========================
# 1. ML 模型載入：未來 ROA 預測
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


def load_greenwashing_dataset():
    """
    讀取 data_2 內已經批次分析完成的永續報告書漂綠結果。
    每一列代表一份公司年度報告，不在網站端重新跑 NLP 模型。
    """
    frames = []

    for path in GREENWASHING_DATA_FILES:
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        company_name = os.path.basename(path).split("_")[0]
        df = df.copy()
        df["company"] = company_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    dataset["year"] = dataset["year"].astype(int)
    return dataset.sort_values(["company", "year"], ascending=[True, False])


def get_greenwashing_options():
    """
    建立下拉選單選項。
    value 用 company|year，避免不同公司同年度報告互相混淆。
    """
    dataset = load_greenwashing_dataset()
    if dataset.empty:
        return []

    options = []
    for _, row in dataset.iterrows():
        company = row["company"]
        year = int(row["year"])
        report_name = row.get("report_name") or f"{company} {year}"
        options.append({
            "value": f"{company}|{year}",
            "label": f"{company}｜{year}｜{report_name}",
            "company": company,
            "year": year
        })

    return options


def format_report_level(level):
    """
    將 CSV 裡的 Low / Medium / High 顯示成原本頁面使用的整體風險標籤。
    """
    levels = {
        "High": "High Greenwashing Risk",
        "Medium": "Medium Greenwashing Risk",
        "Low": "Low Greenwashing Risk",
    }
    return levels.get(str(level), str(level))


def get_cell(row, column, default=None):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return value


def build_top_chunks_from_row(row):
    chunks = []

    for rank in range(1, 6):
        text = get_cell(row, f"top{rank}_text")
        if not text:
            continue

        chunks.append({
            "chunk_id": int(get_cell(row, f"top{rank}_chunk_id", rank)),
            "risk_level": get_cell(row, f"top{rank}_risk_level", "-"),
            "greenwashing_risk": float(get_cell(row, f"top{rank}_risk_score", 0)),
            "commitment_score": float(get_cell(row, f"top{rank}_commitment_score", 0)),
            "evidence_score": float(get_cell(row, f"top{rank}_evidence_score", 0)),
            "credibility_score": float(get_cell(row, f"top{rank}_credibility_score", 0)),
            "explanation": get_cell(row, f"top{rank}_analysis", ""),
            "text": text,
        })

    return chunks


def build_greenwashing_result_from_row(row):
    top_chunks = build_top_chunks_from_row(row)
    top_chunk = top_chunks[0] if top_chunks else {}

    summary = {
        "n_chunks": int(get_cell(row, "n_chunks", 0)),
        "avg_commitment_score": round(float(get_cell(row, "avg_commitment_score", 0)), 2),
        "avg_evidence_score": round(float(get_cell(row, "avg_evidence_score", 0)), 2),
        "avg_credibility_score": round(float(get_cell(row, "avg_credibility_score", 0)), 2),
        "top5_avg_greenwashing_risk": round(float(get_cell(row, "top5_avg_greenwashing_risk", 0)), 2),
        "avg_greenwashing_risk": round(float(get_cell(row, "overall_greenwashing_risk_score", 0)), 2),
        "max_greenwashing_risk": round(float(get_cell(row, "max_greenwashing_risk", 0)), 2),
        "high_risk_ratio": round(float(get_cell(row, "high_risk_ratio", 0)), 2),
        "medium_risk_ratio": round(float(get_cell(row, "medium_risk_ratio", 0)), 2),
        "low_risk_ratio": round(float(get_cell(row, "low_risk_ratio", 0)), 2),
        "report_greenwashing_score": round(float(get_cell(row, "overall_greenwashing_risk_score", 0)), 2),
        "report_level": format_report_level(get_cell(row, "overall_risk_level", "")),
        "summary_reason": get_cell(row, "report_analysis", ""),
        "top_risk_text": top_chunk.get("text", ""),
        "top_risk_score": round(float(top_chunk.get("greenwashing_risk", 0)), 2),
        "top_risk_level": top_chunk.get("risk_level", "-"),
        "top_risk_explanation": top_chunk.get("explanation", ""),
    }

    report_name = get_cell(row, "report_name", "")
    year = int(get_cell(row, "year", 0))
    company = get_cell(row, "company", "")

    return {
        "filename": f"{company}｜{year}｜{report_name}",
        "summary": summary,
        "top_chunks": top_chunks,
    }


def find_greenwashing_report(selection):
    if not selection or "|" not in selection:
        return None

    company, year_text = selection.split("|", 1)
    try:
        year = int(year_text)
    except ValueError:
        return None

    dataset = load_greenwashing_dataset()
    if dataset.empty:
        return None

    matched = dataset.loc[
        (dataset["company"] == company)
        & (dataset["year"] == year)
    ]

    if matched.empty:
        return None

    return build_greenwashing_result_from_row(matched.iloc[0])


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
# 3. ML 預測解釋
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


@app.route("/usage")
def usage():
    """
    網頁使用說明頁面。
    """
    return render_template("usage.html")


@app.route("/about")
def about():
    """
    關於我們頁面。
    """
    return render_template("about.html")


@app.route("/links")
def links():
    """
    相關連結頁面。
    """
    return render_template("links.html")


@app.route("/greenwashing", methods=["GET", "POST"])
def greenwashing():
    """
    永續報告書漂綠偵測頁面。
    GET：顯示 data_2 已批次完成的報告選項
    POST：依公司年度查詢既有結果，不在網站端重新跑 NLP 模型
    """

    result = None
    error = None
    selected_report = None
    report_options = get_greenwashing_options()

    if request.method == "POST":
        selected_report = request.form.get("report_key")

        if not selected_report:
            error = "請先選擇一份已分析的永續報告書。"
        else:
            result = find_greenwashing_report(selected_report)
            if result is None:
                error = "找不到這份報告的固定分析結果，請確認 data_2 資料是否完整。"

    return render_template(
        "greenwashing.html",
        result=result,
        error=error,
        report_options=report_options,
        selected_report=selected_report
    )


@app.route("/dashboard")
def dashboard():
    """
    Colab 成果儀表板。
    直接讀取匯出的 CSV，展示模型表現、特徵重要度與描述統計。
    """
    dashboard_data = build_dashboard_data()
    return render_template("dashboard.html", dashboard=dashboard_data)


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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
