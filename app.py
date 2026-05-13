from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
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

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"pdf"}

# 自動建立資料夾
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ML 模型路徑
ROA_MODEL_PATH = os.path.join(MODEL_FOLDER, "roa_prediction_model.joblib")
FEATURES_PATHS = [
    os.path.join(MODEL_FOLDER, "model_features.pkl"),
    os.path.join(MODEL_FOLDER, "model_features.joblib"),
]
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

def explain_roa_prediction(predicted_roa, user_input):
    """
    根據預測 ROA 給出簡單解釋。
    """

    reasons = []

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

    if esg >= 70:
        reasons.append("ESG 總分較高，代表公司在整體永續表現上具有一定基礎。")
    elif esg < 50:
        reasons.append("ESG 總分偏低，可能代表永續表現仍有改善空間。")

    if e_score >= 70:
        reasons.append("環境構面分數較高，可能對永續形象與風險控管有正面幫助。")

    if s_score >= 70:
        reasons.append("社會構面分數較高，表示公司在社會責任或利害關係人管理上表現較佳。")

    if g_score >= 70:
        reasons.append("公司治理分數較高，可能代表治理制度與管理透明度較佳。")
    elif g_score < 50:
        reasons.append("公司治理分數偏低，可能增加營運與治理風險。")

    if roa_lag >= 2:
        reasons.append("前一期 ROA 較高，代表公司過去獲利能力較佳。")
    elif roa_lag < 0.5:
        reasons.append("前一期 ROA 偏低，可能影響模型對未來績效的預測。")

    if firm_size > 100000000:
        reasons.append("公司規模較大，模型會將其視為影響 ROA 預測的重要因素之一。")

    if not reasons:
        reasons.append("模型主要根據 ESG 分數、E/S/G 構面、公司規模與前一期 ROA 綜合判斷。")

    return {
        "level": level,
        "conclusion": conclusion,
        "reasons": reasons
    }


# =========================
# 7. Flask Routes
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


@app.route("/linear-esg")
def linear_esg():
    """
    線性 ESG 預測頁面。
    目前先放模板頁。
    """
    return render_template("linear_esg.html")


@app.route("/ml-prediction", methods=["GET", "POST"])
def ml_prediction():
    """
    機器學習預測頁面。
    使用已訓練好的模型預測未來 ROA。
    """

    result = None
    error = None

    if request.method == "POST":
        try:
            model, features = load_roa_prediction_model()

            user_input = {
                "TESG分數": float(request.form.get("tesg_score", 0)),
                "環境構面分數": float(request.form.get("env_score", 0)),
                "社會構面分數": float(request.form.get("social_score", 0)),
                "公司治理構面分數": float(request.form.get("gov_score", 0)),
                "市值_avg": float(request.form.get("market_value", 0)),
                "ROA_lag": float(request.form.get("roa_lag", 0))
            }

            input_df = pd.DataFrame([user_input])

            # 確保欄位與訓練時一致
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            input_df = input_df[features]

            prediction = model.predict(input_df)
            predicted_roa = float(prediction[0])

            explanation = explain_roa_prediction(predicted_roa, user_input)

            result = {
                "predicted_roa": round(predicted_roa, 4),
                "level": explanation["level"],
                "conclusion": explanation["conclusion"],
                "reasons": explanation["reasons"],
                "features": user_input,
                "model_features": features
            }

        except Exception as e:
            error = f"預測失敗：{str(e)}"

    return render_template("ml_prediction.html", result=result, error=error)


# =========================
# 8. 啟動 Flask
# =========================

if __name__ == "__main__":
    app.run(debug=True)
