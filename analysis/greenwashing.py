"""Offline greenwashing analysis adapted from the supplied Colab notebook.

Model scoring requires optional NLP dependencies. Importing does not load/download models.
Scores are screening heuristics, not verified findings of misconduct.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


CONFIG = {
    # 若報告多為中文，可優先用 multilingual NLI 模型
    # T4 可跑，但第一次下載較久。
    "zero_shot_model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",

    # 每段最大長度，太長會截斷；ESG 報告建議用段落或句群，不要整本一次丟模型
    "max_chars_per_chunk": 900,
    "min_chars_per_chunk": 80,

    # zero-shot 批次大小；T4 若 OOM 可改成 4 或 2
    "batch_size": 8,

    # 輸出分數權重，可依作業或研究設計調整
    "risk_weights": {
        "high_commitment_low_evidence": 0.45,
        "vagueness": 0.25,
        "low_verifiability": 0.20,
        "promotional_tone": 0.10,
    },

    # 若你想只看 ESG 相關段落，可以打開；初版建議 False，避免漏掉重要段落
    "filter_esg_related_only": False,
    "esg_filter_threshold": 0.45,
}

ESG_KEYWORDS = {
    "E": [
        "碳", "溫室氣體", "排放", "減碳", "淨零", "再生能源", "能源", "用水", "廢棄物",
        "污染", "環境", "氣候", "生物多樣性", "循環經濟", "節能", "Scope 1", "Scope 2", "Scope 3",
        "carbon", "emission", "net zero", "renewable", "energy", "climate", "waste", "water", "biodiversity"
    ],
    "S": [
        "員工", "職安", "安全", "健康", "人權", "多元", "平等", "薪酬", "教育訓練", "供應商",
        "社區", "客戶", "隱私", "labor", "employee", "safety", "human rights", "diversity", "training", "community"
    ],
    "G": [
        "董事會", "治理", "法遵", "稽核", "風險管理", "內部控制", "倫理", "反貪腐", "資訊安全",
        "governance", "board", "compliance", "audit", "risk management", "ethics", "anti-corruption", "cybersecurity"
    ]
}

COMMITMENT_KEYWORDS = [
    "承諾", "目標", "預計", "計畫", "將", "推動", "致力", "達成", "降低", "提升", "改善", "減少",
    "commit", "target", "goal", "aim", "plan", "will", "pledge", "roadmap", "reduce", "increase", "improve"
]

EVIDENCE_KEYWORDS = [
    "達成率", "數據", "指標", "KPI", "統計", "查證", "第三方", "驗證", "確信", "盤查", "ISO", "GRI", "SASB", "TCFD",
    "assurance", "verified", "audit", "metric", "data", "KPI", "certified", "inventory", "baseline"
]

VAGUE_KEYWORDS = [
    "持續", "積極", "努力", "致力於", "重視", "強化", "落實", "逐步", "完善", "優化", "打造", "邁向",
    "continue", "actively", "strive", "endeavor", "enhance", "strengthen", "promote", "ongoing", "world-class", "leading"
]

PROMOTIONAL_KEYWORDS = [
    "卓越", "領先", "最佳", "典範", "榮耀", "優質", "創新", "永續典範", "標竿",
    "excellent", "leading", "best", "world-class", "outstanding", "benchmark", "innovative"
]

YEAR_PATTERN = r"(?:19|20)\d{2}|民國\s*\d{2,3}\s*年"

NUMBER_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|％|噸|公噸|tCO2e|CO2e|度|kWh|MWh|GWh|元|人|件|次|小時|hours|tons?|tonnes?|kg|MW|GW)?"

def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([。！？!?；;])", r"\1\n", text)
    return text.strip()

def count_keywords(text: str, keywords: List[str]) -> int:
    lower = text.lower()
    return sum(lower.count(k.lower()) for k in keywords)

def keyword_ratio(text: str, keywords: List[str]) -> float:
    words_like = max(len(re.findall(r"\w+|[\u4e00-\u9fff]", text)), 1)
    return count_keywords(text, keywords) / words_like

def extract_numeric_features(text: str) -> Dict[str, float]:
    numbers = re.findall(NUMBER_PATTERN, text, flags=re.IGNORECASE)
    years = re.findall(YEAR_PATTERN, text, flags=re.IGNORECASE)
    has_percent = bool(re.search(r"\d\s*(%|％)", text))
    has_unit = bool(re.search(r"\d\s*(噸|公噸|tCO2e|CO2e|kWh|MWh|GWh|人|件|次|小時|tons?|tonnes?|kg|MW|GW)", text, flags=re.IGNORECASE))
    token_count = max(len(re.findall(r"\w+|[\u4e00-\u9fff]", text)), 1)
    return {
        "number_count": len(numbers),
        "year_count": len(years),
        "number_ratio": len(numbers) / token_count,
        "has_percent": int(has_percent),
        "has_unit": int(has_unit),
        "has_year": int(len(years) > 0),
    }

def compute_rule_features(chunks: List[str]) -> pd.DataFrame:
    rows = []
    all_esg_keywords = ESG_KEYWORDS["E"] + ESG_KEYWORDS["S"] + ESG_KEYWORDS["G"]

    for i, text in enumerate(chunks):
        num = extract_numeric_features(text)
        row = {
            "chunk_id": i,
            "text": text,
            "char_len": len(text),
            "commitment_kw_count": count_keywords(text, COMMITMENT_KEYWORDS),
            "evidence_kw_count": count_keywords(text, EVIDENCE_KEYWORDS),
            "vague_kw_count": count_keywords(text, VAGUE_KEYWORDS),
            "promotional_kw_count": count_keywords(text, PROMOTIONAL_KEYWORDS),
            "esg_kw_count": count_keywords(text, all_esg_keywords),
            "E_kw_count": count_keywords(text, ESG_KEYWORDS["E"]),
            "S_kw_count": count_keywords(text, ESG_KEYWORDS["S"]),
            "G_kw_count": count_keywords(text, ESG_KEYWORDS["G"]),
            "commitment_kw_ratio": keyword_ratio(text, COMMITMENT_KEYWORDS),
            "evidence_kw_ratio": keyword_ratio(text, EVIDENCE_KEYWORDS),
            "vague_kw_ratio": keyword_ratio(text, VAGUE_KEYWORDS),
            "promotional_kw_ratio": keyword_ratio(text, PROMOTIONAL_KEYWORDS),
            "esg_kw_ratio": keyword_ratio(text, all_esg_keywords),
            **num,
        }
        rows.append(row)

    return pd.DataFrame(rows)

def add_tfidf_features(df: pd.DataFrame, max_features: int = 500) -> pd.DataFrame:
    """TF-IDF 不直接全部丟進語言模型；這裡先萃取 top terms 與平均 TF-IDF 強度作為輔助特徵。"""
    if len(df) == 0:
        return df

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b|[\u4e00-\u9fff]",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["text"].tolist())
    vocab = np.array(vectorizer.get_feature_names_out())

    mean_tfidf = np.asarray(X.mean(axis=1)).ravel()
    max_tfidf = np.asarray(X.max(axis=1).toarray()).ravel()

    top_terms = []
    for row_idx in range(X.shape[0]):
        row = X[row_idx].toarray().ravel()
        top_idx = row.argsort()[-8:][::-1]
        terms = [vocab[j] for j in top_idx if row[j] > 0]
        top_terms.append(", ".join(terms))

    df = df.copy()
    df["tfidf_mean"] = mean_tfidf
    df["tfidf_max"] = max_tfidf
    df["tfidf_top_terms"] = top_terms
    return df

LABELS = [
    "明確的 ESG 承諾或目標",
    "具體量化證據或績效數據",
    "可驗證且可信的 ESG 說明",
    "模糊宣傳或漂綠風險"
]

def zero_shot_scores(texts: List[str], clf, batch_size: int = 8) -> pd.DataFrame:
    from tqdm.auto import tqdm
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot scoring"):
        batch = texts[i:i + batch_size]
        outs = clf(
            batch,
            candidate_labels=LABELS,
            hypothesis_template="這段文字屬於{}。",
            multi_label=True,
            truncation=True,
        )
        if isinstance(outs, dict):
            outs = [outs]
        for out in outs:
            score_map = {label: score for label, score in zip(out["labels"], out["scores"])}
            results.append({
                "llm_commitment": score_map.get("明確的 ESG 承諾或目標", 0.0),
                "llm_evidence": score_map.get("具體量化證據或績效數據", 0.0),
                "llm_credibility": score_map.get("可驗證且可信的 ESG 說明", 0.0),
                "llm_greenwash": score_map.get("模糊宣傳或漂綠風險", 0.0),
            })
    return pd.DataFrame(results)

def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def compute_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 規則分數：0-1
    commitment_rule = np.clip(
        0.55 * minmax(df["commitment_kw_count"]) +
        0.25 * df["has_year"] +
        0.20 * minmax(df["esg_kw_count"]),
        0, 1
    )

    evidence_rule = np.clip(
        0.45 * minmax(df["number_count"]) +
        0.25 * df["has_unit"] +
        0.15 * df["has_percent"] +
        0.15 * minmax(df["evidence_kw_count"]),
        0, 1
    )

    credibility_rule = np.clip(
        0.35 * evidence_rule +
        0.25 * df["has_year"] +
        0.20 * minmax(df["evidence_kw_count"]) +
        0.20 * (1 - minmax(df["vague_kw_count"])),
        0, 1
    )

    vagueness_rule = np.clip(
        0.65 * minmax(df["vague_kw_count"]) +
        0.35 * minmax(df["promotional_kw_count"]),
        0, 1
    )

    df["commitment_rule"] = commitment_rule
    df["evidence_rule"] = evidence_rule
    df["credibility_rule"] = credibility_rule
    df["vagueness_rule"] = vagueness_rule

    # 最終分數：語言模型 + 規則特徵
    df["Commitment Score"] = np.clip(100 * (0.65 * df["llm_commitment"] + 0.35 * df["commitment_rule"]), 0, 100)
    df["Evidence Score"] = np.clip(100 * (0.60 * df["llm_evidence"] + 0.40 * df["evidence_rule"]), 0, 100)
    df["Credibility Score"] = np.clip(100 * (0.60 * df["llm_credibility"] + 0.40 * df["credibility_rule"]), 0, 100)

    # 漂綠風險核心邏輯：承諾高，但證據與可信度低，風險較高
    high_commit_low_evidence = (df["Commitment Score"] / 100) * (1 - df["Evidence Score"] / 100)
    low_verifiability = 1 - df["Credibility Score"] / 100
    promotional_tone = minmax(df["promotional_kw_count"])

    w = CONFIG["risk_weights"]
    risk = (
        w["high_commitment_low_evidence"] * high_commit_low_evidence +
        w["vagueness"] * df["vagueness_rule"] +
        w["low_verifiability"] * low_verifiability +
        w["promotional_tone"] * promotional_tone
    )

    # 加入模型直接判斷，但不要讓它完全主導
    risk = 0.70 * risk + 0.30 * df["llm_greenwash"]
    df["Greenwashing Risk"] = np.clip(100 * risk, 0, 100)

    # 風險等級
    def risk_level(x):
        if x >= 70:
            return "High"
        elif x >= 40:
            return "Medium"
        else:
            return "Low"

    df["Risk Level"] = df["Greenwashing Risk"].apply(risk_level)
    return df

def explain_row(row: pd.Series) -> str:
    strengths = []
    risks = []

    # 正面證據
    if row["Commitment Score"] >= 60:
        strengths.append("有明顯 ESG 承諾或目標語氣")

    if row["number_count"] > 0 or row["has_percent"] == 1 or row["has_unit"] == 1:
        strengths.append("有出現量化數字、百分比或單位")

    if row["has_year"] == 1:
        strengths.append("有提到年份或時間點")

    if row["Credibility Score"] >= 60:
        strengths.append("整體文字相對具體、較可驗證")

    # 風險訊號
    if row["vague_kw_count"] > 0 or row["promotional_kw_count"] > 0:
        risks.append("同時含有模糊或宣傳性用語")

    if row["Commitment Score"] >= 60 and row["number_count"] == 0 and row["has_percent"] == 0 and row["has_unit"] == 0:
        risks.append("有承諾但缺少量化數據支持")

    if row["Commitment Score"] >= 60 and row["has_year"] == 0:
        risks.append("有承諾但缺少年份或期限")

    if row["Evidence Score"] < 40 and row["Commitment Score"] >= 60:
        risks.append("承諾程度高，但證據支持分數偏低")

    if row["Credibility Score"] < 40 and row["Commitment Score"] >= 60:
        risks.append("承諾程度高，但可信度分數偏低")

    # 組合輸出
    if strengths and risks:
        return "正面訊號：" + "、".join(strengths) + "。風險訊號：" + "、".join(risks) + "。"
    elif strengths:
        return "正面訊號：" + "、".join(strengths) + "。"
    elif risks:
        return "風險訊號：" + "、".join(risks) + "。"
    else:
        return "未偵測到明顯 ESG 承諾、量化證據或高風險訊號。"

def analyze_report(
    input_path: str,
    output_prefix: str = "esg_analysis",
    clf=None,
    save_outputs: bool = False
) -> pd.DataFrame:
    """
    分析單份永續報告。

    參數：
    - input_path: PDF / TXT / MD 檔案路徑
    - output_prefix: 若 save_outputs=True，輸出檔案前綴
    - clf: 已載入的 zero-shot pipeline。
           批次分析時建議外部傳入，避免每份報告重複載入模型。
    - save_outputs:
        False = 只回傳分析結果，不輸出 chunks.csv / chunks.xlsx / summary.json
        True = 輸出單份報告的詳細分析檔案
    """

    raw_text = load_report(input_path)

    chunks = split_into_chunks(
        raw_text,
        max_chars=CONFIG["max_chars_per_chunk"],
        min_chars=CONFIG["min_chars_per_chunk"]
    )

    print(f"Extracted chunks: {len(chunks)}")

    if len(chunks) < 10:
        print("提醒：抽出的段落數偏少，可能是掃描檔或 PDF 文字抽取不完整，建議人工檢查。")

    if len(chunks) == 0:
        raise ValueError("沒有切出可分析段落。可能是掃描 PDF，需要 OCR。")

    df = compute_rule_features(chunks)
    df = add_tfidf_features(df)

    if CONFIG["filter_esg_related_only"]:
        df = df[df["esg_kw_ratio"] >= CONFIG["esg_filter_threshold"]].reset_index(drop=True)
        print(f"After ESG filter: {len(df)} chunks")

    if df.empty:
        raise ValueError("No chunks remain after ESG filtering")

    # 如果外部沒有傳入模型，才載入模型
    if clf is None:
        clf = load_zero_shot_pipeline()
    else:
        print("Using preloaded zero-shot model.")

    llm_df = zero_shot_scores(
        df["text"].tolist(),
        clf,
        batch_size=CONFIG["batch_size"]
    )

    df = pd.concat([df.reset_index(drop=True), llm_df.reset_index(drop=True)], axis=1)

    df = compute_final_scores(df)
    df["Explanation"] = df.apply(explain_row, axis=1)

    # 依漂綠風險排序，方便後續取前 5 高風險段落
    df_sorted = df.sort_values("Greenwashing Risk", ascending=False).reset_index(drop=True)

    # 只有 save_outputs=True 時，才輸出單份報告的詳細檔案
    if save_outputs:
        csv_path = f"{output_prefix}_chunks.csv"
        xlsx_path = f"{output_prefix}_chunks.xlsx"
        summary_path = f"{output_prefix}_summary.json"

        df_sorted.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df_sorted.to_excel(xlsx_path, index=False)

        summary = {
            "input_path": input_path,
            "n_chunks": int(len(df_sorted)),
            "avg_commitment_score": float(df_sorted["Commitment Score"].mean()),
            "avg_evidence_score": float(df_sorted["Evidence Score"].mean()),
            "avg_credibility_score": float(df_sorted["Credibility Score"].mean()),
            "avg_greenwashing_risk": float(df_sorted["Greenwashing Risk"].mean()),
            "risk_level_counts": df_sorted["Risk Level"].value_counts().to_dict(),
            "top_risk_chunks": df_sorted.head(5)[[
                "chunk_id",
                "Greenwashing Risk",
                "Risk Level",
                "Commitment Score",
                "Evidence Score",
                "Credibility Score",
                "Explanation",
                "text"
            ]].to_dict(orient="records")
        }

        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print("Saved:")
        print(" -", csv_path)
        print(" -", xlsx_path)
        print(" -", summary_path)

    return df_sorted

def extract_report_year_from_filename(file_path) -> int | None:
    """
    從檔名中抓報告年份。
    支援：
    - 西元年：2014、2023、2024
    - 民國年：民國112年、112年度、112年CSR報告書
    """
    filename = Path(file_path).stem

    # Ambiguous filenames must be corrected before joining reports to panel data.
    years = set(re.findall(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", filename))
    if len(years) > 1:
        return None
    if years:
        return int(years.pop())

    # 再抓民國年，例如 民國112年、112年度
    roc_match = re.search(r"(?:民國)?\s*(\d{2,3})\s*(?:年|年度)", filename)
    if roc_match:
        roc_year = int(roc_match.group(1))
        if roc_year >= 80:
            return roc_year + 1911

    return None

def safe_filename(text: str) -> str:
    """
    清理檔名，避免 Windows / macOS 不適合的特殊符號。
    """
    text = str(text).strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text if text else "company"

def safe_float(x, digits: int = 2):
    """
    將數值安全轉成 float 並四捨五入；若無法轉換則回傳 None。
    """
    try:
        if pd.isna(x):
            return None
        return round(float(x), digits)
    except Exception:
        return None

def get_overall_risk_level(score: float | int | None) -> str:
    """
    報告層級的整體風險等級。

    注意：這是「整份報告平均風險」的分級，
    因為平均值通常會比單一段落風險低，所以門檻比段落層級略低。
    """
    if score is None or pd.isna(score):
        return "Unknown"
    score = float(score)
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"

def generate_report_analysis(summary: dict) -> str:
    """
    依據報告層級指標，產生整篇報告的分析敘述。
    不使用額外 LLM，採規則式模板，確保輸出穩定且零成本。
    """
    risk_score = summary.get("overall_greenwashing_risk_score")
    risk_level = summary.get("overall_risk_level")
    avg_commitment = summary.get("avg_commitment_score")
    avg_evidence = summary.get("avg_evidence_score")
    avg_credibility = summary.get("avg_credibility_score")
    high_chunks = summary.get("high_risk_chunks") or 0
    high_ratio = summary.get("high_risk_ratio") or 0
    n_chunks = summary.get("n_chunks") or 0

    parts = []

    if risk_level == "High":
        parts.append("整體而言，這份報告具有較高程度的潛在漂綠風險。")
    elif risk_level == "Medium":
        parts.append("整體而言，這份報告具有中等程度的潛在漂綠風險。")
    elif risk_level == "Low":
        parts.append("整體而言，這份報告的潛在漂綠風險相對較低。")
    else:
        parts.append("整體而言，這份報告的漂綠風險等級無法判定。")

    if risk_score is not None:
        parts.append(f"整體漂綠風險分數為 {safe_float(risk_score)} 分。")

    if avg_commitment is not None and avg_evidence is not None:
        diff_ce = float(avg_commitment) - float(avg_evidence)
        if diff_ce >= 20:
            parts.append("報告中的平均承諾分數明顯高於平均證據分數，表示文本較強調永續願景、目標或承諾，但具體量化支持相對不足。")
        elif diff_ce >= 10:
            parts.append("報告中的平均承諾分數高於平均證據分數，表示永續目標與願景敘述較多，仍建議檢視其量化證據是否充分。")
        elif diff_ce <= -10:
            parts.append("報告中的平均證據分數高於平均承諾分數，表示文本中相對較多具體數據或成果資訊。")
        else:
            parts.append("報告中的平均承諾分數與平均證據分數差距不大，代表永續目標敘述與具體支持程度相對接近。")

    if avg_credibility is not None:
        if float(avg_credibility) < 40:
            parts.append("平均可信度分數偏低，部分 ESG 敘述可能缺乏具體時間、方法、量化依據或可檢查資訊。")
        elif float(avg_credibility) < 60:
            parts.append("平均可信度分數屬於中等，部分段落具備一定資訊基礎，但仍有進一步補強揭露品質的空間。")
        else:
            parts.append("平均可信度分數較高，代表多數 ESG 敘述相對具體且較容易被檢查。")

    if n_chunks > 0:
        if high_chunks >= 1:
            parts.append(f"報告中共有 {high_chunks} 個段落被判定為高漂綠風險，建議進一步人工檢視前幾個高風險段落。")
        else:
            parts.append("報告中沒有段落被判定為高漂綠風險。")

        if high_ratio >= 0.10:
            parts.append("高風險段落占比達到需要注意的程度。")
        elif high_ratio > 0:
            parts.append("高風險段落占比不高，但仍應檢視其具體內容。")

    return "".join(parts)

def generate_paragraph_analysis(row: pd.Series) -> str:
    """
    依據單一段落的分數與特徵，產生可直接放到網頁 demo 的段落分析敘述。
    """
    risk = row.get("Greenwashing Risk", np.nan)
    risk_level = row.get("Risk Level", "Unknown")
    commitment = row.get("Commitment Score", np.nan)
    evidence = row.get("Evidence Score", np.nan)
    credibility = row.get("Credibility Score", np.nan)
    explanation = row.get("Explanation", "")

    parts = []

    # 風險總述
    if risk_level == "High":
        parts.append(f"此段落被判定為高漂綠風險，風險分數為 {safe_float(risk)}。")
    elif risk_level == "Medium":
        parts.append(f"此段落被判定為中等漂綠風險，風險分數為 {safe_float(risk)}。")
    elif risk_level == "Low":
        parts.append(f"此段落被判定為低漂綠風險，風險分數為 {safe_float(risk)}。")
    else:
        parts.append(f"此段落風險等級無法判定，風險分數為 {safe_float(risk)}。")

    # 承諾程度
    if pd.notna(commitment):
        if commitment >= 70:
            parts.append("段落具有較強的永續承諾、目標或願景敘述。")
        elif commitment >= 40:
            parts.append("段落具有一定程度的永續承諾或目標語氣。")
        else:
            parts.append("段落中的永續承諾或目標語氣相對較弱。")

    # 證據支持
    if pd.notna(evidence):
        if evidence < 40:
            parts.append("具體量化證據分數偏低，可能缺少明確數據、百分比、年份、單位或成果指標。")
        elif evidence < 60:
            parts.append("具體量化證據分數屬於中等，段落可能有部分數據或成果資訊，但仍不算充分。")
        else:
            parts.append("具體量化證據分數較高，段落中包含較多數據、指標或具體成果。")

    # 可信度
    if pd.notna(credibility):
        if credibility < 40:
            parts.append("可信度分數偏低，表示段落可能較缺乏可檢查資訊或第三方驗證訊號。")
        elif credibility < 60:
            parts.append("可信度分數屬於中等，段落有部分具體資訊，但仍有補強空間。")
        else:
            parts.append("可信度分數較高，段落敘述相對具體且較容易被檢查。")

    # 原本 Explanation 補充
    if isinstance(explanation, str) and explanation.strip():
        parts.append(f"補充判斷：{explanation.strip()}")

    return "".join(parts)

def add_top_risk_columns(summary: dict, df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    將 df 中漂綠風險最高的前 top_n 段落展開成 summary 的欄位。
    維持 CSV 一份報告一列的結構。
    """
    top_df = df.sort_values("Greenwashing Risk", ascending=False).head(top_n).reset_index(drop=True)

    for i in range(1, top_n + 1):
        if i <= len(top_df):
            row = top_df.iloc[i - 1]
            summary[f"top{i}_chunk_id"] = int(row["chunk_id"]) if pd.notna(row.get("chunk_id")) else None
            summary[f"top{i}_risk_score"] = safe_float(row.get("Greenwashing Risk"))
            summary[f"top{i}_risk_level"] = row.get("Risk Level")
            summary[f"top{i}_commitment_score"] = safe_float(row.get("Commitment Score"))
            summary[f"top{i}_evidence_score"] = safe_float(row.get("Evidence Score"))
            summary[f"top{i}_credibility_score"] = safe_float(row.get("Credibility Score"))
            summary[f"top{i}_analysis"] = generate_paragraph_analysis(row)
            summary[f"top{i}_text"] = row.get("text")
        else:
            summary[f"top{i}_chunk_id"] = None
            summary[f"top{i}_risk_score"] = None
            summary[f"top{i}_risk_level"] = None
            summary[f"top{i}_commitment_score"] = None
            summary[f"top{i}_evidence_score"] = None
            summary[f"top{i}_credibility_score"] = None
            summary[f"top{i}_analysis"] = None
            summary[f"top{i}_text"] = None

    return summary

def add_empty_top_risk_columns(summary: dict, top_n: int = 5) -> dict:
    """
    分析失敗時，也補齊 top1 ~ top5 欄位，避免 CSV 欄位不一致。
    """
    for i in range(1, top_n + 1):
        summary[f"top{i}_chunk_id"] = None
        summary[f"top{i}_risk_score"] = None
        summary[f"top{i}_risk_level"] = None
        summary[f"top{i}_commitment_score"] = None
        summary[f"top{i}_evidence_score"] = None
        summary[f"top{i}_credibility_score"] = None
        summary[f"top{i}_analysis"] = None
        summary[f"top{i}_text"] = None
    return summary

def export_human_validation_file(
    df: pd.DataFrame,
    output_path: str = "human_validation_sample.xlsx",
    n_per_level: int = 30,
    random_state: int = 42,
    include_model_scores: bool = True,
    blind_review: bool = False,
) -> pd.DataFrame:
    """
    從模型輸出結果中，依 High / Medium / Low 風險各抽 n_per_level 筆，
    輸出給人工標註用的 Excel。

    人工標註欄位：
    - Human Commitment: 0/1
    - Human Evidence: 0/1
    - Human Credibility: 0/1
    - Human Risk Level: 1/2/3
        1 = Low 低風險
        2 = Medium 中風險
        3 = High 高風險
    """

    df = df.copy()

    if "Risk Level" not in df.columns:
        raise ValueError("df 裡找不到 'Risk Level' 欄位，請確認你傳入的是 analyze_report() 的輸出結果。")

    samples = []

    for level in ["High", "Medium", "Low"]:
        sub = df[df["Risk Level"] == level].copy()

        if len(sub) == 0:
            print(f"提醒：沒有 {level} 風險段落可抽樣。")
            continue

        sample_n = min(n_per_level, len(sub))
        sampled = sub.sample(n=sample_n, random_state=random_state)
        samples.append(sampled)

        print(f"{level}: 抽出 {sample_n} 筆 / 原本 {len(sub)} 筆")

    if len(samples) == 0:
        raise ValueError("沒有抽出任何樣本，請檢查 Risk Level 欄位內容。")

    validation_df = pd.concat(samples, ignore_index=True)

    # 打亂順序，避免人工標註者知道前面都是 High、中間都是 Medium
    validation_df = validation_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 建立人工標註欄位
    validation_df.insert(0, "sample_id", range(1, len(validation_df) + 1))
    validation_df["Human Commitment"] = ""       # 0/1：是否有明確 ESG 承諾或目標
    validation_df["Human Evidence"] = ""         # 0/1：是否有量化數據或具體證據
    validation_df["Human Credibility"] = ""      # 0/1：是否具體、可驗證
    validation_df["Human Risk Level"] = ""       # 1/2/3：1=Low，2=Medium，3=High
    validation_df["Human Note"] = ""             # 人工備註

    base_cols = [
        "sample_id",
        "chunk_id",
        "text",
        "Human Commitment",
        "Human Evidence",
        "Human Credibility",
        "Human Risk Level",
        "Human Note",
    ]

    model_cols = [
        "Risk Level",
        "Greenwashing Risk",
        "Commitment Score",
        "Evidence Score",
        "Credibility Score",
        "Explanation",
        "number_count",
        "year_count",
        "has_percent",
        "has_unit",
        "has_year",
        "commitment_kw_count",
        "evidence_kw_count",
        "vague_kw_count",
        "promotional_kw_count",
        "tfidf_top_terms",
    ]

    if blind_review:
        final_cols = base_cols
    else:
        if include_model_scores:
            final_cols = base_cols + [c for c in model_cols if c in validation_df.columns]
        else:
            final_cols = base_cols + ["Explanation"] if "Explanation" in validation_df.columns else base_cols

    validation_df = validation_df[[c for c in final_cols if c in validation_df.columns]]

    instruction_df = pd.DataFrame({
        "欄位": [
            "Human Commitment",
            "Human Evidence",
            "Human Credibility",
            "Human Risk Level",
            "Human Note",
        ],
        "填寫方式": [
            "填 1 或 0。1 = 段落有明確 ESG 承諾、目標、改善方向或期限；0 = 沒有。",
            "填 1 或 0。1 = 段落有數字、百分比、單位、KPI、第三方查證或具體成果；0 = 沒有。",
            "填 1 或 0。1 = 段落有具體時間、範圍、方法、基準年、查證機制，足以被驗證；0 = 模糊或難以驗證。",
            "填 1、2 或 3。1 = Low 低風險；2 = Medium 中風險；3 = High 高風險。",
            "可寫人工判斷理由，例如：有 2030 目標但缺少基準年、只有口號沒有數據等。",
        ]
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        validation_df.to_excel(writer, sheet_name="human_labeling", index=False)
        instruction_df.to_excel(writer, sheet_name="labeling_instruction", index=False)

        workbook = writer.book
        ws = workbook["human_labeling"]
        ws.freeze_panes = "A2"

        # 調整欄寬
        for col in ws.columns:
            col_letter = col[0].column_letter
            header = col[0].value

            if header == "text":
                ws.column_dimensions[col_letter].width = 80
            elif header in ["Explanation", "Human Note"]:
                ws.column_dimensions[col_letter].width = 45
            else:
                ws.column_dimensions[col_letter].width = 18

        # 加上 Excel 下拉選單，避免人工標註打錯
        from openpyxl.worksheet.datavalidation import DataValidation

        max_row = len(validation_df) + 1

        binary_validation = DataValidation(
            type="list",
            formula1='"0,1"',
            allow_blank=True
        )

        risk_validation = DataValidation(
            type="list",
            formula1='"1,2,3"',
            allow_blank=True
        )

        ws.add_data_validation(binary_validation)
        ws.add_data_validation(risk_validation)

        # 找欄位位置
        header_to_col = {cell.value: cell.column_letter for cell in ws[1]}

        for col_name in ["Human Commitment", "Human Evidence", "Human Credibility"]:
            if col_name in header_to_col:
                col_letter = header_to_col[col_name]
                binary_validation.add(f"{col_letter}2:{col_letter}{max_row}")

        if "Human Risk Level" in header_to_col:
            col_letter = header_to_col["Human Risk Level"]
            risk_validation.add(f"{col_letter}2:{col_letter}{max_row}")

        ws2 = workbook["labeling_instruction"]
        ws2.column_dimensions["A"].width = 25
        ws2.column_dimensions["B"].width = 100

    print(f"已輸出人工標註檔案：{output_path}")
    print("Human Risk Level 填寫規則：1 = Low，2 = Medium，3 = High")

    return validation_df

def evaluate_human_validation(
    labeled_file: str = "human_validation_sample_labeled.xlsx",
    commitment_threshold: float = 60,
    evidence_threshold: float = 60,
    credibility_threshold: float = 60,
    risk_threshold_low: float = 40,
    risk_threshold_high: float = 70,
) -> pd.DataFrame:
    """
    讀取人工標註完成的 Excel，計算 Accuracy / Precision / Recall / F1。

    人工標註格式：
    - Human Commitment: 0/1
    - Human Evidence: 0/1
    - Human Credibility: 0/1
    - Human Risk Level: 1/2/3
        1 = Low
        2 = Medium
        3 = High

    注意：
    這個檔案需要保留模型分數欄位，
    所以 blind_review=True 的檔案不能直接用這個函數。
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )

    df = pd.read_excel(labeled_file, sheet_name="human_labeling")

    required_cols = [
        "Human Commitment",
        "Human Evidence",
        "Human Credibility",
        "Human Risk Level",
        "Commitment Score",
        "Evidence Score",
        "Credibility Score",
        "Greenwashing Risk",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少欄位：{missing}")

    # 人工標註欄位轉成數字
    df["Human Commitment"] = pd.to_numeric(df["Human Commitment"], errors="raise")
    df["Human Evidence"] = pd.to_numeric(df["Human Evidence"], errors="raise")
    df["Human Credibility"] = pd.to_numeric(df["Human Credibility"], errors="raise")
    df["Human Risk Level"] = pd.to_numeric(df["Human Risk Level"], errors="raise")

    for column in ["Human Commitment", "Human Evidence", "Human Credibility", "Human Risk Level"]:
        allowed = [1, 2, 3] if column == "Human Risk Level" else [0, 1]
        if not df[column].dropna().isin(allowed).all():
            raise ValueError(f"Invalid labels in {column}; expected {allowed}")
    for column in ["Commitment Score", "Evidence Score", "Credibility Score", "Greenwashing Risk"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or not df[column].between(0, 100).all():
            raise ValueError(f"Invalid model scores in {column}")

    # 模型分數轉成 0/1
    df["Pred Commitment"] = (df["Commitment Score"] >= commitment_threshold).astype(int)
    df["Pred Evidence"] = (df["Evidence Score"] >= evidence_threshold).astype(int)
    df["Pred Credibility"] = (df["Credibility Score"] >= credibility_threshold).astype(int)

    # 模型 Greenwashing Risk 轉成 1/2/3
    # 1 = Low, 2 = Medium, 3 = High
    def pred_risk_level(score):
        if score >= risk_threshold_high:
            return 3
        elif score >= risk_threshold_low:
            return 2
        else:
            return 1

    df["Pred Risk Level Numeric"] = df["Greenwashing Risk"].apply(pred_risk_level)

    # Commitment / Evidence / Credibility 三個二元任務
    for task in ["Commitment", "Evidence", "Credibility"]:
        y_true = df[f"Human {task}"]
        y_pred = df[f"Pred {task}"]

        valid = y_true.notna()

        print("=" * 60)
        print(task)
        print("有效標註數：", int(valid.sum()))

        if valid.sum() == 0:
            print("沒有有效標註，跳過。")
            continue

        print("Accuracy:", round(accuracy_score(y_true[valid], y_pred[valid]), 4))
        print("Precision:", round(precision_score(y_true[valid], y_pred[valid], zero_division=0), 4))
        print("Recall:", round(recall_score(y_true[valid], y_pred[valid], zero_division=0), 4))
        print("F1:", round(f1_score(y_true[valid], y_pred[valid], zero_division=0), 4))

        print("Confusion Matrix:")
        print(confusion_matrix(
            y_true[valid],
            y_pred[valid],
            labels=[0, 1]
        ))

    # Greenwashing Risk Level 三分類任務
    print("=" * 60)
    print("Greenwashing Risk Level")

    y_true = df["Human Risk Level"]
    y_pred = df["Pred Risk Level Numeric"]

    valid = y_true.notna()

    print("有效標註數：", int(valid.sum()))

    if valid.sum() == 0:
        print("沒有有效標註，跳過。")
    else:
        print(classification_report(
            y_true[valid],
            y_pred[valid],
            labels=[1, 2, 3],
            target_names=["Low(1)", "Medium(2)", "High(3)"],
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(
            y_true[valid],
            y_pred[valid],
            labels=[1, 2, 3]
        ))

    return df


def load_zero_shot_pipeline(model_name=CONFIG['zero_shot_model']):
    """Load lazily, only for an explicit offline analysis request."""
    import torch
    from transformers import pipeline
    return pipeline('zero-shot-classification', model=model_name, tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1)


def load_report(path, ocr=False):
    path = Path(path)
    if path.suffix.lower() in {'.txt', '.md'}:
        return path.read_text(encoding='utf-8')
    if path.suffix.lower() != '.pdf':
        raise ValueError('Supported reports: PDF, TXT, MD')
    import fitz
    texts = []
    with fitz.open(path) as document:
        for page in document:
            text = page.get_text()
            if ocr and len(text.strip()) < 80:
                text = page.get_text(textpage=page.get_textpage_ocr(language='chi_tra+eng', dpi=200))
            texts.append(text)
    return '\n'.join(texts)


# Override the notebook chunker: long sentences are bounded instead of truncated by the model.
def split_into_chunks(text, max_chars=900, min_chars=80):
    if not 0 < min_chars <= max_chars:
        raise ValueError('Require 0 < min_chars <= max_chars')
    sentences = [s.strip() for s in clean_text(text).splitlines() if s.strip()]
    pieces = [sentence[i:i + max_chars] for sentence in sentences
              for i in range(0, len(sentence), max_chars)]
    chunks, current = [], ''
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            if len(current) >= min_chars:
                chunks.append(current)
            current = ''
        current = f'{current} {piece}'.strip()
    if len(current) >= min_chars:
        chunks.append(current)
    return [c for c in chunks if not re.fullmatch(r'[\d\s\-.]+', c)]


def summarize_report(df, path, company):
    score = df['Greenwashing Risk'].mean()
    summary = dict(company=company, report_name=Path(path).stem,
                   year=extract_report_year_from_filename(path), n_chunks=len(df),
                   overall_greenwashing_risk_score=safe_float(score),
                   overall_risk_level=get_overall_risk_level(score),
                   avg_commitment_score=safe_float(df['Commitment Score'].mean()),
                   avg_evidence_score=safe_float(df['Evidence Score'].mean()),
                   avg_credibility_score=safe_float(df['Credibility Score'].mean()),
                   max_greenwashing_risk=safe_float(df['Greenwashing Risk'].max()),
                   top5_avg_greenwashing_risk=safe_float(df.nlargest(5, 'Greenwashing Risk')['Greenwashing Risk'].mean()),
                   error='')
    for level in ['High', 'Medium', 'Low']:
        summary[f'{level.lower()}_risk_chunks'] = int((df['Risk Level'] == level).sum())
        summary[f'{level.lower()}_risk_ratio'] = safe_float((df['Risk Level'] == level).mean(), digits=4)
    summary['report_analysis'] = generate_report_analysis(summary)
    return add_top_risk_columns(summary, df)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analyze = commands.add_parser('analyze', help='Analyze one company folder; never deletes input files')
    analyze.add_argument('input', type=Path)
    analyze.add_argument('--company', required=True)
    analyze.add_argument('--output', type=Path, default=Path('output/greenwashing'))
    analyze.add_argument('--model', default=CONFIG['zero_shot_model'])
    analyze.add_argument('--ocr', action='store_true', help='OCR text-poor PDF pages (requires Tesseract language packs)')
    sample = commands.add_parser('sample', help='Create human review workbook from scored chunks')
    sample.add_argument('chunks', type=Path); sample.add_argument('--output', default='output/human_validation.xlsx')
    sample.add_argument('--per-level', type=int, default=30)
    evaluate = commands.add_parser('evaluate', help='Evaluate completed, non-blind human review workbook')
    evaluate.add_argument('workbook'); evaluate.add_argument('--output', default='output/human_validation_evaluated.csv')
    args = parser.parse_args()
    if args.command == 'sample':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        export_human_validation_file(pd.read_csv(args.chunks), args.output, n_per_level=args.per_level)
        return
    if args.command == 'evaluate':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        evaluate_human_validation(args.workbook).to_csv(args.output, index=False)
        return
    if not args.input.is_dir():
        parser.error('input must be a directory')
    if not args.company.strip() or any(c in args.company for c in '/\\_'):
        parser.error('company must be nonempty and contain no path separators or underscores')
    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.pdf', '.txt', '.md'})
    if not files:
        parser.error('No PDF/TXT/MD reports found')
    years = [extract_report_year_from_filename(p) for p in files]
    if any(y is None for y in years) or len(set(years)) != len(years):
        parser.error('Each input report must have a unique, unambiguous year in its filename')
    args.output.mkdir(parents=True, exist_ok=True)
    clf = load_zero_shot_pipeline(args.model)
    summaries = []
    for path in files:
        chunks = split_into_chunks(load_report(path, ocr=args.ocr))
        if not chunks:
            raise ValueError(f'No usable text: {path.name}; inspect extraction or enable --ocr')
        df = add_tfidf_features(compute_rule_features(chunks))
        df = pd.concat([df, zero_shot_scores(df.text.tolist(), clf)], axis=1)
        df = compute_final_scores(df)
        df['Explanation'] = df.apply(explain_row, axis=1)
        df = df.sort_values('Greenwashing Risk', ascending=False).reset_index(drop=True)
        df.to_csv(args.output / f'{path.stem}_chunks.csv', index=False, encoding='utf-8-sig')
        summaries.append(summarize_report(df, path, args.company))
    target = args.output / f'{args.company}_{min(years)}_{max(years)}_batch_summary.csv'
    pd.DataFrame(summaries).sort_values('year').to_csv(target, index=False, encoding='utf-8-sig')
    print(target)


if __name__ == '__main__':
    main()
