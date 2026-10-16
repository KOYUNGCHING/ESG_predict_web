# Research coverage and provenance

## Source inventory

| Supplied source | Treatment |
| --- | --- |
| 知行合一期末報告＿機器學習的部分.ipynb | Archived as `notebooks/esg_financial_research.ipynb`; local analysis implemented in `analysis/research.py`. |
| ESG_期中.ipynb | All 81 cell sources match the final ML notebook. Recorded as a duplicate in `notebooks/sources.json`. |
| Esg_Greenwashing_Analyzer_template.ipynb | Archived as `notebooks/greenwashing_analyzer.ipynb`; scoring and human-validation functions adapted into `analysis/greenwashing.py`. |
| The three Colab PDF exports | Used to cross-check notebook methodology and displayed outputs; not treated as additional executable code. |
| AI 知行合一 期中報告.pdf | Research questions, t-test, EDA, firm-size heterogeneity, and current-year model discussion checked against available data. |
| 期末報告.pdf | Used for the separately labeled report-result transcription and identifying missing extensions. |

Notebook outputs and execution metadata were cleared in the archive. Original code remains historical context, including Colab installation/download/cleanup cells. Do not run all historical cells unattended. The adapted module does not include automatic folder deletion, interactive prompts, Colab uploads, or automatic downloads of result files.

## Implementation map

| Analysis | Local implementation | Web presentation / status |
| --- | --- | --- |
| Missingness, descriptive statistics, lag/lead panel | `analysis.research.prepare_panel` / `run` | `/research` overview; tables in `data/analysis/` |
| Correlations, scatterplots, annual trends, size distributions | `analysis.research.figures` | `/research`, generated PNGs |
| ROA-group ESG t-test | Welch, two-sided, ROA median split | `/research`; dependent panel observations limit inference |
| OLS, ESG × Large, separate size-group regressions | `ols_tables`, HC3 covariance | `/research`, full coefficient tables |
| Five current/future ROA models | `evaluate_models`, `split_indices` | Current results separate from forecast results; forecast comparison on `/research` |
| Feature importance | Held-out RF permutation importance in MAE units; legacy RF impurity table retained | CSV export; not SHAP |
| Financial classification | New documented training-only ROA-next tertile labels | `/research`, per-class metrics and exported confusion matrix; old model unchanged |
| Four company clusters | K-Means on standardized seven-variable research matrix + PCA | `/research`, profiles and projection; old deployment cluster model unchanged |
| Report scoring / OCR / TF-IDF / zero-shot | `analysis.greenwashing` | Offline command; existing `/greenwashing` consumes reviewed summary CSVs |
| Human review sampling and evaluation | Adapted notebook functions + CLI | Ready for user annotations; no actual human validation scores available |
| Text/TESG external validation | Explicit company-code mapping, same report year, pairwise complete Pearson | `/research`; confirmed entities only, no assumed parent/subsidiary equivalence |
| Baseline vs text-augmented prediction | Same complete matched cohort and split for both feature sets | Implemented; insufficient confirmed matched rows to run with supplied data |
| Year-effect ROA / ROE / Tobin Q / TESG regressions | `analysis.panel_regressions` | Code ready; unavailable extended input prevents empirical reproduction |
| SHAP | `analysis.explain` provides optional TreeExplainer exports with an additive check for the new ROA-next forest | Original values are unavailable; new SHAP command is provided but has not been run |

## Methodological repairs

- Original model comparison exports predict **current ROA**. The dashboard now labels this correctly.
- Forecast lag and lead construction rejects duplicate company-years and does not bridge missing years.
- Default forecast evaluation uses a purged chronological holdout. Random and company holdouts are explicit alternatives.
- Forecast complete-case selection occurs before splitting. No test data are used to choose financial-risk tertiles.
- Current and future results, original model files, recomputed outputs, and transcribed report results remain distinct.
- Source notebook long sentences could exceed the configured chunk length; the adapted splitter bounds them.
- Human labels outside the allowed domains are rejected rather than silently treated as valid annotations.

## Important reproducibility limits

The final ML notebook supplied here has the same cells as the midterm-named notebook. Its last save cells refer to undefined `model` and `feature_columns`, and do not reliably identify a final deployment artifact. The final risk and clustering training scripts are absent even though trained artifacts exist. We preserve the legacy artifacts, create independently documented research runs, and export a new six-feature `roa_next_model.joblib` with target/schema metadata. The website now uses that explicit forecast artifact, while its existing risk and cluster models remain unchanged.

The final presentation contains regression results not present in these notebooks. No supplied panel has ROE or Tobin's Q. The report's size transformation, complete training split, and paragraph annotations are missing. New scripts validate their inputs and require actual data; report coefficients and performance are never fabricated.

Only 彰銀 and 統一證 have confirmed report labels matching panel entities. 台新, 富邦, and 國泰 must be checked for holding-company versus bank scope. Add reviewed aliases to `data/company_aliases.csv`. At least 30 complete matched rows and sufficient years are required to run the illustrative text-feature forecasting comparison; this threshold is an execution guard, not evidence that a small sample is statistically adequate.

The time split is based on accounting years, not publication timestamps. Report scores use within-report normalization; length, language, OCR quality, and the selected model can affect them. Repeated observations within companies, omitted fixed effects, multicollinearity between TESG and its components, and selection into complete cases all constrain interpretation.

## Rebuilding and inspecting artifacts

Run the commands in the English README. Generated results record library versions and the input SHA-256 in `data/analysis/summary.json`. Exact numbers can change across environments. Research requirements are independent of the web runtime; full NLP requires optional dependencies and model weights. No automatic Git commit, push, or live deployment is part of these workflows.
