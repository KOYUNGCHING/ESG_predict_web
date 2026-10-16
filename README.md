# Earth Shareholders — ESG Research & Financial Analysis

A Flask application and reproducible research workspace for studying ESG scores, profitability, and sustainability-report disclosures in Taiwan's financial sector. The website interface is in Traditional Chinese; this README documents setup and research workflows in English.

The project brings together three components:

- **Company analysis:** next-year ROA inference using a newly documented Random Forest model, financial-risk classification, K-Means grouping, and a separate rule-based ESG warning score.
- **Sustainability report screening:** browse precomputed company-year report scores and the five highest-risk passages.
- **Research results:** explore data quality, correlations, annual trends, OLS and firm-size heterogeneity, five forecasting models, risk classification, clustering, and text-feature validation.

These are exploratory research tools. Statistical association is not causation, relative profitability classes are not default probabilities, and text-screening scores do not establish greenwashing.

## Quick start

Use Python **3.11** for the original model environment (`runtime.txt`). The existing scikit-learn artifacts were saved with **1.6.1**, so the web requirements pin that version. Loading serialized models with other scikit-learn versions is unsupported. Only load trusted model artifacts.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows, activate with `.venv\Scripts\activate` and use `py -3.11` to create the environment. The application defaults to **http://127.0.0.1:8000**. Override the port with the `PORT` environment variable. The `Procfile` runs Gunicorn for hosting services that provide `$PORT`:

```bash
gunicorn app:app --bind 0.0.0.0:8000 --workers 1 --threads 2 --timeout 120
```

The research and NLP dependencies are optional: installing the web requirements is enough to serve the committed results. Training never runs during a web request.

## Website routes

| Route | Purpose |
| --- | --- |
| `/` | Home and feature navigation |
| `/ml-prediction` | Company-level ROA inference, risk class, cluster, and heuristic warnings |
| `/greenwashing` | Select a pre-analyzed company-year sustainability report |
| `/dashboard` | Original exported **current-year ROA** comparisons and summary tables |
| `/research` | Expanded research results, figures, and explicitly labeled report excerpts |
| `/usage` | User guide |
| `/about` | Project background |
| `/links` | Related resources |

The live report page reads `data_2/*.csv`; it does not upload documents or run a language model. New reports are analyzed offline and then added to that folder.

## Repository structure

```text
app.py                         Flask routes and existing inference logic
analysis/
  research.py                  Rebuild research tables, figures, and evaluations
  greenwashing.py              Offline PDF/TXT/MD scoring and human validation
  panel_regressions.py         Extended year-effect regression workflow
  explain.py                   Optional SHAP decomposition of the forecast forest
notebooks/                     Historical source notebooks, with outputs cleared
  sources.json                 Source hashes and duplicate-notebook provenance
data/
  master_full_merged.csv        Original extended research table (284 rows)
  model_base_merged.csv         Original nine-column model table
  company_aliases.csv           Explicit report-company to panel-code mappings
  analysis/                    Newly computed outputs + separate reported results
data_2/                        Precomputed report summaries and top passages
models/                        Preserved legacy artifacts + new roa_next_* forecast model
static/images/analysis/         Generated research figures
templates/                     Traditional Chinese web pages
docs/RESEARCH_COVERAGE.md        What was integrated, changed, or remains unavailable
tests/                         Panel integrity, scoring, regression, and route tests
requirements.txt               Web/inference dependencies
requirements-analysis.txt      Optional tabular research dependencies
requirements-nlp.txt           Optional transformer/OCR/human-review dependencies
vercel.json                    Disables automatic Vercel Git deployments
```

## Rebuild tabular research

```bash
python -m pip install -r requirements-analysis.txt
python -m analysis.research
```

On macOS, LightGBM requires the OpenMP runtime (`brew install libomp`). Matplotlib may need a writable configuration directory in restricted environments (`MPLCONFIGDIR=/tmp/esg-matplotlib`).

The default input is `data/master_full_merged.csv`. Outputs go to `data/analysis/` and `static/images/analysis/`; the existing `models/` and original CSV exports are **not overwritten**.

The workflow produces:

1. A validated company-year panel, missing-value ratios, descriptive statistics, correlations, annual trends, and firm-size group summaries.
2. Baseline OLS, ESG × large-firm interactions, separate large/small company regressions, HC3 standard errors, confidence intervals, and an exploratory Welch t-test of ESG across ROA groups.
3. Separate comparisons for **current ROA** and **next-year ROA** using Linear Regression, Random Forest, XGBoost, LightGBM, and scaled RBF SVM. Forecast outputs include held-out predictions and Random Forest permutation importance.
4. A new research risk classifier with **training-set next-year ROA tertiles** as labels, per-class metrics, and a confusion matrix. This explicit proxy definition does not reproduce undocumented labels in the old classifier.
5. Standardized four-cluster K-Means, cluster profiles, company-year assignments, and a PCA visualization.
6. Text/TESG external correlations where company-year matching is confirmed. Baseline versus text-augmented models run only with at least 30 complete matched observations and sufficient years.
7. `summary.json`, including input SHA-256, package versions, seed, split, sample sizes, and limitations.

### Evaluation design

The default `--split time` reserves the latest eligible feature year. It removes training observations whose next-year targets reach the test feature year. Lag and lead values are generated only for **adjacent calendar years** within the same company. This reduces temporal leakage but does not reconstruct real-time availability: financial-statement and report publication dates are unavailable.

Alternative evaluations can be written to separate output folders:

```bash
python -m analysis.research --split company --output output/company --figures output/company/figures
python -m analysis.research --split random --output output/random --figures output/random/figures
```

A random company-year split can place the same firm in training and test sets; it is retained for comparison with the original notebooks, not as proof of out-of-time performance. The source notebooks also imputed some features before splitting; the rebuilt prediction workflow instead uses complete cases, while descriptive OLS uses median-imputed explanatory variables.

### Text-to-panel matching

`data/company_aliases.csv` currently maps only **彰銀 → 2801** and **統一證 → 2855**, which match the panel entities. The 富邦, 國泰, and 台新 report labels do not identify whether the documents cover a bank, holding company, or group. Confirm the reporting entity before adding mappings. Never assume a subsidiary report represents its parent.

Only two confirmed companies overlap with the current panel, so the text-augmented forecasting comparison currently reports **insufficient matched data**. Code is provided; missing data are not replaced with invented results. Report-year joins also require publication-date checks before any operational forecasting use. Simple Pearson p-values and the Welch test are exploratory because company-year observations are dependent.

## Analyze new sustainability reports offline

```bash
python -m pip install -r requirements-nlp.txt
python -m analysis.greenwashing analyze /path/to/reports \
  --company ExampleCompany --output output/greenwashing
```

Use one company's reports per directory. Each filename must contain a unique report year, such as `ExampleCompany_2023.pdf`. Supported files are PDF, TXT, and MD. The command loads `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` once, which downloads model weights on first use. GPU acceleration is optional; CPU analysis can be slow. Use `--model /path/to/local/model` for existing local weights.

The pipeline extracts text, bounds chunks to 900 characters, derives keyword/numeric/TF-IDF features, applies zero-shot classification, and combines language-model scores with the original notebook's heuristic formulas. It exports per-report paragraph CSVs and a company-year batch summary compatible with `data_2/`.

For scanned PDF pages, add `--ocr`. Install Tesseract with Traditional Chinese and English language data first. OCR uses PyMuPDF's Tesseract integration. Input reports are never deleted automatically.

Scoring conventions inherited from the source notebook:

- Paragraph levels: Low < 40, Medium 40–<70, High ≥70.
- Report levels (mean paragraph risk): Low <30, Medium 30–<60, High ≥60.
- The rule component uses within-report min–max normalization. Scores depend on report composition and should not be treated as calibrated probabilities or directly comparable validated measurements.
- The company input page's tabular warning score is a different heuristic from PDF text scoring.

To publish precomputed results in the local website, review the generated batch summary and copy it to `data_2/`, then restart the app. Model weights and source PDF reports are not required by the web runtime.

### Human review

```bash
python -m analysis.greenwashing sample output/greenwashing/ExampleCompany_2023_chunks.csv \
  --output output/human_validation.xlsx --per-level 30
# Fill the Human Commitment/Evidence/Credibility and Human Risk Level columns.
python -m analysis.greenwashing evaluate output/human_validation.xlsx \
  --output output/human_validation_evaluated.csv
```

The workbook has annotation instructions and dropdown values. The evaluator prints accuracy, precision, recall, F1, and confusion matrices for available labels. A genuinely blind review should retain model scores in a separate keyed file and merge by sample/chunk ID before evaluation; a blind workbook alone cannot be scored. No completed human-label workbook was supplied, so human-validation performance is not claimed.

## Extended panel regressions

The final presentation discusses next-year ROA, ROE, Tobin's Q, and TESG with year effects. The supplied repository table lacks ROE and Tobin's Q, and the original size transformation is unspecified. To execute the added regression workflow, provide a CSV with these columns:

```text
company,year,tesg_score,roa,roe,tobin_q,size
```

```bash
python -m analysis.panel_regressions /path/to/extended_panel.csv \
  --output output/panel_regressions
```

This runs the full/reduced ROA specifications, ROE and Tobin's Q equations, and four reverse-direction TESG specifications. Each uses adjacent-year targets, year indicators, complete cases, and HC3 standard errors. HC3 is the rebuilt workflow's inference choice; without the original regression scripts and data it cannot promise identical report p-values.

### Export the website forecast model

The website now loads `models/roa_next_model.joblib`, its matching feature list, and `roa_next_metadata.json`. The new artifact is explicitly trained on `ROA_next` with all six forecast features. The loader verifies the recorded target and exact feature schema. The historical five-feature artifact had no reliable target metadata and is preserved as a legacy file.

```bash
python -m analysis.research --export-forecast
```

This explicit option refits the evaluated Random Forest specification on all complete adjacent-year pairs and writes new `roa_next_*` artifacts. Held-out evaluation scores still describe the separate training/test fit, not the deployment refit. Random Forest is retained for the deployment's existing explanation behavior; the highest scoring model in a single comparison is not automatically promoted. The app's feature importance now comes from the actual loaded forecast artifact. The original risk classifier and cluster models remain in use on the company input page.

### Optional SHAP explanation

```bash
python -m pip install -r requirements-explain.txt
python -m analysis.explain --output output/shap --max-samples 50
```

This optional command exports per-company-year SHAP values, base values, and mean absolute SHAP importance for the new deployed forest. It verifies that the SHAP contributions reconstruct model predictions. It explains the refitted deployment model, not out-of-sample performance. SHAP inference has not been run in the bundled research results; the optional dependency is not needed by the website. These outputs are distinct from the unavailable original report's SHAP analysis.

## Original results versus recomputed results

The original `data/model_results_summary.csv` contains three **current ROA** model scores, including XGBoost R² ≈0.858. Those scores must not be presented as next-year forecast performance or as the accuracy of the deployed artifact.

`data/analysis/reported_results.json` separately transcribes the supplied final presentation's five future-model scores (page 22), text/TESG correlations (page 8), risk metrics (page 23), and selected regression summaries (pages 12–17). The website labels them **reported, not recomputed**. Different targets, splits, model versions, and preprocessing can produce different results.

The two supplied ML notebooks contain identical cell sources and are archived once. Archived notebooks retain Colab-specific cells and historical inconsistencies (including undefined final save variables); use the `analysis` modules for a repeatable local workflow. The original final risk-training labels, SHAP values, full text-augmented dataset, and extended regression data were not supplied. Feature importance is not labeled SHAP. See [research coverage](docs/RESEARCH_COVERAGE.md) for details.

## Verification

```bash
python -m unittest discover -s tests -v
```

The full test suite needs `requirements-analysis.txt`; NLP formula tests do not download a transformer model. Full transformer inference and OCR require the optional models/language data and source reports.

## Hosting

The existing `Procfile` supports Gunicorn-based hosting such as Render. `vercel.json` sets `git.deploymentEnabled` to `false`, disabling Vercel's automatic deployments from Git once pushed. This leaves historical deployment statuses intact. No hosting credentials are stored in the project.

## Data and research scope

The bundled panel contains 284 company-year observations from 41 financial-sector companies across 2016–2022. Only 277 rows have observed current ROA. Data originate from the project's TEJ-based research exports; sustainability summaries cover five report-company labels over differing year ranges. Check the applicable data permissions before redistributing research inputs. No license is inferred from a repository being public.

Model behavior outside the sampled sector, period, and variable definitions has not been established. Market value and ROA inputs should use the same units and definitions as the training exports; the source does not fully document market-value units. Results support coursework and research review, rather than investment or regulatory decisions.
