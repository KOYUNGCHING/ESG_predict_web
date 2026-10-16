"""Rebuild research tables, figures and held-out evaluations from repository data.

Run with: python -m analysis.research
Outputs are separate from the legacy CSVs and deployed models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
ESG = ['TESG分數', '環境構面分數', '社會構面分數', '公司治理構面分數']
BASE = ESG + ['市值_avg']
FUTURE = BASE + ['ROA_lag']
LABELS = dict(zip(ESG + ['市值_avg', 'ROA_avg', 'ROA_lag', 'ROA_next'],
                  ['ESG', 'E', 'S', 'G', 'Market value', 'ROA', 'ROA(t-1)', 'ROA(t+1)']))
TEXT_FEATURES = ['avg_commitment_score', 'avg_evidence_score',
                 'avg_credibility_score', 'overall_greenwashing_risk_score']


def prepare_panel(df):
    """Require unique company-years; shifts refer only to adjacent calendar years."""
    df = df.copy()
    required = ['代號', '名稱', '年度', *BASE, 'ROA_avg']
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f'Missing panel columns: {sorted(missing)}')
    if df[['代號', '年度']].isna().any().any():
        raise ValueError('Company and year must be present')
    df['代號'] = df['代號'].astype(str).str.strip()
    for col in ['年度', *BASE, 'ROA_avg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['代號', '年度']].isna().any().any():
        raise ValueError('Company and year must be present')
    if df['代號'].eq('').any() or np.isinf(df[['年度', *BASE, 'ROA_avg']].to_numpy()).any():
        raise ValueError('Empty company codes and infinite numeric values are invalid')
    if not df['年度'].eq(df['年度'].round()).all():
        raise ValueError('Years must be integers')
    if df.duplicated(['代號', '年度']).any():
        raise ValueError('Duplicate company-year observations')
    df = df.sort_values(['代號', '年度']).reset_index(drop=True)
    groups = df.groupby('代號')
    df['ROA_lag'] = groups['ROA_avg'].shift(1).where(df['年度'] - groups['年度'].shift(1) == 1)
    df['ROA_next'] = groups['ROA_avg'].shift(-1).where(groups['年度'].shift(-1) - df['年度'] == 1)
    cutoff = float(df['市值_avg'].median())
    df['Large_Firm'] = (df['市值_avg'] > cutoff).astype(int)
    df['ESG_x_Large'] = df['TESG分數'] * df['Large_Firm']
    df['ESG_balance_risk'] = df[ESG[1:]].std(axis=1, ddof=0)
    return df, cutoff


def split_indices(frame, mode):
    indices = np.arange(len(frame))
    if mode == 'random':
        return train_test_split(indices, test_size=.2, random_state=42)
    if mode == 'company':
        return next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                    .split(frame, groups=frame['代號']))
    # Purge training targets that would only be available during the test year.
    years = sorted(frame['年度'].unique())
    if len(years) < 3:
        raise ValueError('Time evaluation requires at least three feature years')
    test_year = years[-1]
    train = np.flatnonzero((frame['年度'] + 1 < test_year).to_numpy())
    test = np.flatnonzero((frame['年度'] >= test_year).to_numpy())
    if len(train) < 10 or len(test) < 2:
        raise ValueError('Too few observations for a purged time split')
    return train, test


def model_suite():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=5,
                                               min_samples_leaf=5, random_state=42, n_jobs=1),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=3, learning_rate=.03,
                               subsample=.8, colsample_bytree=.8, reg_alpha=.5,
                               reg_lambda=1, random_state=42, n_jobs=1),
        'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=.03, num_leaves=31,
                                 random_state=42, n_jobs=1, verbosity=-1),
        'SVM': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1, epsilon=.1)),
    }


def evaluate_models(frame, features, target, mode):
    frame = frame.dropna(subset=[*features, target]).reset_index(drop=True)
    train, test = split_indices(frame, mode)
    x, y = frame[features], frame[target]
    rows, predictions, fitted = [], [], {}
    for name, model in model_suite().items():
        model.fit(x.iloc[train], y.iloc[train])
        pred = model.predict(x.iloc[test])
        rows.append(dict(Model=name, Target=target, Split=mode, N_train=len(train), N_test=len(test),
                         R2=float(r2_score(y.iloc[test], pred)),
                         RMSE=float(np.sqrt(mean_squared_error(y.iloc[test], pred))),
                         MAE=float(mean_absolute_error(y.iloc[test], pred))))
        for i, value in zip(test, pred):
            predictions.append(dict(Model=name, company=frame.iloc[i]['代號'], year=int(frame.iloc[i]['年度']),
                                    actual=float(y.iloc[i]), predicted=float(value)))
        fitted[name] = model
    return pd.DataFrame(rows), pd.DataFrame(predictions), fitted, (frame, train, test)


def ols_tables(panel, output):
    import statsmodels.api as sm
    rows, fits = [], []
    specifications = [('Baseline', panel, BASE),
                      ('ESG x Large', panel, ['TESG分數', '市值_avg', 'Large_Firm', 'ESG_x_Large']),
                      ('Small firms', panel[panel.Large_Firm == 0], BASE),
                      ('Large firms', panel[panel.Large_Firm == 1], BASE)]
    for name, subset, features in specifications:
        sample = subset.dropna(subset=['ROA_avg']).copy()
        x = sample[features].fillna(sample[features].median())
        model = sm.OLS(sample['ROA_avg'], sm.add_constant(x)).fit(cov_type='HC3')
        ci = model.conf_int()
        for term in model.params.index:
            rows.append(dict(Model=name, Term=term, Coefficient=model.params[term],
                             Std_error=model.bse[term], P_value=model.pvalues[term],
                             CI_low=ci.loc[term, 0], CI_high=ci.loc[term, 1]))
        fits.append(dict(Model=name, N=int(model.nobs), R2=model.rsquared,
                         Adjusted_R2=model.rsquared_adj, Covariance='HC3'))
    pd.DataFrame(rows).to_csv(output / 'ols_coefficients.csv', index=False)
    pd.DataFrame(fits).to_csv(output / 'ols_models.csv', index=False)
    return rows, fits


def load_text_features(folder):
    frames = []
    for path in sorted(folder.glob('*_batch_summary.csv')):
        data = pd.read_csv(path)
        if 'error' in data:
            data = data[data['error'].fillna('').eq('')].copy()
        data['company'] = path.name.split('_')[0]
        frames.append(data[['company', 'year', *TEXT_FEATURES]])
    if not frames:
        return pd.DataFrame(columns=['company', 'year', *TEXT_FEATURES])
    data = pd.concat(frames, ignore_index=True)
    for col in ['year', *TEXT_FEATURES]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    if data.duplicated(['company', 'year']).any():
        raise ValueError('Duplicate company-year text summaries')
    return data


def match_text(panel, text, mapping_path):
    """Match exact company aliases only; require an explicit map for differing names."""
    if not mapping_path.exists():
        return pd.DataFrame(), []
    mapping = pd.read_csv(mapping_path, dtype={'代號': str})
    if mapping['company'].duplicated().any():
        raise ValueError('Each text company must map to exactly one company code')
    text = text.merge(mapping[['company', '代號']], on='company', how='inner', validate='many_to_one')
    matched = panel.merge(text, left_on=['代號', '年度'], right_on=['代號', 'year'],
                          how='inner', validate='one_to_one')
    return matched, sorted(set(text['company']))


def figures(panel, results, predictions, clusters, coords, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    def save(name):
        plt.tight_layout(); plt.savefig(directory / name, dpi=150, bbox_inches='tight'); plt.close()
    corr = panel[list(LABELS)].rename(columns=LABELS).corr()
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax); ax.set_title('Pearson correlation (pairwise complete observations)'); save('correlation.png')
    trend = panel.groupby('年度')[['TESG分數', 'ROA_avg']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, trend):
        ax.plot(trend.index, trend[col], marker='o'); ax.set_title(LABELS[col]); ax.set_xlabel('Year')
    save('annual_trends.png')
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, col in zip(axes, ESG):
        ax.scatter(panel[col], panel.ROA_avg, s=12, alpha=.5); ax.set_xlabel(LABELS[col]); ax.set_ylabel('ROA')
    save('esg_scatter.png')
    plt.figure(figsize=(6, 4)); plt.boxplot([panel.loc[panel.Large_Firm == g, 'ROA_avg'].dropna() for g in [0, 1]], tick_labels=['Small', 'Large']); plt.ylabel('ROA'); save('size_distribution.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ['R2', 'RMSE', 'MAE']):
        ax.bar(results.Model, results[metric], color='#49aa83'); ax.set_title(metric); ax.tick_params(axis='x', rotation=65)
    save('future_model_comparison.png')
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for ax, (name, subset) in zip(axes, predictions.groupby('Model', sort=False)):
        ax.scatter(subset.actual, subset.predicted, s=16, alpha=.6)
        limits = [min(subset.actual.min(), subset.predicted.min()), max(subset.actual.max(), subset.predicted.max())]
        ax.plot(limits, limits, '--', color='gray'); ax.set_title(name); ax.set_xlabel('Actual ROA(t+1)'); ax.set_ylabel('Predicted')
    save('future_predictions.png')
    plt.figure(figsize=(7, 5)); plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='tab10', alpha=.7)
    plt.xlabel('Principal component 1'); plt.ylabel('Principal component 2'); plt.title('K-Means groups projected with PCA'); save('clusters.png')


def run(input_path, output, image_dir, mode='time', mapping_path=None, export_forecast=False):
    output.mkdir(parents=True, exist_ok=True)
    panel, cutoff = prepare_panel(pd.read_csv(input_path, dtype={'代號': str}))
    panel.to_csv(output / 'research_panel.csv', index=False)
    panel[list(LABELS)].isna().mean().rename('Missing_ratio').to_csv(output / 'missingness.csv')
    panel[list(LABELS)].describe().T.to_csv(output / 'descriptive_statistics.csv')
    panel[list(LABELS)].corr().to_csv(output / 'correlation.csv')
    panel.groupby('年度')[[*ESG, 'ROA_avg']].mean().to_csv(output / 'annual_trends.csv')
    panel.groupby('Large_Firm')[[*BASE, 'ROA_avg']].agg(['count', 'mean', 'median']).to_csv(output / 'size_groups.csv')
    ols, ols_fits = ols_tables(panel, output)
    valid = panel.dropna(subset=['ROA_avg', 'TESG分數'])
    median_roa = valid.ROA_avg.median()
    low = valid.loc[valid.ROA_avg <= median_roa, 'TESG分數']
    high = valid.loc[valid.ROA_avg > median_roa, 'TESG分數']
    test = ttest_ind(low, high, equal_var=False)
    ttest = dict(test='Welch two-sided; ROA median split', cutoff=float(median_roa),
                 n_low=len(low), n_high=len(high), mean_low=float(low.mean()), mean_high=float(high.mean()),
                 statistic=float(test.statistic), p_value=float(test.pvalue))
    # Both targets are evaluated independently; never label current ROA scores as forecasts.
    current, _, _, _ = evaluate_models(panel, BASE, 'ROA_avg', 'random')
    current.to_csv(output / 'current_roa_comparison.csv', index=False)
    results, predictions, fitted, split = evaluate_models(panel, FUTURE, 'ROA_next', mode)
    results.to_csv(output / 'future_roa_comparison.csv', index=False)
    predictions.to_csv(output / 'future_roa_predictions.csv', index=False)
    frame, train, holdout = split
    importance = permutation_importance(fitted['Random Forest'], frame[FUTURE].iloc[holdout],
                                        frame.ROA_next.iloc[holdout], n_repeats=10,
                                        random_state=42, scoring='neg_mean_absolute_error')
    importance_table = pd.DataFrame(dict(Feature=FUTURE, Importance=importance.importances_mean,
                                         Std=importance.importances_std)).sort_values('Importance', ascending=False)
    importance_table.to_csv(output / 'future_permutation_importance.csv', index=False)
    # New, explicit research definition: training-only tertiles of next-year ROA.
    q1, q2 = np.quantile(frame.ROA_next.iloc[train], [1/3, 2/3])
    risk_y = np.where(frame.ROA_next <= q1, 'High Risk', np.where(frame.ROA_next <= q2, 'Medium Risk', 'Low Risk'))
    classifier = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                        class_weight='balanced', random_state=42, n_jobs=1)
    classifier.fit(frame[FUTURE].iloc[train], risk_y[train]); risk_pred = classifier.predict(frame[FUTURE].iloc[holdout])
    risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
    risk = dict(definition='Training-set tertiles of ROA_next; research proxy, not default probability',
                q1=float(q1), q2=float(q2), n_train=len(train), n_test=len(holdout),
                accuracy=float(accuracy_score(risk_y[holdout], risk_pred)),
                report=classification_report(risk_y[holdout], risk_pred, labels=risk_labels, output_dict=True, zero_division=0),
                labels=risk_labels, confusion_matrix=confusion_matrix(risk_y[holdout], risk_pred, labels=risk_labels).tolist())
    cluster_cols = BASE + ['ROA_avg', 'ESG_balance_risk']
    complete = panel.dropna(subset=cluster_cols).copy()
    scaled = StandardScaler().fit_transform(complete[cluster_cols])
    clusters = KMeans(n_clusters=4, n_init=20, random_state=42).fit_predict(scaled)
    complete['Cluster'] = clusters
    pca = PCA(n_components=2); coords = pca.fit_transform(scaled)
    complete['PC1'], complete['PC2'] = coords[:, 0], coords[:, 1]
    complete[['代號', '名稱', '年度', 'Cluster', 'PC1', 'PC2']].to_csv(output / 'cluster_assignments.csv', index=False)
    profiles = complete.groupby('Cluster')[cluster_cols].mean().reset_index()
    profiles['N'] = complete.groupby('Cluster').size().to_numpy()
    profiles.to_csv(output / 'cluster_profiles.csv', index=False)
    text = load_text_features(ROOT / 'data_2')
    matched, aliases = match_text(panel, text, mapping_path or ROOT / 'data/company_aliases.csv')
    external, advanced = [], dict(status='insufficient_matched_data', matched_rows=len(matched), companies=aliases)
    if not matched.empty:
        for col in TEXT_FEATURES:
            pair = matched[['TESG分數', col]].dropna()
            if len(pair) >= 3 and pair[col].nunique() > 1 and pair['TESG分數'].nunique() > 1:
                stat = pearsonr(pair['TESG分數'], pair[col])
                external.append(dict(Feature=col, N=len(pair), Correlation=float(stat.statistic), P_value=float(stat.pvalue)))
        comparable = matched.dropna(subset=[*FUTURE, *TEXT_FEATURES, 'ROA_next'])
        if len(comparable) >= 30 and comparable['年度'].nunique() >= 3:
            baseline, _, _, _ = evaluate_models(comparable, FUTURE, 'ROA_next', mode)
            augmented, _, _, _ = evaluate_models(comparable, FUTURE + TEXT_FEATURES, 'ROA_next', mode)
            baseline['Features'], augmented['Features'] = 'Baseline', 'With text'
            pd.concat([baseline, augmented]).to_csv(output / 'text_model_comparison.csv', index=False)
            advanced['status'] = 'completed'
    pd.DataFrame(external, columns=['Feature', 'N', 'Correlation', 'P_value']).to_csv(output / 'text_external_validation.csv', index=False)
    figures(panel, results, predictions, clusters, coords, image_dir)
    import sklearn, scipy, xgboost, lightgbm, statsmodels
    summary = dict(source_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                   versions={m.__name__: m.__version__ for m in [np, pd, sklearn, scipy, xgboost, lightgbm, statsmodels]},
                   seed=42, split=mode, n_rows=len(panel), n_companies=int(panel['代號'].nunique()),
                   year_min=int(panel['年度'].min()), year_max=int(panel['年度'].max()), size_cutoff=cutoff,
                   future_models=results.to_dict('records'), current_models=current.to_dict('records'),
                   forecast_importance=importance_table.to_dict('records'),
                   ols=ols, ols_fits=ols_fits, ttest=ttest, risk=risk,
                   clusters=profiles.to_dict('records'), pca_variance=pca.explained_variance_ratio_.tolist(),
                   external_validation=external, text_comparison=advanced,
                   notes=['ROA_next uses adjacent calendar years only.',
                          'Time split purges training targets in the first test feature year.',
                          'ROA observed at t+1 is not necessarily disclosed at t+1; publication dates unavailable.',
                          'OLS uses HC3 standard errors; observational associations are not causal.',
                          'Risk classifier and clusters are new research runs; deployed models are unchanged.',
                          'Text report-year matches do not establish availability at forecast time.',
                          'Repeated company-years make simple Pearson p-values exploratory.'])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if export_forecast:
        import joblib
        from sklearn.base import clone
        # Refit the evaluated RF specification on all observed forecast pairs.
        # Evaluation metrics above remain those of the held-out fit, not this refit.
        forecast = clone(fitted['Random Forest'])
        forecast.fit(frame[FUTURE], frame.ROA_next)
        model_dir = ROOT / 'models'
        model_dir.mkdir(exist_ok=True)
        joblib.dump(forecast, model_dir / 'roa_next_model.joblib')
        joblib.dump(FUTURE, model_dir / 'roa_next_features.joblib')
        metadata = dict(target='ROA_next', estimator='RandomForestRegressor', features=FUTURE,
                        n_fit=len(frame), source_sha256=summary['source_sha256'],
                        sklearn_version=sklearn.__version__, seed=42,
                        evaluation=next(row for row in summary['future_models'] if row['Model'] == 'Random Forest'),
                        note='Deployment refit on all complete adjacent-year pairs; evaluation used a separate held-out fit.')
        (model_dir / 'roa_next_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote analysis to {output}; figures to {image_dir}')
    print(results[['Model', 'R2', 'RMSE', 'MAE']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/master_full_merged.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/analysis')
    parser.add_argument('--figures', type=Path, default=ROOT / 'static/images/analysis')
    parser.add_argument('--split', choices=['time', 'company', 'random'], default='time')
    parser.add_argument('--company-map', type=Path, default=ROOT / 'data/company_aliases.csv')
    parser.add_argument('--export-forecast', action='store_true', help='Refit RF on all complete pairs and export new roa_next_* artifacts; keeps legacy models intact')
    args = parser.parse_args()
    run(args.input, args.output, args.figures, args.split, args.company_map, args.export_forecast)


if __name__ == '__main__':
    main()
