"""Year-effect regressions for the final report, given the missing extended panel.

Input schema: company, year, tesg_score, roa, roe, tobin_q, size.
The size column must use the original study's documented transformation/units.
This module does not infer ROE or Tobin's Q from ROA.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import statsmodels.api as sm


def regressions(data):
    required = ['company', 'year', 'tesg_score', 'roa', 'roe', 'tobin_q', 'size']
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    data = data[required].copy()
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    if data[['company', 'year']].isna().any().any() or data.duplicated(['company', 'year']).any():
        raise ValueError('Require nonmissing, unique company-year keys')
    data = data.sort_values(['company', 'year'])
    groups = data.groupby('company')
    adjacent = groups.year.shift(-1) - data.year == 1
    for col in ['roa', 'roe', 'tobin_q', 'tesg_score']:
        data[f'{col}_next'] = groups[col].shift(-1).where(adjacent)
    specs = [
        ('ROA A', 'roa_next', ['tesg_score', 'roa', 'size', 'tobin_q']),
        ('ROA B', 'roa_next', ['tesg_score', 'roa', 'size']),
        ('ROA C', 'roa_next', ['tesg_score', 'size', 'tobin_q']),
        ('ROE', 'roe_next', ['tesg_score', 'roe', 'size', 'tobin_q']),
        ('Tobin Q', 'tobin_q_next', ['tesg_score', 'tobin_q', 'roa', 'size']),
        ('TESG A', 'tesg_score_next', ['roe', 'tobin_q', 'size', 'tesg_score']),
        ('TESG B', 'tesg_score_next', ['roa', 'tobin_q', 'size', 'tesg_score']),
        ('TESG C', 'tesg_score_next', ['roa', 'size', 'tesg_score']),
        ('TESG D', 'tesg_score_next', ['tobin_q', 'size', 'tesg_score']),
    ]
    coefficients, summaries = [], []
    for name, target, features in specs:
        sample = data.dropna(subset=[target, *features])
        years = pd.get_dummies(sample.year.astype(int).astype(str), prefix='year', drop_first=True, dtype=float)
        x = sm.add_constant(pd.concat([sample[features], years], axis=1)).astype(float)
        if len(sample) <= x.shape[1] + 1:
            raise ValueError(f'Too few complete observations for {name}')
        fit = sm.OLS(sample[target], x).fit(cov_type='HC3')
        for term in fit.params.index:
            coefficients.append(dict(Model=name, Target=target, Term=term, Coefficient=fit.params[term],
                                     P_value=fit.pvalues[term], Std_error=fit.bse[term]))
        summaries.append(dict(Model=name, Target=target, N=int(fit.nobs), R2=fit.rsquared,
                              Adjusted_R2=fit.rsquared_adj, F=fit.fvalue, F_p_value=fit.f_pvalue,
                              Covariance='HC3', Year_effects=True))
    return pd.DataFrame(coefficients), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/panel_regressions'))
    args = parser.parse_args()
    coefficients, models = regressions(pd.read_csv(args.input, dtype={'company': str}))
    args.output.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(args.output / 'coefficients.csv', index=False)
    models.to_csv(args.output / 'models.csv', index=False)


if __name__ == '__main__':
    main()
