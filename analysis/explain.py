"""Create SHAP explanations for the new, documented deployed ROA-next forest.

Optional: pip install shap; python -m analysis.explain
Explains the deployment refit, not the independently evaluated holdout model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .research import ROOT, FUTURE, prepare_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/shap')
    parser.add_argument('--max-samples', type=int, default=50)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error('--max-samples must be positive')
    try:
        import shap
    except ImportError as exc:
        raise SystemExit('Install optional SHAP dependencies: pip install -r requirements-explain.txt') from exc
    metadata = json.loads((ROOT / 'models/roa_next_metadata.json').read_text())
    if metadata['target'] != 'ROA_next' or metadata['features'] != FUTURE:
        raise ValueError('Expected the documented ROA_next forest and feature schema')
    model = joblib.load(ROOT / 'models/roa_next_model.joblib')
    panel, _ = prepare_panel(pd.read_csv(ROOT / 'data/master_full_merged.csv', dtype={'代號': str}))
    sample = panel.dropna(subset=FUTURE).tail(args.max_samples)
    x = sample[FUTURE]
    explanation = shap.TreeExplainer(model)(x)
    # Validate the additive decomposition against the actual model predictions.
    if not np.allclose(explanation.base_values + explanation.values.sum(axis=1), model.predict(x), atol=1e-5):
        raise ValueError('SHAP additive reconstruction does not match model predictions')
    args.output.mkdir(parents=True, exist_ok=True)
    values = pd.DataFrame(explanation.values, columns=FUTURE, index=x.index)
    values.insert(0, 'base_value', explanation.base_values)
    values.insert(0, '年度', sample['年度'])
    values.insert(0, '代號', sample['代號'])
    values.to_csv(args.output / 'shap_values.csv', index=False)
    pd.DataFrame({'Feature': FUTURE, 'Mean_absolute_SHAP': np.abs(explanation.values).mean(axis=0)}).sort_values(
        'Mean_absolute_SHAP', ascending=False).to_csv(args.output / 'shap_importance.csv', index=False)
    print(f'Saved SHAP decompositions for {len(sample)} company-years to {args.output}')


if __name__ == '__main__':
    main()
