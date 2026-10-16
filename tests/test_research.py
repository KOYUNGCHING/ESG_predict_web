import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from analysis.research import BASE, FUTURE, prepare_panel, split_indices, match_text, load_text_features
from analysis.greenwashing import (compute_final_scores, compute_rule_features, split_into_chunks,
                                   get_overall_risk_level, summarize_report, extract_report_year_from_filename)

ROOT = Path(__file__).resolve().parents[1]


def sample_panel():
    return pd.DataFrame([
        {'代號': code, '名稱': code, '年度': year, 'ROA_avg': year - 2010,
         **{key: 50 + j + (year - 2016) for j, key in enumerate(BASE)}}
        for code in ['A', 'B', 'C', 'D', 'E'] for year in range(2016, 2023)
    ])


class PanelTests(unittest.TestCase):
    def test_lag_and_lead_do_not_cross_missing_years_or_companies(self):
        source = sample_panel()
        source = source[~((source['代號'] == 'A') & (source['年度'] == 2018))]
        panel, _ = prepare_panel(source.sample(frac=1, random_state=1))
        a = panel[panel['代號'] == 'A'].set_index('年度')
        self.assertTrue(pd.isna(a.loc[2017, 'ROA_next']))
        self.assertTrue(pd.isna(a.loc[2019, 'ROA_lag']))
        self.assertEqual(a.loc[2020, 'ROA_lag'], 9)
        for _, group in panel.groupby('代號'):
            self.assertTrue(pd.isna(group.iloc[0].ROA_lag))
            self.assertTrue(pd.isna(group.iloc[-1].ROA_next))

    def test_duplicate_company_year_rejected(self):
        source = sample_panel()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            prepare_panel(pd.concat([source, source.iloc[:1]]))

    def test_purged_time_split(self):
        panel, _ = prepare_panel(sample_panel())
        panel = panel.dropna(subset=FUTURE + ['ROA_next']).reset_index(drop=True)
        train, test = split_indices(panel, 'time')
        self.assertLess((panel.iloc[train]['年度'] + 1).max(), panel.iloc[test]['年度'].min())
        self.assertEqual(set(panel.iloc[test]['年度']), {2021})

    def test_company_holdout_has_no_shared_firms(self):
        panel, _ = prepare_panel(sample_panel())
        train, test = split_indices(panel, 'company')
        self.assertFalse(set(panel.iloc[train]['代號']) & set(panel.iloc[test]['代號']))

    def test_only_explicit_company_mapping_is_used(self):
        panel, _ = prepare_panel(pd.read_csv(ROOT / 'data/master_full_merged.csv', dtype={'代號': str}))
        text = load_text_features(ROOT / 'data_2')
        matched, _ = match_text(panel, text, ROOT / 'data/company_aliases.csv')
        self.assertEqual(set(matched.company), {'彰銀', '統一證'})
        self.assertFalse(matched.duplicated(['代號', '年度']).any())


class ScoringTests(unittest.TestCase):
    def test_long_sentences_are_bounded(self):
        text = '環境改善' * 800
        chunks = split_into_chunks(text)
        self.assertTrue(chunks)
        self.assertLessEqual(max(map(len, chunks)), 900)
        self.assertGreater(sum(map(len, chunks)), 2500)

    def test_years_and_ambiguous_filenames(self):
        self.assertEqual(extract_report_year_from_filename('company_2024.pdf'), 2024)
        self.assertEqual(extract_report_year_from_filename('民國112年報告.pdf'), 2023)
        self.assertIsNone(extract_report_year_from_filename('company_2022_2023.pdf'))

    def test_original_score_formula_and_distinct_report_thresholds(self):
        frame = compute_rule_features(['plain text'])
        for col in ['llm_commitment', 'llm_evidence', 'llm_credibility', 'llm_greenwash']:
            frame[col] = 0.
        scored = compute_final_scores(frame)
        # No keywords/numbers: credibility rule=.2 => final credibility=8.
        self.assertAlmostEqual(scored.iloc[0]['Credibility Score'], 8.)
        self.assertAlmostEqual(scored.iloc[0]['Greenwashing Risk'], 12.88)
        self.assertEqual(scored.iloc[0]['Risk Level'], 'Low')
        self.assertEqual(get_overall_risk_level(60), 'High')
        self.assertEqual(get_overall_risk_level(30), 'Medium')
        self.assertEqual(get_overall_risk_level(29.9), 'Low')
        summary = summarize_report(scored, 'company_2023.txt', 'company')
        self.assertEqual(summary['year'], 2023)
        self.assertEqual(summary['n_chunks'], 1)
        self.assertIn('top5_text', summary)
        self.assertIsNone(summary['top5_text'])


class HumanReviewTests(unittest.TestCase):
    def test_scoring_and_completed_workbook_roundtrip(self):
        from analysis.greenwashing import analyze_report, export_human_validation_file, evaluate_human_validation
        # Deterministic classifier fixture exercises the data flow, not real NLP accuracy.
        def classifier(batch, candidate_labels, **kwargs):
            return [{'labels': candidate_labels, 'scores': [.8, .7, .6, .2]} for _ in batch]
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'Example_2023.txt'
            report.write_text(('Our company will reduce carbon emissions by 20 percent in 2030. '
                               'Independent audit verified energy data and employee safety indicators. ') * 20)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                scored = analyze_report(str(report), clf=classifier)
                workbook = Path(folder) / 'review.xlsx'
                sample = export_human_validation_file(scored, str(workbook), n_per_level=3)
                for column in ['Human Commitment', 'Human Evidence', 'Human Credibility']:
                    sample[column] = 1
                sample['Human Risk Level'] = 1
                with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
                    sample.to_excel(writer, sheet_name='human_labeling', index=False)
                evaluated = evaluate_human_validation(str(workbook))
                self.assertEqual(len(evaluated), len(sample))
                self.assertIn('Pred Risk Level Numeric', evaluated)
                sample['Human Risk Level'] = 9
                with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
                    sample.to_excel(writer, sheet_name='human_labeling', index=False)
                with self.assertRaisesRegex(ValueError, 'Invalid labels'):
                    evaluate_human_validation(str(workbook))


class RegressionTests(unittest.TestCase):
    def test_extended_inputs_are_required(self):
        from analysis.panel_regressions import regressions
        with self.assertRaisesRegex(ValueError, 'Missing columns'):
            regressions(pd.DataFrame({'company': ['A'], 'year': [2020]}))

    def test_extended_regressions_have_year_effects_and_future_targets(self):
        from analysis.panel_regressions import regressions
        rng = np.random.default_rng(10)
        source = pd.DataFrame([
            dict(company=str(company), year=year, tesg_score=rng.normal(60, 10),
                 roa=rng.normal(1, .5), roe=rng.normal(5, 1), tobin_q=rng.uniform(.8, 2), size=rng.uniform(10, 20))
            for company in range(15) for year in range(2016, 2023)
        ])
        coefficients, models = regressions(source)
        self.assertEqual(len(models), 9)
        self.assertTrue(models.Year_effects.all())
        self.assertTrue(coefficients.Term.str.startswith('year_').any())
        self.assertTrue(coefficients.Target.str.endswith('_next').all())
        self.assertEqual(set(models.N), {90})


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.module = app
        cls.client = app.app.test_client()

    def test_pages_and_generated_assets(self):
        for path in ['/', '/usage', '/about', '/links', '/greenwashing', '/dashboard', '/research', '/ml-prediction']:
            with self.subTest(path=path), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.assertEqual(self.client.get(path).status_code, 200)
        response = self.client.get('/research').get_data(as_text=True)
        self.assertIn('期末報告摘錄', response)
        self.assertIn('36 筆', response)
        import re
        for url in re.findall(r'<img src="([^"]+)"', response):
            with self.client.get(url) as asset:
                self.assertEqual(asset.status_code, 200, url)

    def test_original_comparison_is_not_labeled_as_forecast(self):
        response = self.client.get('/dashboard').get_data(as_text=True)
        self.assertIn('原始當期 ROA', response)
        self.assertNotIn('最佳 ROA 預測模型', response)

    def test_real_prediction_post_and_model_importance(self):
        form = {'tesg_score': '60', 'env_score': '60', 'social_score': '60', 'gov_score': '60',
                'market_value': '50000000', 'roa_lag': '1', 'roa_avg': '1'}
        # Align keys with the existing input form, not the research column names.
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            response = self.client.post('/ml-prediction', data=form)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('預測失敗', response.get_data(as_text=True))
        self.assertTrue(self.module.get_feature_importance_rows())
        self.assertEqual(list(self.module.roa_model.feature_names_in_), FUTURE)


if __name__ == '__main__':
    unittest.main()
