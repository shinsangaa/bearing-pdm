import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.train_ims_fault_types import (FEATURES, CLASSES, load_features, assign_blocks, metrics)

ROOT = Path(__file__).resolve().parents[1]


class FaultTypesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = load_features()
        cls.labeled, cls.audit = assign_blocks(cls.frame)

    def test_sensor_metadata_exclusion_and_purged_blocks(self):
        self.assertEqual(set(self.frame.test_id), {'1st_test', '2nd_test', '3rd_test'})
        self.assertEqual(set(self.frame.sensor_within_bearing), {1})
        self.assertFalse(set(FEATURES) & {'test_id','bearing_id','measured_at','measurement_index',
                                        'rul_calendar_hours_proxy','source_path','end_defect'})
        labeled = self.labeled
        self.assertEqual(labeled.groupby(['test_id','source_path']).partition.nunique().max(), 1)
        self.assertTrue(labeled.loc[labeled.region.eq('middle'), 'partition'].eq('excluded_middle').all())
        for (_, _, _), group in labeled.loc[labeled.region.ne('middle')].groupby(['test_id','bearing_id','region']):
            train, test = [group.loc[group.partition.eq(p)] for p in ('train','test')]
            self.assertGreater(test.position.min()-train.position.max(), 6)
            self.assertGreaterEqual((test.measured_at.min()-train.measured_at.max()).total_seconds(), 3600)
        shuffled, _ = assign_blocks(self.frame.sample(frac=1, random_state=42))
        keys = ['test_id','bearing_id','measured_at']
        pd.testing.assert_frame_equal(labeled.sort_values(keys).reset_index(drop=True), shuffled.sort_values(keys).reset_index(drop=True))

    def test_saved_model_metrics_and_train_only_preprocessing(self):
        out = ROOT / 'reports/ims_fault_types'
        bundle = joblib.load(out / 'rf_fault_types.joblib')
        pipeline = bundle['pipeline']
        self.assertIsInstance(pipeline[-1], RandomForestClassifier)
        self.assertEqual(bundle['features'], FEATURES)
        train, test = [self.labeled.loc[self.labeled.partition.eq(p)] for p in ('train','test')]
        np.testing.assert_allclose(pipeline[0].statistics_, train[FEATURES].median())
        prediction = pipeline.predict(test[FEATURES])
        saved = pd.read_csv(out / 'test_predictions.csv')
        np.testing.assert_array_equal(prediction, saved.prediction)
        report = json.loads((out / 'evaluation.json').read_text(encoding='utf-8'))
        measured = metrics(test.label, prediction)
        self.assertEqual(measured, report['metrics'])
        self.assertEqual(set(train.label), set(CLASSES))
        self.assertEqual(set(test.label), set(CLASSES))
        self.assertEqual(report['strict_forward_diagnostic']['train_classes'], ['Normal proxy'])
        self.assertFalse(report['strict_forward_diagnostic']['valid_four_class_validation'])
        for audit in report['split_audit']:
            self.assertGreaterEqual(audit['gap_rows'], 6)
            self.assertGreaterEqual(audit['actual_gap_seconds'], 3600)

    def test_missing_test_and_duplicate_sensor_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            self.frame.loc[self.frame.test_id.ne('3rd_test')].to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_features(path)
            pd.concat([self.frame,self.frame.iloc[:1]]).to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_features(path)

    def test_bilstm_recovered_metrics_and_provenance(self):
        out = ROOT / 'reports/bilstm_rul'
        report = json.loads((out / 'restoration_verification.json').read_text(encoding='utf-8'))
        self.assertTrue(report['passed'])
        self.assertFalse(report['retrained'])
        self.assertEqual(report['prediction_rows'], 676)
        self.assertAlmostEqual(report['measured_bilstm']['MAE_hours'], 57.49865536159939)
        self.assertGreater(report['measured_bilstm']['MAE_hours'], report['measured_last_train_target']['MAE_hours'])
        import hashlib
        for item in json.loads((out / 'restoration_manifest.json').read_text(encoding='utf-8')):
            self.assertEqual(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest(), item['sha256'])


if __name__ == '__main__':
    unittest.main()
