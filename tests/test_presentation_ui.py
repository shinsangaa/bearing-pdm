"""Presentation controls, real events and verified bundle availability."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from app import (verified_demo_cases, prepare_alarm_frame, compute_alarm_thresholds,
                 sustained_alarm_states)
from src.cloud_waterfall import BUNDLE_DIR

ROOT = Path(__file__).resolve().parents[1]


class PresentationUITests(unittest.TestCase):
    def start(self, app):
        next(b for b in app.button if b.label == '▶ 시뮬레이션 시작').click().run(timeout=60)
        self.assertFalse(app.exception)
        return app.session_state['ims_alarm_result']

    def test_default_and_four_real_cases_locally(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 1)
        selector = app.selectbox(key='ims_demo_case')
        self.assertEqual(selector.value, '3rd_test / Bearing 3 — 외륜 결함')
        cases = verified_demo_cases()
        self.assertEqual([(c['experiment'], c['bearing'], c['sensor']) for c in cases],
                         [('1st_test', 3, 1), ('1st_test', 4, 1), ('2nd_test', 1, 1), ('3rd_test', 3, 1)])
        self.assertEqual(selector.options, [c['label'] for c in cases])
        self.assertFalse(next(e for e in app.expander if e.label == '상세 분석 정보').proto.expanded)
        for case in cases:
            with self.subTest(case=case['label']):
                app.selectbox(key='ims_demo_case').select(case['label']).run(timeout=60)
                result = self.start(app)
                self.assertEqual(result['selection'], (case['experiment'], case['bearing'], case['sensor'], 5, 3, 'Alarm'))
                frame = prepare_alarm_frame(case['experiment'], case['bearing'], case['sensor'])
                thresholds = compute_alarm_thresholds(frame, 5)
                event = sustained_alarm_states(frame, thresholds, 3)[1]['Alarm']
                self.assertEqual(result['trigger_index'], event)
                self.assertEqual(result['time'], frame.iloc[event].measured_at)
                self.assertEqual(result['rms'], frame.iloc[event].rms)
                self.assertNotIn('source', result['waterfall'])  # local TXT computation
                self.assertEqual(result['waterfall']['times'][-1], result['time'])
                charts = app.get('plotly_chart')
                self.assertEqual(len(charts), 3)
                replay = json.loads(charts[0].proto.spec)
                np.testing.assert_allclose(replay['data'][0]['y'], frame.iloc[max(0, event-47):event+1].rms)
                self.assertEqual(replay['data'][0]['x'][-1], result['time'].isoformat())
                self.assertEqual([s['y0'] for s in replay['layout']['shapes']],
                                 [thresholds[k] for k in ('warning', 'alarm', 'critical')])
                self.assertEqual(json.loads(charts[-1].proto.spec)['data'][0]['type'], 'surface')
                self.assertTrue(any('Alarm 임계값=' in e.value for e in app.error))
                self.assertTrue(any('MAD × 5 | 연속 3회 초과 | Alarm | Sensor 1' in c.value for c in app.caption))
                details = next(e for e in app.expander if e.label == '상세 분석 정보')
                self.assertEqual(len(details.get('json')), len(app.get('json')))
                self.assertEqual(len(app.get('json')), 1)

    def test_cloud_default_and_all_presentation_cases(self):
        actual_is_file = Path.is_file
        def offline(path):
            return False if path.is_relative_to(ROOT / 'data/raw') else actual_is_file(path)
        with patch.object(Path, 'is_file', offline), patch(
                'src.preprocessing.read_measurement', side_effect=AssertionError('cloud TXT read')):
            app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
            self.assertIn('3rd_test / Bearing 3', app.selectbox[0].value)
            result = self.start(app)
            self.assertEqual(result['selection'], ('3rd_test', 3, 1, 5, 3, 'Alarm'))
            for label in app.selectbox[0].options:
                app.selectbox[0].select(label).run(timeout=60)
                result = self.start(app)
                self.assertEqual(result['waterfall']['source'], 'precomputed_actual_txt')
                self.assertEqual(json.loads(app.get('plotly_chart')[-1].proto.spec)['data'][0]['type'], 'surface')

    def test_missing_and_invalid_bundles_are_not_selectable(self):
        # Only Test 1 B3 S2 remains valid: auto-select S2; omit every other case.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for sensor in (1, 2):
                name = f'1st_test_b3_s{sensor}_mad5_n3_alarm'
                (target / (name + '.npz')).write_bytes((BUNDLE_DIR / (name + '.npz')).read_bytes())
                metadata = json.loads((BUNDLE_DIR / (name + '.json')).read_text(encoding='utf-8'))
                if sensor == 1:
                    metadata['bundle_sha256'] = 'damaged'
                (target / (name + '.json')).write_text(json.dumps(metadata), encoding='utf-8')
            with patch('src.cloud_waterfall.BUNDLE_DIR', target):
                cases = verified_demo_cases()
                self.assertEqual([(c['experiment'], c['bearing'], c['sensor']) for c in cases], [('1st_test', 3, 2)])
                app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
                self.assertEqual(app.selectbox[0].options, [cases[0]['label']])
                self.assertFalse(app.exception)
            with patch('src.cloud_waterfall.BUNDLE_DIR', target / 'absent'):
                app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
                self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox), 0)
                self.assertFalse(any('시뮬레이션 시작' in b.label for b in app.button))
                self.assertTrue(any('시연 미지원' in w.value for w in app.warning))
                self.assertEqual(len(app.get('plotly_chart')), 0)


if __name__ == '__main__':
    unittest.main()
