import json
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from app import (prepare_alarm_frame, compute_alarm_thresholds, sustained_alarm_states,
                 build_alarm_event_waterfall, compute_fft_spectrum)
from src.cloud_waterfall import load_cloud_waterfall, BUNDLE_DIR, cloud_support_message
from src.preprocessing import resolve_source_path, read_measurement

ROOT=Path(__file__).resolve().parents[1]
CASES = [('1st_test', 3, 1), ('1st_test', 3, 2), ('1st_test', 4, 1),
         ('1st_test', 4, 2), ('2nd_test', 1, 1), ('3rd_test', 3, 1)]


class CloudWaterfallTests(unittest.TestCase):
    def test_actual_fft_bundle_matches_original_and_rejects_other_events(self):
        for experiment,bearing,sensor in CASES:
            frame=prepare_alarm_frame(experiment,bearing,sensor)
            thresholds=compute_alarm_thresholds(frame,5)
            event=sustained_alarm_states(frame,thresholds,3)[1]['Alarm']
            original=build_alarm_event_waterfall(experiment,bearing,sensor,frame,event)
            args=(experiment,bearing,sensor,frame,event,thresholds,3,'Alarm',build_alarm_event_waterfall,compute_fft_spectrum)
            cached=load_cloud_waterfall(*args)
            np.testing.assert_array_equal(cached['spectra'],original['spectra'])
            np.testing.assert_array_equal(cached['frequency'],original['frequency'])
            np.testing.assert_array_equal(cached['hours'],original['hours'])
            self.assertEqual(cached['times'],original['times'])
            self.assertFalse(original['missing'])
            for source in cached['provenance']:
                raw = resolve_source_path(source['path'])
                self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), source['sha256'])
                signal = read_measurement(raw, experiment)[:, cached['channel'] - 1]
                self.assertAlmostEqual(float(np.sqrt(np.mean(signal ** 2))), source['txt_rms'])
            changed=frame.copy();changed.loc[0,'rms']*=2
            with self.assertRaises(ValueError):
                load_cloud_waterfall(experiment,bearing,sensor,changed,event,thresholds,3,'Alarm',build_alarm_event_waterfall,compute_fft_spectrum)
            with self.assertRaises(ValueError):
                load_cloud_waterfall(experiment,bearing,sensor,frame,None,thresholds,3,'Alarm',build_alarm_event_waterfall,compute_fft_spectrum)
            with self.assertRaises(ValueError):
                load_cloud_waterfall(experiment,bearing,sensor,frame,event,thresholds,5,'Alarm',build_alarm_event_waterfall,compute_fft_spectrum)
            for changed_event, multiplier, level in ((event + 1, 5, 'Alarm'),
                                                      (event, 7, 'Alarm'), (event, 5, 'Warning')):
                with self.assertRaises(ValueError):
                    load_cloud_waterfall(experiment, bearing, sensor, frame, changed_event,
                                         dict(thresholds, multiplier=multiplier), 3, level,
                                         build_alarm_event_waterfall, compute_fft_spectrum)

    def test_cloud_app_without_any_raw_txt_uses_verified_fft(self):
        actual_is_file=Path.is_file
        def offline(path):
            return False if path.is_relative_to(ROOT/'data/raw') else actual_is_file(path)
        with patch.object(Path,'is_file',offline), patch('src.preprocessing.read_measurement',side_effect=AssertionError('cloud raw read')):
            for experiment, bearing, sensor in CASES:
                with self.subTest(experiment=experiment, bearing=bearing, sensor=sensor):
                    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
                    app.checkbox(key='ims_analysis_mode').check().run(timeout=60)
                    app.selectbox[0].select(experiment).run(timeout=60)
                    app.selectbox[1].select(bearing).run(timeout=60)
                    app.selectbox[2].select(sensor).run(timeout=60)
                    app.selectbox[6].select(app.selectbox[6].options[-1])
                    next(button for button in app.button if '시뮬레이션 시작' in button.label).click().run(timeout=60)
                    self.assertFalse(app.exception)
                    result=app.session_state['ims_alarm_result']
                    self.assertEqual(result['waterfall']['source'],'precomputed_actual_txt')
                    self.assertEqual(len(result['waterfall']['times']),16)
                    self.assertEqual(json.loads(app.get('plotly_chart')[-1].proto.spec)['data'][0]['type'],'surface')
                    self.assertTrue(any('실제 NASA TXT' in info.value for info in app.info))
            app.selectbox[5].select(4)
            next(button for button in app.button if '시뮬레이션 시작' in button.label).click().run(timeout=60)
            self.assertFalse(app.exception)
            result = app.session_state['ims_alarm_result']
            self.assertIsNone(result['waterfall'])
            self.assertIn(cloud_support_message(), result['waterfall_error'])
            self.assertNotIn('data/raw', result['waterfall_error'])

    def test_tampered_bundles_are_rejected(self):
        experiment, bearing, sensor = CASES[0]
        frame = prepare_alarm_frame(experiment, bearing, sensor)
        thresholds = compute_alarm_thresholds(frame, 5)
        event = sustained_alarm_states(frame, thresholds, 3)[1]['Alarm']
        original_path = BUNDLE_DIR / '1st_test_b3_s1_mad5_n3_alarm.npz'
        original_metadata = json.loads(original_path.with_suffix('.json').read_text(encoding='utf-8'))
        with np.load(original_path, allow_pickle=False) as bundle:
            original_arrays = {key: bundle[key] for key in bundle.files}
        for damage in ('algorithm', 'sha256', 'times', 'hours', 'frequency'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'test.npz'
                metadata = json.loads(json.dumps(original_metadata))
                arrays = {key: value.copy() for key, value in original_arrays.items()}
                if damage == 'algorithm':
                    metadata['algorithm_sha256'] = 'changed'
                elif damage == 'times':
                    metadata['times'][-1] = metadata['times'][-2]
                elif damage == 'hours':
                    arrays['hours'][-1] += 1
                elif damage == 'frequency':
                    arrays['frequency'][1] = arrays['frequency'][0]
                np.savez_compressed(path, **arrays)
                metadata['bundle_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                if damage == 'sha256':
                    metadata['bundle_sha256'] = 'changed'
                path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
                with patch('src.cloud_waterfall.BUNDLE_DIR', Path(directory)), self.assertRaises(ValueError):
                    load_cloud_waterfall(experiment, bearing, sensor, frame, event, thresholds, 3,
                                         'Alarm', build_alarm_event_waterfall, compute_fft_spectrum)


if __name__=='__main__':
    unittest.main()
