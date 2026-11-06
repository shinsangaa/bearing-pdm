"""경계 사례는 독립 시험 입력으로 확인하며 실측 CSV는 수정하지 않는다."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.ims_alarm import (compute_alarm_thresholds, find_first_alarm, sustained_alarm_states,
                           validate_alarm_frame, compare_series)


def sample(n=200):
    return pd.DataFrame(dict(measured_at=pd.date_range("2000-01-01", periods=n, freq="10min"),
                             rms=np.ones(n)))


class AlarmTests(unittest.TestCase):
    def test_initial_tenth_cap_and_no_future_leakage(self):
        for size, expected in ((101, 11), (984, 99), (2156, 216), (6324, 300)):
            frame = sample(size)
            thresholds = compute_alarm_thresholds(frame)
            self.assertEqual(thresholds["baseline_n"], expected)
            frame.loc[expected:, "rms"] = 1000
            self.assertEqual(compute_alarm_thresholds(frame), thresholds)

    def test_peak_rejected_and_confirmation_at_nth(self):
        frame = sample()
        thresholds = compute_alarm_thresholds(frame)
        frame.loc[25, "rms"] = 5
        for duration in (3, 4, 5):
            self.assertIsNone(find_first_alarm(frame, thresholds, "Alarm", duration))
        self.assertEqual(find_first_alarm(frame, thresholds, "Alarm", 1), 25)
        frame.loc[40:44, "rms"] = 5
        for duration in (3, 4, 5):
            self.assertEqual(find_first_alarm(frame, thresholds, "Alarm", duration), 39+duration)

    def test_independent_levels_and_latch(self):
        frame = sample()
        thresholds = compute_alarm_thresholds(frame)
        frame.loc[30:34, "rms"] = thresholds['warning'] + .01
        frame.loc[35, "rms"] = thresholds['critical'] + 1
        states, events = sustained_alarm_states(frame, thresholds, 3)
        self.assertEqual(events, dict(Warning=32, Alarm=None, Critical=None))
        self.assertEqual(states[-1], "Warning")

    def test_invalid_and_gaps_break_run_without_removing_rows(self):
        for invalid in (np.nan, np.inf, -1, 0, "bad"):
            frame = sample().astype({'rms': object})
            frame.loc[30:35, "rms"] = 5
            frame.loc[32, "rms"] = invalid
            checked = validate_alarm_frame(frame)
            self.assertEqual(len(checked), 200)
            self.assertEqual(find_first_alarm(checked, compute_alarm_thresholds(checked), "Alarm", 3), 35)
        frame = sample()
        frame.loc[30:34, "rms"] = 5
        frame.loc[32:, "measured_at"] += pd.Timedelta(hours=5)
        self.assertEqual(find_first_alarm(frame, compute_alarm_thresholds(frame), "Alarm", 3), 34)

    def test_bad_baseline_and_timestamp_validation(self):
        frame = sample()
        frame.loc[:5, "rms"] = np.nan
        with self.assertRaises(ValueError):
            compute_alarm_thresholds(frame)
        frame = sample()
        frame.loc[0, "rms"] = np.nan
        thresholds = compute_alarm_thresholds(frame)
        self.assertEqual(thresholds['baseline_n'], 20)
        self.assertEqual(thresholds['baseline_valid_n'], 19)
        for value in (pd.NaT, frame.loc[1, "measured_at"]):
            frame.loc[0, "measured_at"] = value
            with self.assertRaises(ValueError):
                validate_alarm_frame(frame)

    def test_initial_drift_and_outlier_diagnostics(self):
        frame = sample()
        frame.loc[12:19, 'rms'] = 1.5
        thresholds = compute_alarm_thresholds(frame)
        self.assertGreater(thresholds['baseline_drift_scale'], 3)
        frame = sample()
        frame.loc[2, 'rms'] = 50
        self.assertEqual(compute_alarm_thresholds(frame)['baseline_outlier_n'], 1)
        self.assertEqual(compute_alarm_thresholds(frame)['baseline_valid_n'], 20)

    def test_comparison_rows_and_end_duration(self):
        frame = sample()
        frame.loc[100:, 'rms'] = 100
        table = compare_series(frame, 'unknown', 9, 8)
        self.assertEqual(len(table), 27)
        self.assertTrue(table.alarm_occurred.all())
        event = table.loc[(table.level == 'Alarm') & (table.consecutive == 5)].iloc[0]
        self.assertEqual(event.first_alarm_at, frame.iloc[104].measured_at)
        self.assertAlmostEqual(event.remaining_calendar_hours, 95/6)
        # 동일 값은 초과가 아님.
        frame = sample()
        thresholds = compute_alarm_thresholds(frame)
        frame.loc[30:, 'rms'] = thresholds['alarm']
        self.assertIsNone(find_first_alarm(frame, thresholds, 'Alarm'))


class AlarmAppTests(unittest.TestCase):
    def test_replay_first_point_is_visible_and_gaps_stay_missing(self):
        from app import replay_rms_figure
        frame = sample()
        thresholds = compute_alarm_thresholds(frame)
        first = json.loads(replay_rms_figure(frame.iloc[:1], thresholds).to_json())
        self.assertEqual(first['data'][0]['mode'], 'lines+markers')
        self.assertEqual(first['data'][0]['x'], [frame.iloc[0].measured_at.isoformat()])
        self.assertEqual(first['data'][0]['y'], [1.0])
        frame.loc[1, 'rms'] = np.nan
        gap = json.loads(replay_rms_figure(frame.iloc[:3], thresholds).to_json())
        self.assertEqual(gap['data'][0]['y'], [1.0, None, 1.0])
        self.assertFalse(gap['data'][0]['connectgaps'])

    def test_all_plots_and_four_measurement_selection(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        app.checkbox(key='ims_analysis_mode').check().run(timeout=60)
        app.selectbox[5].select(4)
        app.selectbox[6].select('즉시 (대기 없음)')
        next(button for button in app.button if '시뮬레이션 시작' in button.label).click().run(timeout=90)
        self.assertFalse(app.exception)
        selected = app.dataframe[0].value
        self.assertEqual(len(selected), 48)
        self.assertEqual(set(selected.consecutive), {4})
        next(c for c in app.checkbox if '전체 센서 RMS' in c.label).check().run(timeout=60)
        self.assertFalse(app.exception)
        # 전체 센서 16개와 선택 결과의 RMS/FFT 2개.
        self.assertEqual(len(app.get('plotly_chart')), 18)

    def test_actual_replay_latch_and_no_raw_reads_on_rerun(self):
        with patch('src.preprocessing.read_measurement', side_effect=AssertionError('unexpected raw read')):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
            app.checkbox(key='ims_analysis_mode').check().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('plotly_chart')), 0)
            app.selectbox[0].select('1st_test').run(timeout=60)
            self.assertEqual(app.selectbox[2].options, ['1', '2'])
            app.selectbox[2].select(2).run(timeout=60)
            self.assertFalse(app.exception)
        # 실측 재생과 FFT는 실제 파일을 사용한다.
        app.selectbox[0].select('2nd_test').run(timeout=60)
        app.selectbox[1].select(1).run(timeout=60)
        app.selectbox[6].select('즉시 (대기 없음)').run(timeout=60)
        next(button for button in app.button if '시뮬레이션 시작' in button.label).click().run(timeout=90)
        self.assertFalse(app.exception)
        result = app.session_state['ims_alarm_result']
        self.assertIsNotNone(result['trigger_index'])
        self.assertIsNotNone(result['waterfall'])
        self.assertGreaterEqual(len(result['waterfall']['times']), 2)
        self.assertEqual(result['time'], result['frame'].iloc[result['trigger_index']].measured_at)
        charts = app.get('plotly_chart')
        self.assertEqual(len(charts), 3)  # 재생 RMS, 전체 RMS, 실측 FFT
        replay = json.loads(charts[0].proto.spec)
        trace = replay['data'][0]
        last = result['trigger_index']
        expected = result['frame'].iloc[max(0, last-47):last+1]
        self.assertEqual(trace['x'], [pd.Timestamp(t).isoformat() for t in expected.measured_at])
        np.testing.assert_allclose(trace['y'], expected.rms)
        self.assertEqual(trace['mode'], 'lines+markers')
        self.assertEqual(replay['layout']['xaxis']['type'], 'date')
        self.assertEqual([s['y0'] for s in replay['layout']['shapes']],
                         [result['thresholds'][k] for k in ('warning', 'alarm', 'critical')])
        self.assertEqual(json.loads(charts[2].proto.spec)['data'][0]['type'], 'surface')
        details = next(e for e in app.expander if e.label == '상세 분석 정보')
        self.assertFalse(details.proto.expanded)
        self.assertEqual(len(details.get('json')), 1)
        self.assertEqual(len(app.get('json')), 1)
        with patch('src.preprocessing.read_measurement', side_effect=AssertionError('rerun read raw')):
            app.run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['ims_alarm_result']['time'], result['time'])

    def test_peak_does_not_read_txt(self):
        import streamlit as st
        st.cache_data.clear()
        frame = sample()
        frame.loc[25, 'rms'] = 5
        frame = frame.assign(bearing_id=3, sensor_within_bearing=1, source_path='absent')
        with patch('src.demo_support.load_monitoring', return_value=frame), patch(
                'src.preprocessing.read_measurement', side_effect=AssertionError('peak must not read TXT')):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
            app.checkbox(key='ims_analysis_mode').check().run(timeout=60)
            app.selectbox[6].select('즉시 (대기 없음)').run(timeout=60)
            next(button for button in app.button if '시뮬레이션 시작' in button.label).click().run(timeout=60)
            self.assertFalse(app.exception)
            result = app.session_state['ims_alarm_result']
            self.assertIsNone(result['trigger_index'])
            self.assertIsNone(result['waterfall'])
            self.assertEqual(result['level'], 'Normal')
        st.cache_data.clear()


if __name__ == '__main__':
    unittest.main()
