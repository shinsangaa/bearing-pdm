"""NASA 전처리·저장 모델·대시보드를 검사한다. 모델은 학습하지 않는다."""
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import io
import json
import math
import csv
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from streamlit.testing.v1 import AppTest

from src.demo_support import ROOT, load_monitoring, load_report, reference_status, baseline_predictions
from src.preprocessing import RAW_DIR, BEARING_CHANNELS, EXPERIMENTS, build_timeline, discover_measurements, feature_rows, read_measurement, process_experiment, resolve_source_path
from src.train_rul_baseline import load_measurements, MODEL_FEATURES, TARGET


class IMSTests(unittest.TestCase):
    """외부 원본 데이터 없이 잘못된 연결과 시간 라벨을 탐지한다."""

    def records(self):
        start = datetime(2003, 10, 22, 12)
        return [(start + timedelta(seconds=offset), Path(str(i)))
                for i, offset in enumerate((0, 300, 900, 4500))]

    def test_gap_and_calendar_are_separate(self):
        rows, gaps = build_timeline(self.records(), 1200)
        self.assertEqual(rows[0]["remaining_measurements"], 3)
        self.assertEqual(rows[0]["remaining_calendar_hours"], 1.25)
        self.assertEqual(rows[0]["remaining_gap_excluded_hours_proxy"], 0.25)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(rows[-1]["segment_id"], 1)
        self.assertEqual(rows[-1]["remaining_measurements"], 0)
        self.assertIsNone(rows[-1]["rul_operating_hours"])

    def test_mapping_keeps_both_sensors(self):
        self.assertEqual(BEARING_CHANNELS, {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, 8)})
        # 각 열에 채널 번호를 채워 RMS가 5, 6, 7, 8로 나오는지 검사한다.
        data = np.tile(np.arange(1, 9, dtype=float), (20480, 1))
        timeline, _ = build_timeline(self.records())
        rows = feature_rows(Path("sample"), data, timeline[0])
        self.assertEqual([(r["bearing_id"], r["channel_id"], r["array_column"], r["rms"]) for r in rows],
                         [(3, 5, 4, 5), (3, 6, 5, 6), (4, 7, 6, 7), (4, 8, 7, 8)])
        self.assertEqual({r["rul_count_proxy"] for r in rows}, {3})

    def test_order_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "1st_test"
            root.mkdir()
            for name in ("2003.10.22.12.10.00", "2003.10.22.12.00.00"):
                (root / name).touch()
            records, _ = discover_measurements(root)
            self.assertLess(records[0][0], records[1][0])
            nested = root / "duplicate"
            nested.mkdir()
            (nested / records[0][1].name).touch()
            with self.assertRaises(ValueError):
                discover_measurements(root)

    def test_wrong_channel_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad"
            np.savetxt(path, np.ones((10, 4)))
            with self.assertRaises(ValueError):
                read_measurement(path)

    def test_nonascending_timestamps(self):
        with self.assertRaises(ValueError):
            build_timeline(list(reversed(self.records())))

class ExperimentTests(unittest.TestCase):
    def test_all_channel_maps_and_censored_labels(self):
        timeline, _ = build_timeline([(datetime(2004, 1, 1) + timedelta(minutes=i * 10), Path(str(i))) for i in range(3)])
        for experiment, columns, expected_rows, failures in [
                ("1st_test", 8, 8, {3, 4}), ("2nd_test", 4, 4, {1}), ("3rd_test", 4, 4, {3})]:
            with self.subTest(experiment=experiment):
                data = np.tile(np.arange(1, columns + 1, dtype=float), (20480, 1))
                rows = feature_rows(Path("source"), data, timeline[0], experiment, True)
                self.assertEqual(len(rows), expected_rows)
                self.assertEqual({r["bearing_id"] for r in rows}, {1, 2, 3, 4})
                for row in rows:
                    self.assertEqual(row["rms"], row["channel_id"])
                    self.assertEqual(row["rul_calendar_hours_proxy"] is not None, row["bearing_id"] in failures)
                    self.assertIsNone(row["rul_operating_hours"])

    def test_deployment_nested_paths_and_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for experiment, nested, columns in [("1st_test", "1st_test", 8), ("2nd_test", "2nd_test", 4), ("3rd_test", "", 4)]:
                folder = root / experiment / nested
                folder.mkdir(parents=True)
                path = folder / "2004.03.04.09.27.46"
                np.savetxt(path, np.ones((20480, columns)))
                records, _ = discover_measurements(root, experiment)
                self.assertEqual(len(records), 1)
                self.assertEqual(read_measurement(path, experiment).shape, (20480, columns))

    def test_third_period_proxy_and_complete_file_count(self):
        start = datetime(2004, 3, 4, 9, 27, 46)
        end = datetime(2004, 4, 18, 2, 42, 55)
        timeline, _ = build_timeline([(start, Path('first')), (end, Path('last'))])
        rows = feature_rows(Path('first'), np.ones((20480, 4)), timeline[0], '3rd_test', True)
        for row in rows:
            if row['bearing_id'] == 3:
                self.assertEqual(row['end_defect'], 'outer_race')
                self.assertAlmostEqual(row['rul_calendar_hours_proxy'], (end-start).total_seconds()/3600)
            else:
                self.assertIsNone(row['rul_calendar_hours_proxy'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'3rd_test'
            root.mkdir()
            (root/start.strftime('%Y.%m.%d.%H.%M.%S')).touch()
            with self.assertRaises(ValueError):
                process_experiment(Path(temp), '3rd_test', Path(temp)/'result.csv')
            self.assertFalse((Path(temp)/'result.csv').exists())

    def test_reviewed_quality_records_keep_features_and_proxy_labels(self):
        from src.preprocessing import QUALITY_SUSPECT_TIMES, QUALITY_FLAG
        for experiment, filenames in QUALITY_SUSPECT_TIMES.items():
            for filename in filenames:
                timeline, _ = build_timeline([(datetime.strptime(filename, '%Y.%m.%d.%H.%M.%S'), Path(filename))])
                rows = feature_rows(Path(filename), np.ones((20480, 4)), timeline[0], experiment, True)
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row['quality_flag'] == QUALITY_FLAG and row['rms'] == 1 for row in rows))
                self.assertEqual(sum(row['rul_calendar_hours_proxy'] == 0 for row in rows), 1)
                frame = pd.DataFrame(rows)
                self.assertEqual(reference_status(frame)['status'], '품질 주의: 상태 판단 보류')

    def test_reference_uses_only_initial_observations(self):
        frame = pd.DataFrame({"measured_at": pd.date_range("2004-01-01", periods=100, freq="10min"),
                              "rms": np.r_[np.linspace(1, 2, 50), np.repeat(100., 50)]})
        result = reference_status(frame)
        self.assertEqual(result["status"], "참고선 초과")
        self.assertEqual(result["baseline_rows"], 50)
        changed = frame.copy()
        changed.loc[50:, "rms"] *= 100
        self.assertEqual(reference_status(changed)["threshold"], result["threshold"])

    def test_training_defaults_refuse_overwriting_final_results(self):
        from src import train_rul_baseline, train_rf_enhanced, train_ridge_ims
        with patch.object(sys, "argv", [train_rul_baseline.__name__]), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                train_rul_baseline.main()
            self.assertEqual(error.exception.code, 2)
        for module in (train_rf_enhanced, train_ridge_ims):
            with self.subTest(module=module.__name__), self.assertRaises(FileExistsError):
                module.main()


class AppTests(unittest.TestCase):
    def test_offline_archive_independent_screens(self):
        actual_open, actual_is_file = Path.open, Path.is_file
        forbidden_reads = []

        def is_file(path):
            if path.as_posix().casefold().startswith(("c:/datasets/", RAW_DIR.as_posix().casefold() + "/")):
                return False
            return actual_is_file(path)

        def guarded_open(path, *args, **kwargs):
            normalized = path.as_posix().casefold()
            if normalized.startswith(("c:/datasets/", "c:/bearing-pdm_archive/", RAW_DIR.as_posix().casefold() + "/")):
                forbidden_reads.append(str(path))
                raise PermissionError("offline test denied external data/archive")
            return actual_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Path, "is_file", is_file):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            app.checkbox(key='ims_analysis_mode').check().run(timeout=60)
            self.assertEqual(app.selectbox[0].options, ["1st_test", "2nd_test", "3rd_test"])
            for experiment in EXPERIMENTS:
                app.selectbox[0].select(experiment).run(timeout=60)
                for bearing in [1, 2, 3, 4]:
                    app.selectbox[1].select(bearing).run(timeout=60)
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(len(app.dataframe[0].value), 48)
                    self.assertEqual(len(app.dataframe[1].value), 432)
            app.button(key='nav_home').click().run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            self.assertGreaterEqual(len(app.get('plotly_chart')), 1)
            self.assertEqual(len(app.dataframe), 2)  # RF 두 평가와 특징 추가 비교표
            self.assertEqual(len(app.get('image')), 1)
            app.button(key='nav_logs').click().run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            app.button(key='nav_user_info').click().run(timeout=60)
            self.assertEqual(len(app.exception), 0)
        self.assertEqual(forbidden_reads, [])


def verify_original_ims(input_root, feature_path):
    """원본 루트와 특징 경로를 받아 전체 행 및 처음·마지막 신호의 계산값을 검사한다."""
    records = []
    for path in (input_root / "1st_test").rglob("*"):
        if path.is_file():
            try:
                time = datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S")
            except ValueError:
                continue
            records.append((time, path.resolve()))
    records.sort()
    if not records:
        raise FileNotFoundError("원본 NASA IMS 측정 파일이 없습니다.")
    metadata = load_report('reports/evaluation/preprocessing.json')['experiments']['1st_test']
    threshold = metadata["gap_threshold_seconds"]
    elapsed_proxy, segments = [0.0], [0]
    for index in range(1, len(records)):
        interval = (records[index][0] - records[index - 1][0]).total_seconds()
        gap = interval > threshold
        elapsed_proxy.append(elapsed_proxy[-1] + (0 if gap else interval) / 3600)
        segments.append(segments[-1] + int(gap))
    with feature_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames
    # 공통 CSV에는 B1·2도 있다. 기존 모델에 해당하는 B3·4를 독립적으로 대조한다.
    rows = [row for row in rows if int(row['bearing_id']) in (3, 4)]
    if not rows or len(rows) % 4:
        raise ValueError("특징 행은 비어 있지 않은 4의 배수여야 합니다.")
    # 검증 실패 시 -O 옵션 여부와 관계없이 중단하도록 명시적인 예외를 사용한다.
    def check(condition, message):
        if not condition:
            raise ValueError(message)
    groups = [rows[i:i + 4] for i in range(0, len(rows), 4)]
    for index, group in enumerate(groups):
        time, path = records[index]
        check([(int(r["bearing_id"]), int(r["channel_id"]), int(r["array_column"])) for r in group]
              == [(3, 5, 4), (3, 6, 5), (4, 7, 6), (4, 8, 7)], "센서 매핑 또는 행 순서 오류")
        for row in group:
            check(resolve_source_path(row["source_path"]) == path, "원본 경로 순서 오류")
            check(datetime.fromisoformat(row["measured_at"]) == time, "측정 시각 오류")
            check(int(row["measurement_index"]) == index, "측정 순번 오류")
            check(int(row["remaining_measurements"]) == len(records) - index - 1, "남은 횟수 오류")
            check(math.isclose(float(row["remaining_calendar_hours"]),
                               (records[-1][0] - time).total_seconds() / 3600), "남은 달력 시간 오류")
            check(row["rul_count_proxy"] == row["remaining_measurements"], "횟수 대리 라벨 오류")
            check(row["rul_calendar_hours_proxy"] == row["remaining_calendar_hours"], "달력 대리 라벨 오류")
            check(row["elapsed_operating_hours"] == row["rul_operating_hours"] == "", "실제 가동시간을 임의 생성함")
            check(int(row["sampling_hz"]) == 20000 and int(row["n_samples"]) == 20480, "측정 조건 오류")
            check(int(row["segment_id"]) == segments[index], "공백별 구간 번호 오류")
            check(math.isclose(float(row["elapsed_gap_excluded_hours_proxy"]), elapsed_proxy[index], abs_tol=1e-9), "공백 제외 경과 시간 오류")
            check(math.isclose(float(row["remaining_gap_excluded_hours_proxy"]), elapsed_proxy[-1] - elapsed_proxy[index], abs_tol=1e-9), "공백 제외 잔여 시간 오류")
    # 처음과 마지막 처리 파일의 원시값만 다시 읽는다. 전체 원시 데이터를 모으지 않는다.
    checked_signals = 0
    for group in (groups[0], groups[-1]):
        data = np.loadtxt(resolve_source_path(group[0]["source_path"]))
        check(data.shape == (20480, 8), "원본 크기 오류")
        for row in group:
            x = data[:, int(row["channel_id"]) - 1]
            # RMS는 math.fsum, 첨도는 SciPy 구현으로 기존 함수와 독립적으로 계산한다.
            rms = math.sqrt(math.fsum(float(v) ** 2 for v in x) / len(x))
            values = dict(rms=rms, std=float(np.std(x)),
                          kurtosis=float(kurtosis(x, fisher=False, bias=True)),
                          crest_factor=float(np.abs(x).max() / rms))
            for key, value in values.items():
                check(math.isclose(float(row[key]), value, rel_tol=1e-10, abs_tol=1e-12), f"{key} 계산 불일치")
            checked_signals += 1
        del data
    report = dict(status="PASS", discovered_files=len(records), processed_files=len(groups),
                  feature_rows=len(rows), columns=len(columns), endpoint=records[-1][0].isoformat(),
                  independently_checked_signals=checked_signals,
                  bearings=[3, 4], channels=[5, 6, 7, 8])
    return report


class SavedProjectTests(unittest.TestCase):
    def test_experiment_holdout_and_common_features(self):
        from src.train_rul_baseline import load_experiment_splits, COMMON_FEATURES
        splits, audit = load_experiment_splits()
        expected = {'train':('1st_test',{3,4},4312),'validation':('2nd_test',{1},984),'test':('3rd_test',{3},6324)}
        for part,(experiment,bearings,rows) in expected.items():
            self.assertEqual(set(splits[part].test_id),{experiment})
            self.assertEqual(set(splits[part].bearing_id),bearings)
            self.assertEqual(len(splits[part]),rows)
        self.assertEqual(int(splits['validation'].quality_flag.ne('unflagged').sum()),2)
        self.assertEqual(int(splits['test'].quality_flag.ne('unflagged').sum()),1)
        self.assertTrue(audit['experiment_ids_disjoint'])
        self.assertTrue(audit['source_files_disjoint'])
        self.assertEqual(len(COMMON_FEATURES),12)
        self.assertFalse(set(COMMON_FEATURES)&{'test_id','bearing_id','measured_at','measurement_index','rul_calendar_hours_proxy','quality_flag'})
        frame=load_monitoring('1st_test')
        first=splits['train'].iloc[0]
        sensors=frame.loc[frame.bearing_id.eq(first.bearing_id)&frame.measured_at.eq(first.measured_at)]
        np.testing.assert_allclose(first[COMMON_FEATURES].to_numpy(dtype=float),sensors[COMMON_FEATURES].mean(),rtol=0,atol=1e-12)

    def test_experiment_loader_rejects_mixed_ids_and_censored_labels(self):
        from src.train_rul_baseline import load_experiment_splits
        frames={name:pd.read_csv(ROOT/f'data/processed/{name}.csv') for name in EXPERIMENTS}
        def read(path,**kwargs):
            return frames[Path(path).stem].copy()
        frames['2nd_test'].loc[0,'test_id']='1st_test'
        with patch('src.train_rul_baseline.pd.read_csv',side_effect=read):
            with self.assertRaises(ValueError):load_experiment_splits()
        frames['2nd_test'].loc[0,'test_id']='2nd_test'
        frames['1st_test'].loc[0,'rul_calendar_hours_proxy']=0
        with patch('src.train_rul_baseline.pd.read_csv',side_effect=read):
            with self.assertRaises(ValueError):load_experiment_splits()


    def test_relative_sources_and_historical_prediction_matching(self):
        from src.train_rul_baseline import DEFAULT_INPUT, chronological_split
        for experiment in EXPERIMENTS:
            frame = load_monitoring(experiment)
            paths = frame.source_path.drop_duplicates()
            self.assertTrue(all(not Path(value).is_absolute() and value.startswith('data/raw/') for value in paths))
            self.assertTrue(all(resolve_source_path(value).is_file() for value in paths))
        data, _ = load_measurements(DEFAULT_INPUT)
        reference = load_report('reports/evaluation/rf_basic_metrics.json')
        _, test, _, _ = chronological_split(data, reference['train_fraction'])
        saved = pd.read_csv(ROOT/'reports/predictions/rf_basic_predictions.csv')
        saved['measured_at'] = pd.to_datetime(saved['measured_at'])
        # 과거 저장 예측의 원본 경로는 당시의 절대 경로이므로 파일명으로 대조한다.
        for frame in (test, saved):
            frame['source_file'] = frame.source_path.str.replace('\\', '/', regex=False).str.rsplit('/', n=1).str[-1]
        joined = test.merge(saved, on=['measured_at', 'bearing_id', 'source_file'], validate='one_to_one')
        self.assertEqual(len(joined), 864)

    def test_shared_input_compatibility_rejects_modified_csv(self):
        from src.train_rul_baseline import DEFAULT_INPUT, validate_reference_input
        frame, _ = load_measurements(DEFAULT_INPUT)
        reference = load_report('reports/evaluation/rf_basic_metrics.json')
        validate_reference_input(DEFAULT_INPUT, frame, reference)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'changed.csv'
            modified = pd.read_csv(DEFAULT_INPUT)
            row = modified.index[modified.bearing_id.eq(3)][0]
            modified.loc[row, 'rms'] *= 2
            modified.to_csv(path,index=False)
            changed, _ = load_measurements(path)
            with self.assertRaises(ValueError):
                validate_reference_input(path, changed, reference)

    def test_training_preparation_uses_shared_csv_without_fitting(self):
        from src import train_rul_baseline
        with tempfile.TemporaryDirectory() as temp:
            for module, suffix, method in [(train_rul_baseline,'.joblib','fit')]:
                args = [module.__name__, '--model-output', str(Path(temp)/('model'+suffix)), '--report-dir', str(Path(temp)/module.__name__)]
                with self.subTest(module=module.__name__), patch.object(sys,'argv',args), patch('sys.stdout',io.StringIO()), patch('sklearn.pipeline.Pipeline.'+method,side_effect=RuntimeError('검증: 학습 직전 중단')):
                    with self.assertRaisesRegex(RuntimeError,'검증: 학습 직전 중단'):
                        module.main()
                self.assertFalse((Path(temp)/('model'+suffix)).exists())

    def test_saved_features_and_rul_input_format(self):
        self.assertEqual(list(EXPERIMENTS), ["1st_test", "2nd_test", "3rd_test"])
        with self.assertRaises(ValueError):
            load_monitoring("excluded_test")
        for experiment, rows, observations in [('1st_test',17248,2156),('2nd_test',3936,984),('3rd_test',25296,6324)]:
            frame = load_monitoring(experiment)
            self.assertEqual(len(frame),rows)
            self.assertEqual(frame.measured_at.nunique(),observations)
            self.assertEqual(set(frame.bearing_id),{1,2,3,4})
            self.assertFalse(frame.duplicated(['measured_at','bearing_id','sensor_within_bearing']).any())
            no_failure = ~frame.bearing_id.isin(EXPERIMENTS[experiment]['defects'])
            self.assertTrue(frame.loc[no_failure,'rul_calendar_hours_proxy'].isna().all())
            self.assertTrue(frame.loc[no_failure,'rul_count_proxy'].isna().all())
            self.assertTrue(frame.rul_operating_hours.isna().all())
            expected_marked = {'1st_test': 0, '2nd_test': 8, '3rd_test': 4}[experiment]
            self.assertEqual(int(frame.quality_flag.ne('unflagged').sum()), expected_marked)
        model_input, _ = load_measurements(ROOT/'data/processed/1st_test.csv')
        self.assertEqual(len(model_input),4312)
        self.assertEqual(len(MODEL_FEATURES),24)
        self.assertNotIn(TARGET,MODEL_FEATURES)
        self.assertTrue(np.isfinite(model_input[MODEL_FEATURES]).all().all())
        with tempfile.TemporaryDirectory() as temp:
            invalid = Path(temp)/'invalid.csv'
            pd.read_csv(ROOT/'data/processed/1st_test.csv').drop(columns=['rms']).to_csv(invalid,index=False)
            with self.assertRaises(ValueError):
                load_measurements(invalid)
            # 베어링 선택 전에 입력 전체의 실험 ID를 검사해야 한다.
            original = pd.read_csv(ROOT/'data/processed/1st_test.csv')
            for experiment in ('2nd_test', '3rd_test', 'excluded_test'):
                other = original.head(1).assign(test_id=experiment, bearing_id=1)
                pd.concat([original, other], ignore_index=True).to_csv(invalid, index=False)
                with self.assertRaises(ValueError):
                    load_measurements(invalid)

    def test_checkpoints_reproduce_saved_predictions(self):
        for predict, folder, column, tolerance in [
            (baseline_predictions,'rf_basic_predictions','prediction_random_forest',1e-10)]:
            actual = predict()
            saved = pd.read_csv(ROOT/f'reports/predictions/{folder}.csv',parse_dates=['measured_at'])
            keys = ['measured_at','bearing_id']
            joined = actual[keys+[column]].merge(saved[keys+[column]],on=keys,validate='one_to_one',suffixes=('_actual','_saved'))
            self.assertEqual(len(joined),len(actual))
            self.assertEqual(len(joined),len(saved))
            np.testing.assert_allclose(joined[column+'_actual'],joined[column+'_saved'],rtol=0,atol=tolerance)

    def test_holdout_rf_checkpoint_reproduces_full_test(self):
        import joblib
        from src.train_rul_baseline import COMMON_FEATURES
        saved = pd.read_csv(ROOT/'reports/predictions/rf_basic_holdout_test_predictions.csv')
        model = joblib.load(ROOT/'models/rf_basic_holdout.joblib')
        np.testing.assert_allclose(model.predict(saved[COMMON_FEATURES]), saved.prediction, rtol=0, atol=1e-10)
        self.assertEqual(len(saved), 6324)
        self.assertEqual(model.feature_names_in_.tolist(), COMMON_FEATURES)

    def test_flat_reports_and_separate_nasa_experiment_artifacts(self):
        report_root = ROOT/'reports'
        self.assertEqual({p.name for p in report_root.iterdir()},
                         {'evaluation','figures','predictions','ims_fault_types','bilstm_rul'})
        self.assertFalse(any(p.is_dir() for folder in report_root.iterdir() for p in folder.iterdir()))
        summary = load_report('reports/evaluation/summary.json')
        self.assertEqual(set(summary['models']), {'rf_basic','rf_engineered','ridge'})
        for details in summary['models'].values():
            self.assertTrue((report_root/details['evaluation']).is_file())
            if 'predictions' in details:
                self.assertTrue((report_root/details['predictions']).is_file())
            if 'model' in details:
                self.assertTrue((ROOT/details['model']).is_file())

    def test_original_data_models_evaluations_and_figures_unchanged(self):
        summary = load_report('reports/evaluation/summary.json')
        for row in summary['preserved_assets']:
            with self.subTest(path=row['path']):
                actual = hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()
                self.assertEqual(actual,row['sha256'])
        for path in (ROOT/'reports').rglob('*.json'):
            json.loads(path.read_text(encoding='utf-8-sig'))
        from PIL import Image
        for path in (ROOT/'reports').rglob('*.png'):
            with Image.open(path) as image:
                image.verify()
        for experiment in EXPERIMENTS:
            for name in summary['experiments'][experiment]['figures']:
                self.assertTrue((ROOT/'reports'/name).is_file())

    def test_report_writers_keep_three_folders_without_training(self):
        import joblib
        from src import train_rul_baseline
        from src.demo_support import write_summary
        rf = load_report('reports/evaluation/rf_basic_metrics.json')
        model = joblib.load(ROOT/'models/rf_basic.joblib')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'reports'
            with patch('sklearn.pipeline.Pipeline.fit',side_effect=AssertionError('검증 중 학습 금지')):
                train_rul_baseline.save_results(rf,
                    pd.read_csv(ROOT/'reports/predictions/rf_basic_predictions.csv',parse_dates=['measured_at']),
                    pd.read_csv(ROOT/'reports/evaluation/rf_basic_split_assignments.csv',parse_dates=['measured_at']),
                    model,output,Path(temp)/'models/rf.joblib')
            self.assertEqual({p.name for p in output.iterdir()},{'figures','evaluation','predictions'})
            for name, key, source in [('rf_basic_metrics','metrics',rf)]:
                saved = json.loads((output/f'evaluation/{name}.json').read_text(encoding='utf-8'))
                self.assertEqual(saved[key],source[key])
            summary = write_summary(output)
            self.assertEqual(set(summary['models']),{'rf_basic'})
            with self.assertRaises(FileExistsError):
                write_summary(output)
            self.assertFalse(any(p.name in {'metrics.csv','training_history.csv','evaluation.md','figures.json'} for p in output.rglob('*')))

    def test_original_ims_source(self):
        source = RAW_DIR
        if not source.is_dir():
            self.skipTest('원본 NASA IMS가 없어 독립 원본 대조는 미검증')
        report = verify_original_ims(source,ROOT/'data/processed/1st_test.csv')
        self.assertEqual(report['feature_rows'],8624)
        self.assertEqual(report['independently_checked_signals'],8)


if __name__ == '__main__':
    unittest.main()

