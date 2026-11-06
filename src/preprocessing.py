"""NASA IMS 실험·관측 구간별 센서 특징을 처리한다. 기존 결과는 덮어쓰지 않는다.

원본은 읽기 전용이다. 실제 가동시간과 정확한 고장 시각을 알 수 없으므로
달력 시간, 남은 측정 횟수, 긴 공백 제외 추정 시간을 서로 구분한다.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re

import numpy as np

if __package__:
    from .signal_processing import signal_features
else:
    from signal_processing import signal_features


# 동봉 IMS 설명서의 1st Test 매핑이다. 채널 번호는 1부터, 배열 열은 0부터 시작한다.
BEARING_CHANNELS = {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, 8)}
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
DATA_DIR = ROOT / "data/processed"
PREPROCESSING_REPORT = ROOT / "reports/evaluation/preprocessing.json"
END_DEFECTS = {3: "inner_race", 4: "rolling_element"}
EXPERIMENTS = {
    "1st_test": {"channels": BEARING_CHANNELS, "defects": END_DEFECTS, "columns": 8},
    "2nd_test": {"channels": {b: (b,) for b in range(1, 5)}, "defects": {1: "outer_race"}, "columns": 4},
    "3rd_test": {"channels": {b: (b,) for b in range(1, 5)}, "defects": {3: "outer_race"}, "columns": 4,
                 "file_count": 6324, "first": "2004.03.04.09.27.46", "last": "2004.04.18.02.42.55"},
}
THIRD_RELEASE_NOTE = "3rd_test는 전체 배포본·PHM 2021·Sensors 2024 기준 6,324개를 사용합니다. README의 4,448개 명세와 다릅니다. B3 RUL은 4월 18일 마지막 기록까지의 대리값이며 실제 고장 시각은 아닙니다."
# 네 채널이 함께 급락한 기록이다. 원인을 확정하거나 원본·특징 행을 삭제하지 않는다.
QUALITY_SUSPECT_TIMES = {
    "2nd_test": ("2004.02.19.06.12.39", "2004.02.19.06.22.39"),
    "3rd_test": ("2004.04.18.02.42.55",),
}
QUALITY_FLAG = "suspected_stop_or_measurement_change"
SAMPLING_HZ = 20000
EXPECTED_SAMPLES = 20480
TIME_FORMAT = "%Y.%m.%d.%H.%M.%S"
FILENAME_PATTERN = re.compile(r"\d{4}(?:\.\d{2}){5}")
FEATURE_NAMES = (
    "rms std peak kurtosis crest_factor fft_centroid_hz constant_signal outlier_8std_count "
    "fft_peak_1_hz fft_peak_1_amplitude fft_peak_2_hz fft_peak_2_amplitude "
    "fft_peak_3_hz fft_peak_3_amplitude"
).split()
TIME_FIELDS = (
    "measurement_index measured_at dataset_end_at elapsed_calendar_hours interval_seconds "
    "gap_before segment_id remaining_measurements remaining_calendar_hours "
    "elapsed_gap_excluded_hours_proxy remaining_gap_excluded_hours_proxy "
    "elapsed_operating_hours rul_operating_hours"
).split()
CSV_FIELDS = (
    "source_path test_id bearing_id channel_id sensor_within_bearing array_column "
    "sampling_hz sampling_hz_source n_samples snapshot_seconds end_defect "
    "rul_label_kind rul_count_proxy rul_calendar_hours_proxy"
).split() + TIME_FIELDS + FEATURE_NAMES + ["quality_flag"]


def resolve_source_path(value):
    """CSV의 프로젝트 기준 상대 경로를 실제 파일 경로로 바꾼다."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def discover_measurements(root, experiment="1st_test"):
    """선택한 실험 하위의 실제 날짜 파일을 수집한다. 실험끼리 섞지 않는다."""
    # 선택한 실험의 중첩 폴더만 검색해 다른 실험과 섞이지 않게 한다.
    if experiment not in EXPERIMENTS:
        raise ValueError(f"알 수 없는 실험: {experiment}")
    base = root / experiment if (root / experiment).is_dir() else root
    if experiment not in base.parts:
        raise ValueError(f"{experiment} 폴더 또는 NASA_IMS 루트를 지정해 주세요.")
    records, ignored = [], []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if not FILENAME_PATTERN.fullmatch(path.name):
            ignored.append(str(path))
            continue
        records.append((datetime.strptime(path.name, TIME_FORMAT), path.resolve()))
    records.sort(key=lambda record: record[0])
    if not records:
        raise ValueError("날짜 형식의 측정 파일이 없습니다.")
    if len({time for time, _ in records}) != len(records):
        raise ValueError("측정 시각이 중복됩니다. 원본 폴더 구조를 확인해 주세요.")
    return records, ignored


def build_timeline(records, gap_threshold_seconds=1200):
    """전체 파일의 시각으로 종료 기준 시간 라벨과 긴 기록 공백 목록을 만든다."""
    if not math.isfinite(gap_threshold_seconds) or gap_threshold_seconds <= 0:
        raise ValueError("공백 판정 기준은 양의 유한한 값으로 지정해 주세요.")
    times = [time for time, _ in records]
    intervals = np.array([0.0] + [(b - a).total_seconds() for a, b in zip(times, times[1:])])
    if np.any(intervals[1:] <= 0):
        raise ValueError("측정 시각이 엄격한 오름차순이 아닙니다.")
    gaps = intervals > gap_threshold_seconds
    # 긴 간격 전체를 제외한 참고값이다. 실제 정지 구간 경계를 모르므로 가동시간으로 확정하지 않는다.
    observed = np.cumsum(np.where(gaps, 0.0, intervals)) / 3600
    segments = np.cumsum(gaps)
    rows, gap_rows = [], []
    for index, time in enumerate(times):
        remaining = len(times) - index - 1
        calendar = (times[-1] - time).total_seconds() / 3600
        rows.append(dict(measurement_index=index, measured_at=time.isoformat(sep=" "),
                         dataset_end_at=times[-1].isoformat(sep=" "),
                         elapsed_calendar_hours=(time - times[0]).total_seconds() / 3600,
                         interval_seconds=float(intervals[index]), gap_before=bool(gaps[index]),
                         segment_id=int(segments[index]), remaining_measurements=remaining,
                         remaining_calendar_hours=calendar,
                         elapsed_gap_excluded_hours_proxy=float(observed[index]),
                         remaining_gap_excluded_hours_proxy=float(observed[-1] - observed[index]),
                         elapsed_operating_hours=None, rul_operating_hours=None))
        if gaps[index]:
            gap_rows.append(dict(before_at=times[index - 1].isoformat(sep=" "),
                                 after_at=time.isoformat(sep=" "),
                                 interval_hours=float(intervals[index] / 3600),
                                 after_index=index,
                                 interpretation="기록 중단 후보. 간격 전체가 정지시간이라고 단정할 수 없음"))
    return rows, gap_rows


def read_measurement(path, experiment="1st_test"):
    """실험별 8열/4열, 20,480행과 유한값을 검사한다."""
    # 시간 열과 헤더가 없으며, 실험에 따라 8열 또는 4열의 센서값이다.
    data = np.loadtxt(path, dtype=np.float64, encoding="ascii", ndmin=2)
    expected = (EXPECTED_SAMPLES, EXPERIMENTS[experiment]["columns"])
    if data.shape != expected:
        raise ValueError(f"측정 배열의 크기가 잘못되었습니다: {data.shape}, 기대값: {expected}")
    if not np.isfinite(data).all():
        raise ValueError("결측값 또는 무한대 측정값이 있습니다.")
    return data


def feature_rows(path, data, timeline, experiment="1st_test", all_bearings=False):
    """센서별 특징을 보존한다. 기존 함수 호출의 1st B3·4 선택도 유지한다."""
    result = []
    config = EXPERIMENTS[experiment]
    bearings = config["channels"] if all_bearings or experiment != "1st_test" else (3, 4)
    source = path.resolve()
    source = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source)
    for bearing in bearings:
        defect = config["defects"].get(bearing)
        kind = "test_end_proxy_not_exact_failure_time" if defect else "censored_no_reported_failure"
        for sensor, channel in enumerate(config["channels"][bearing], 1):
            # Ch5는 배열 인덱스 4다. 같은 베어링의 두 센서를 평균하지 않고 각각 보존한다.
            column = channel - 1
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                features = signal_features(data[:, column], SAMPLING_HZ)
            result.append(dict(source_path=source, test_id=experiment, bearing_id=bearing,
                               channel_id=channel, sensor_within_bearing=sensor, array_column=column,
                               sampling_hz=SAMPLING_HZ, sampling_hz_source="IMS_Readme_20kHz",
                               n_samples=len(data), snapshot_seconds=len(data) / SAMPLING_HZ,
                               end_defect=defect or "not_confirmed",
                               rul_label_kind=kind,
                               rul_count_proxy=timeline["remaining_measurements"] if defect else None,
                               rul_calendar_hours_proxy=timeline["remaining_calendar_hours"] if defect else None,
                               **timeline, **features,
                               quality_flag=QUALITY_FLAG if path.name in QUALITY_SUSPECT_TIMES.get(experiment, ()) else "unflagged"))
    return result






def process_experiment(input_root, experiment, output, limit=0, gap_threshold_seconds=1200):
    """원본은 읽기만 하고 새 특징 파일만 생성한다. 기존 파일은 거부한다."""
    records, ignored = discover_measurements(input_root, experiment)
    # 이 배포본은 기간과 파일 수를 함께 검사해 다른 관측 구간이 섞이는 것을 막는다.
    config = EXPERIMENTS[experiment]
    if "file_count" in config:
        actual = (len(records), records[0][1].name, records[-1][1].name)
        expected = (config["file_count"], config["first"], config["last"])
        if actual != expected:
            raise ValueError(f"{experiment} 파일 수·기간 불일치: 실제 {actual}, 기대 {expected}")
    timeline, gaps = build_timeline(records, gap_threshold_seconds)
    if output.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output}")
    selected = records[:limit] if limit else records
    output.parent.mkdir(parents=True, exist_ok=True)
    errors, row_count = [], 0
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i, (_, path) in enumerate(selected):
            try:
                data = read_measurement(path, experiment)
                rows = feature_rows(path, data, timeline[i], experiment, all_bearings=True)
                writer.writerows(rows)
                row_count += len(rows)
            except (OSError, ValueError, FloatingPointError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
            if (i + 1) % 500 == 0 or i + 1 == len(selected):
                print(f"{experiment}: {i + 1}/{len(selected)}, feature rows={row_count}, errors={len(errors)}", flush=True)
    input_label = input_root.resolve()
    input_label = input_label.relative_to(ROOT).as_posix() if input_label.is_relative_to(ROOT) else str(input_label)
    report = dict(experiment=experiment, input=input_label, discovered_files=len(records),
                  attempted=len(selected), feature_rows=row_count, errors=errors, ignored_files=ignored,
                  first_at=str(records[0][0]), end_at=str(records[-1][0]), channel_mapping=EXPERIMENTS[experiment]["channels"],
                  sampling_hz=SAMPLING_HZ, gap_threshold_seconds=gap_threshold_seconds, gap_count=len(gaps),
                  reported_end_defects=EXPERIMENTS[experiment]["defects"],
                  release_note=THIRD_RELEASE_NOTE if experiment == "3rd_test" else "동봉 IMS 설명서의 채널·종료 시 결함 명세 사용",
                  rul_policy="보고된 결함 베어링만 마지막 기록까지의 달력 시간 대리값; 품질 의심 마지막 기록도 포함. 실제 고장 시각/가동시간 아님. 그 외 라벨은 결측.",
                  rul_endpoint=dict(at=str(records[-1][0]), last_measurement_included=True, exact_failure_time_known=False),
                  quality_review=dict(flag=QUALITY_FLAG, interpretation="정지 또는 측정 상태 변화 의심; 원인 미확정. 원본·특징 행 유지, 상태 판단 보류.",
                                      marked_measurements=[path.name for _, path in selected if path.name in QUALITY_SUSPECT_TIMES.get(experiment, ())]),
                  gaps=gaps, complete=len(selected) == len(records) and not errors)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="1st_test")
    parser.add_argument("--limit", type=int, default=0, help="0은 전체; 소량 결과는 별도 폴더로 지정")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-output", type=Path, default=PREPROCESSING_REPORT, help="설정·오류·공백·품질 기록을 모을 새 JSON")
    parser.add_argument("--gap-threshold-seconds", type=float, default=1200)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다.")
    if not args.input.is_dir():
        print("원본 NASA IMS 데이터가 필요합니다. 재추출을 비활성화합니다.")
        return 0
    if args.output_dir.resolve().is_relative_to(args.input.resolve()):
        parser.error("출력은 원본 데이터 폴더 밖에 지정하세요.")
    if args.report_output.exists() or args.report_output.resolve().is_relative_to(args.input.resolve()):
        parser.error("새 --report-output을 원본 데이터 밖에 지정하세요. 기존 기록은 덮어쓰지 않습니다.")
    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    failed = False
    reports = {}
    for experiment in experiments:
        try:
            report = process_experiment(args.input, experiment, args.output_dir / f"{experiment}.csv", args.limit, args.gap_threshold_seconds)
            reports[experiment] = report
            failed |= bool(report["errors"])
        except FileExistsError as exc:
            parser.error(str(exc))
        except ValueError as exc:
            print(f"{experiment}: 원본 데이터 확인이 필요하여 비활성화합니다: {exc}")
            failed = True
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(dict(experiments=reports), ensure_ascii=False, indent=2), encoding="utf-8")
    return int(failed)



if __name__ == "__main__":
    raise SystemExit(main())
