"""python -m src.analyze_ims_alarm: 실측 CSV 비교표와 독립 HTML 그래프 생성."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .ims_alarm import compare_series, compute_alarm_thresholds, rms_figure, validate_alarm_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1]/"data/processed")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="보고서 루트: evaluation/figures 사용")
    args = parser.parse_args()
    evaluation, figures = args.output / "evaluation", args.output / "figures"
    evaluation.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables, errors, charts, inputs = [], [], [], {}
    # 종료 결함 정보는 아래의 결과 검토 열에서만 사용한다. 판정 함수에 전달하지 않는다.
    reported = {"1st_test": {3, 4}, "2nd_test": {1}, "3rd_test": {3}}
    for experiment in reported:
        path = args.data_dir / f"{experiment}.csv"
        if not path.is_file():
            errors.append(f"{path}: CSV 없음; 추정 결과 없음")
            continue
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = pd.read_csv(path)
        for (bearing, sensor), group in data.groupby(["bearing_id", "sensor_within_bearing"]):
            try:
                frame = validate_alarm_frame(group)
                table = compare_series(frame, experiment, int(bearing), int(sensor))
                table["end_defect_reported"] = int(bearing) in reported[experiment]
                tables.append(table)
                for multiplier in (5, 7, 10):
                    thresholds = compute_alarm_thresholds(frame, multiplier)
                    for duration in (1, 3, 5):
                        fig = rms_figure(frame, thresholds, duration,
                            f"{experiment} B{bearing} S{sensor} · MAD {multiplier} · consecutive {duration}")
                        charts.append(fig.to_html(full_html=False, include_plotlyjs=True if not charts else False))
            except (ValueError, KeyError) as exc:
                errors.append(f"{experiment} B{bearing} S{sensor}: {exc}")
    if not tables:
        raise SystemExit("분석 가능한 실제 CSV 없음: " + "; ".join(errors))
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(evaluation/"ims_alarm_comparison.csv", index=False, encoding="utf-8-sig")
    summary = result.groupby(["mad_multiplier", "consecutive", "level", "end_defect_reported"], sort=False).agg(
        sensors=("alarm_occurred", "size"), triggered=("alarm_occurred", "sum"),
        median_remaining_hours=("remaining_calendar_hours", "median")).reset_index()
    summary.to_csv(evaluation/"ims_alarm_summary.csv", index=False, encoding="utf-8-sig")
    baseline = result.drop_duplicates(["test_id", "bearing_id", "sensor"])
    baseline = baseline[["test_id", "bearing_id", "sensor", "invalid_total"] +
                        [col for col in result if col.startswith("baseline_")]]
    baseline.to_csv(evaluation/"ims_alarm_baseline_checks.csv", index=False, encoding="utf-8-sig")
    (figures/"ims_alarm_rms_all_combinations.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>IMS RMS alarms</title></head><body>'
        '<h1>Actual CSV RMS: MAD 5/7/10 × consecutive 1/3/5</h1>'
        '<p>Gray region: baseline candidate. Markers: confirmed events. 1: peak sensitivity comparison only. '
        'Remaining calendar time is time to test end, not time to fault onset.</p>' + ''.join(charts) + '</body></html>',
        encoding="utf-8")
    (evaluation/"ims_alarm_analysis_metadata.json").write_text(json.dumps(dict(
        input_sha256=inputs, errors=errors, series=len(baseline), rows=len(result),
        note="No raw TXT read. End defect reports used only for review grouping, never detection."),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Series={len(baseline)}, comparison rows={len(result)}, errors={errors}")


if __name__ == "__main__":
    main()
