"""대시보드에 필요한 CSV·평가·모델 로딩, 참고선과 그림 저장을 돕는다."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .preprocessing import DATA_DIR, PREPROCESSING_REPORT, EXPERIMENTS, read_measurement, resolve_source_path
    from .signal_processing import configure_font, signal_figure
else:
    from preprocessing import DATA_DIR, PREPROCESSING_REPORT, EXPERIMENTS, read_measurement, resolve_source_path
    from signal_processing import configure_font, signal_figure

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_path(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"지원하지 않는 실험: {experiment}")
    return DATA_DIR / f"{experiment}.csv"


def load_monitoring(experiment):
    frame = pd.read_csv(feature_path(experiment), encoding="utf-8-sig", parse_dates=["measured_at"])
    # 기존 첫 실험 CSV는 계산·내용을 유지하며 추가 품질 열 없이도 읽는다.
    if "quality_flag" not in frame:
        frame["quality_flag"] = "unflagged"
    return frame


def load_report(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def reference_status(frame, baseline_count=50):
    """센서별 초기 50시점 RMS의 median+6*MAD 참고선. 정상 보증/고장 확률 아님."""
    frame = frame.sort_values("measured_at").copy()
    if "quality_flag" in frame and not frame.empty and frame.iloc[-1].quality_flag != "unflagged":
        return dict(status="품질 주의: 상태 판단 보류", baseline_rows=min(len(frame), baseline_count), threshold=None)
    if len(frame) <= baseline_count:
        return dict(status="기준 구간 부족", baseline_rows=min(len(frame), baseline_count), threshold=None)
    baseline = frame.iloc[:baseline_count]["rms"].to_numpy(dtype=float)
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    if not np.isfinite(baseline).all() or mad == 0:
        return dict(status="기준 변동 추정 불가", baseline_rows=baseline_count, threshold=None)
    threshold = median + 6 * 1.4826 * mad
    return dict(status="참고선 초과" if float(frame.iloc[-1].rms) > threshold else "참고선 이내",
                baseline_rows=baseline_count, threshold=threshold,
                baseline_first_at=str(frame.iloc[0].measured_at),
                baseline_last_at=str(frame.iloc[baseline_count - 1].measured_at),
                baseline_assumption="초기 구간 정상 가정이며 실제 정상/고장 라벨 아님")


def checked_model(name, report):
    path = ROOT / "models" / name
    if sha256(path) != report["model_sha256"]:
        raise ValueError(f"평가 기록과 모델 해시가 다릅니다: {name}")
    return path


def baseline_predictions():
    # 기존 로컬 재현 기능에만 ML 라이브러리가 필요하다. UI 시작 때는 불러오지 않는다.
    import joblib
    if __package__:
        from .train_rul_baseline import DEFAULT_INPUT, MODEL_FEATURES, chronological_split, load_measurements, validate_reference_input
    else:
        from train_rul_baseline import DEFAULT_INPUT, MODEL_FEATURES, chronological_split, load_measurements, validate_reference_input
    report = load_report("reports/evaluation/rf_basic_metrics.json")
    data, _ = load_measurements(DEFAULT_INPUT)
    validate_reference_input(DEFAULT_INPUT, data, report)
    _, test, _, _ = chronological_split(data, report["train_fraction"], report.get("gap_measurements", 0))
    model = joblib.load(checked_model("rf_basic.joblib", report))
    return test.assign(prediction_random_forest=model.predict(test[MODEL_FEATURES]))


def build_figures(experiment, output_dir=None):
    frame = load_monitoring(experiment)
    if frame.empty:
        raise ValueError("특징이 비어 있습니다.")
    report_root = (output_dir or ROOT / "reports").resolve()
    output = report_root / "figures"
    if any(output.glob(f"ims_{experiment}_*.png")):
        raise FileExistsError(f"기존 신호/추세 그래프를 덮어쓰지 않습니다: {output}")
    # 필요한 원본은 실험당 마지막 관측 파일 하나뿐이다.
    latest = frame.sort_values("measured_at").iloc[-1]
    path = resolve_source_path(latest.source_path)
    if not path.is_file():
        print(f"{experiment}: 원본 측정 파일이 필요하여 그래프 생성을 비활성화합니다.")
        return dict(experiment=experiment, status="SKIP", reason="raw measurement absent")
    data = read_measurement(path, experiment)
    output.mkdir(parents=True, exist_ok=True)
    configure_font()
    files = []
    for bearing, channels in EXPERIMENTS[experiment]["channels"].items():
        for sensor, channel in enumerate(channels, 1):
            figure, _ = signal_figure(data[:, channel - 1], title=f"{experiment} Bearing {bearing} / Ch{channel} / {path.name}")
            target = output / f"ims_{experiment}_bearing{bearing}_sensor{sensor}_signal.png"
            figure.savefig(target, dpi=110)
            plt.close(figure)
            files.append(str(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target))
        figure, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained", sharex=True)
        for sensor in range(1, len(channels) + 1):
            group = frame.loc[frame.bearing_id.eq(bearing) & frame.sensor_within_bearing.eq(sensor)].sort_values("measured_at")
            segments = (group.measured_at.diff().dt.total_seconds().gt(1200) | group.measurement_index.diff().ne(1)).cumsum()
            for ax, feature in zip(axes, ("rms", "kurtosis", "crest_factor")):
                for part, (_, section) in enumerate(group.groupby(segments)):
                    ax.plot(section.measured_at, section[feature], linewidth=.7, color=f"C{sensor - 1}", label=f"sensor{sensor}" if part == 0 else None)
                marked = group.loc[group.quality_flag.ne("unflagged")]
                if not marked.empty:
                    ax.scatter(marked.measured_at, marked[feature], color="darkorange", marker="x", zorder=5,
                               label="정지/측정 변화 의심 (원본 유지)")
                ax.set_ylabel(feature)
                ax.legend()
        axes[0].set_title(f"{experiment} Bearing {bearing} / 달력 시각, 긴 공백 연결 안 함")
        target = output / f"ims_{experiment}_bearing{bearing}_degradation.png"
        figure.savefig(target, dpi=110)
        plt.close(figure)
        files.append(str(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target))
    report = dict(experiment=experiment, status="PASS", signal_source=str(latest.source_path),
                  source_sha256=sha256(path), signal_at=str(latest.measured_at), figures=files)
    return report


def write_summary(report_dir=None, output=None, figure_sources=None, dataset_provenance=None):
    """기존 평가와 실험·관측 구간 출처를 모은다. 상세 평가와 기존 요약은 덮어쓰지 않는다."""
    report_dir = (report_dir or ROOT / "reports").resolve()
    output = output or report_dir / "evaluation/summary.json"
    if output.exists():
        raise FileExistsError(f"기존 요약을 덮어쓰지 않습니다. 새 --output을 지정하세요: {output}")
    previous_path = report_dir / "evaluation/summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    experiments = {}
    preprocessing = json.loads(PREPROCESSING_REPORT.read_text(encoding="utf-8"))["experiments"]
    for name in EXPERIMENTS:
        metadata = preprocessing[name]
        signal_source = (figure_sources or {}).get(name, previous.get("experiments", {}).get(name, {}).get("signal_analysis"))
        experiments[name] = dict(
            feature_csv=f"data/processed/{name}.csv", measurements=metadata["discovered_files"],
            feature_rows=metadata["feature_rows"], channel_mapping=metadata["channel_mapping"],
            reported_end_defects=metadata["reported_end_defects"], release_note=metadata["release_note"],
            quality_review=metadata.get("quality_review"), rul_endpoint=metadata.get("rul_endpoint"),
            rul_policy=metadata["rul_policy"], signal_analysis=signal_source,
            figures=[p.relative_to(report_dir).as_posix() for p in sorted((report_dir / "figures").glob(f"ims_{name}_*.png"))])
    models = {}
    path = report_dir / "evaluation/rf_basic_metrics.json"
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        models["rf_basic"] = dict(evaluation="evaluation/rf_basic_metrics.json",
            predictions="predictions/rf_basic_predictions.csv", model="models/rf_basic.joblib",
            evaluation_protocol="1st Test 내부 시간순 평가. 다른 시험의 일반화 성능 아님.",
            evaluation_rows=report["split_audit"]["test_rows"], metrics=report["metrics"],
            model_sha256=report["model_sha256"])
        if (report_dir / "evaluation/rf_basic_holdout_metrics.json").is_file():
            models["rf_basic"]["experiment_holdout"] = dict(
                evaluation="evaluation/rf_basic_holdout_metrics.json",
                predictions="predictions/rf_basic_holdout_test_predictions.csv",
                model="models/rf_basic_holdout.joblib")
    path = report_dir / "evaluation/rf_engineered_feature_comparison.csv"
    if path.is_file():
        models["rf_engineered"] = dict(evaluation="evaluation/rf_engineered_feature_comparison.csv",
            metrics=pd.read_csv(path).to_dict("records"),
            evaluation_protocol="1st Train → 2nd Validation, 기존 특징 6개와 이동 통계 추가 특징 11개 비교",
            artifact_status="성능 비교표만 저장됨. 가중치와 개별 예측 파일은 생성되지 않았음.")
    if (report_dir / "predictions/ridge_test3_predictions.csv").is_file():
        models["ridge"] = dict(evaluation="evaluation/ridge_validation_alpha.csv",
            predictions="predictions/ridge_test3_predictions.csv", model="models/ridge_all_tests.joblib",
            evaluation_protocol="2nd로 alpha 선택, 1st+2nd 학습 → 3rd 탐색적 평가. 최종 가중치는 세 시험 모두 학습.")
    # 테스트가 이전 모델·특징·그림 보존을 확인할 수 있도록 원형 해시를 함께 기록한다.
    paths = [p for folder in [DATA_DIR, ROOT / "models", report_dir] for p in folder.rglob("*")
             if p.is_file() and p.suffix != ".log"
             and p.name not in ("summary.json", "verification.json", "project_structure_migration.json", "project_structure_verification.json")
             and "__pycache__" not in p.parts]
    assets = [dict(path=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p), sha256=sha256(p)) for p in sorted(paths)]
    provenance = dict(dataset_provenance or previous.get("dataset_provenance") or {})
    provenance["active_readme"] = "reports/evaluation/ims_data_readme.pdf"
    summary = dict(dataset="NASA IMS", experiments=experiments, models=models,
                   dataset_provenance=provenance,
                   note="모델 평가 JSON의 절대 경로와 코드 해시는 당시 실험 출처다. 현재 실행 경로는 이 요약과 코드에서 지정한다.",
                   preserved_assets=assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="저장 특징과 실제 원본으로 신호·열화 그림 생성")
    parser.add_argument("--summary", action="store_true", help="저장 평가·실험 출처 요약 생성; 학습 없음")
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    parser.add_argument("--output-dir", type=Path, help="보고서 루트. 하위 figures/evaluation/predictions 사용; 기본 reports")
    parser.add_argument("--signal-file", type=Path, help="선택한 NASA 원본 파일 하나의 FFT/STFT 분석")
    parser.add_argument("--bearing", type=int, choices=range(1,5), default=3)
    parser.add_argument("--sensor", type=int, default=1)
    parser.add_argument("--output", type=Path, help="선택 신호의 새 PNG 또는 새 요약 JSON")
    args = parser.parse_args()
    if sum([args.figures, args.summary, args.signal_file is not None]) > 1:
        parser.error("--figures, --summary, --signal-file 중 하나를 선택하세요.")
    if args.figures:
        report_root = (args.output_dir or ROOT / "reports").resolve()
        if (report_root / "evaluation/summary.json").exists():
            parser.error("기존 보고서 요약이 있습니다. 새 --output-dir을 지정하세요.")
        sources = {}
        for experiment in EXPERIMENTS if args.experiment == "all" else [args.experiment]:
            sources[experiment] = build_figures(experiment, report_root)
            print(json.dumps(sources[experiment], ensure_ascii=False), flush=True)
        write_summary(report_root, figure_sources=sources)
    elif args.summary:
        write_summary(args.output_dir, args.output)
    elif args.signal_file:
        if args.experiment == "all" or args.output is None:
            parser.error("한 실험의 --experiment와 새 --output을 지정하세요.")
        if not args.signal_file.is_file():
            print("원본 NASA IMS 데이터가 필요합니다. 신호 재계산을 비활성화합니다.")
            return
        channels = EXPERIMENTS[args.experiment]["channels"][args.bearing]
        if not 1 <= args.sensor <= len(channels):
            parser.error("해당 베어링에 없는 센서입니다.")
        if args.output.exists():
            parser.error("기존 그래프를 덮어쓰지 않습니다. 새 --output을 지정하세요.")
        data = read_measurement(args.signal_file,args.experiment)
        figure,_ = signal_figure(data[:,channels[args.sensor-1]-1],title=f"{args.experiment} B{args.bearing} Ch{channels[args.sensor-1]} / {args.signal_file.name}")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        figure.savefig(args.output,dpi=120)
        plt.close(figure)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
