"""NASA IMS 대리 RUL 회귀. --experiment-split은 1st/2nd/3rd 독립 실험 분리를 사용한다.

같은 측정 시점의 Bearing 3·4와 센서들을 함께 시간순으로 분할한다.
정답 단위는 남은 달력 시간이며 실제 고장 수명이나 실제 가동시간이 아니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import matplotlib

# 화면을 열지 않고 평가 그림을 저장한다.
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

if __package__:
    from .preprocessing import DATA_DIR, PREPROCESSING_REPORT, EXPERIMENTS
    from .signal_processing import configure_font
else:
    from preprocessing import DATA_DIR, PREPROCESSING_REPORT, EXPERIMENTS
    from signal_processing import configure_font


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATA_DIR / "1st_test.csv"
# 모니터링은 세 실험, 저장 RUL 모델은 두 센서가 있는 첫 실험 전용이다.
TRAINING_EXPERIMENTS = ("1st_test",)
TARGET = "rul_calendar_hours_proxy"
LABEL_KIND = "test_end_proxy_not_exact_failure_time"
# 허용한 현재 시점 신호 특징만 사용한다. 숫자형 열도 자동으로 추가하지 않는다.
SIGNAL_FEATURES = ["rms", "std", "peak", "kurtosis", "crest_factor", "fft_centroid_hz"] + [
    f"fft_peak_{rank}_{kind}" for rank in (1, 2, 3) for kind in ("hz", "amplitude")
]
# 두 센서를 평균하지 않고 각각 보존한다. 센서 번호를 임의의 축 방향으로 해석하지 않는다.
MODEL_FEATURES = [f"sensor{sensor}_{name}" for sensor in (1, 2) for name in SIGNAL_FEATURES]
EXPERIMENT_PARTITIONS = {"train": "1st_test", "validation": "2nd_test", "test": "3rd_test"}
COMMON_FEATURES = list(SIGNAL_FEATURES)
METHODS = {
    "random_forest": "Random Forest",
    "train_median": "학습 정답 중앙값 고정",
    "last_train_target": "마지막 학습 정답 고정",
}
LIMITATIONS = [
    "정답은 마지막 기록까지 남은 달력 시간의 대리값이다. 정확한 고장 시각, 실제 잔여 수명, 실제 가동시간이 아니다.",
    "Bearing 3의 내륜 결함과 Bearing 4의 전동체 결함은 종료 시 확인되었다. 두 베어링의 고장이 마지막 기록에 동시에 발생했다고 가정하지 않는다.",
    "학습·평가가 동일 1st Test와 동일 베어링 수명 궤적을 공유한다. 다른 시험·베어링·설비에서의 일반화 성능은 검증하지 않았다.",
    "학습 구간의 RUL 정답도 시험 종료 후 생성된 회고적 정답이다. 해당 정답을 가진 모델이 분할 시점에 실제로 이용 가능했다는 뜻이 아니다.",
    "시간·종료점·순번·정답 파생 열은 입력에서 제외했으며 결측 대체값도 학습 부분에만 적합했다. 미래 평가 특징이나 정답을 적합·튜닝에 사용하지 않았다.",
    "앞부분 학습 정답보다 뒷부분 평가 정답이 작다. Random Forest는 학습 정답 범위 밖으로 외삽하지 못해 종료 직전의 작은 RUL을 예측하기 어렵다.",
    "단순 기준 예측도 학습 정답만 사용하지만 그 정답 자체는 종료 후 알려진 대리값이다. 실시간으로 이용 가능한 잔여 수명 기준선으로 해석하면 안 된다.",
    "두 베어링은 같은 시험 조건과 종료점을 공유하며 인접 측정도 상관되어 있다. 행 수를 독립 수명 실험 수로 해석하지 않는다.",
    "긴 기록 공백도 달력 RUL에 포함된다. 기록 공백이 모두 정지시간인지, 센서의 물리 단위가 무엇인지 임의로 단정하지 않는다.",
    "고정된 시간순 홀드아웃을 한 번 평가한다. 평가 결과로 설정을 반복 선택하면 별도 최종 시험셋이 필요하다.",
]


def sha256(path):
    """입력·코드·모델의 재현성 확인용 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_measurements(path):
    """실제 스키마와 종료 기준 정답을 확인하고 베어링별 두 센서를 결합한다."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = set(SIGNAL_FEATURES + [
        "test_id", "bearing_id", "channel_id", "sensor_within_bearing", "source_path",
        "measurement_index", "measured_at", "dataset_end_at", "end_defect", "rul_label_kind", TARGET,
    ])
    if missing := sorted(required - set(raw.columns)):
        raise ValueError(f"필수 컬럼 누락: {missing}")
    # 다른 실험의 행을 베어링 필터로 조용히 버리지 않고 입력 전체에서 거부한다.
    if raw["test_id"].isna().any() or not raw["test_id"].isin(TRAINING_EXPERIMENTS).all():
        raise ValueError("현재 RUL 모델은 1st_test 전용입니다. 다른 실험을 섞거나 임의 정답으로 학습하지 마세요.")
    data = raw.loc[raw["bearing_id"].isin([3, 4])].copy()
    if data.empty or set(data["bearing_id"]) != {3, 4}:
        raise ValueError("Bearing 3·4가 모두 필요합니다.")
    if not data["test_id"].eq("1st_test").all() or not data["rul_label_kind"].eq(LABEL_KIND).all():
        raise ValueError("1st Test의 시험 종료 기준 대리 정답만 지원합니다.")
    for bearing, defect in ((3, "inner_race"), (4, "rolling_element")):
        if not data.loc[data["bearing_id"].eq(bearing), "end_defect"].eq(defect).all():
            raise ValueError(f"Bearing {bearing}의 종료 시 결함 정보가 예상과 다릅니다.")
    for name in ("measured_at", "dataset_end_at"):
        data[name] = pd.to_datetime(data[name], errors="raise")
        if data[name].isna().any():
            raise ValueError(f"필수 시각 결측: {name}")
    if data["source_path"].isna().any() or data["dataset_end_at"].nunique() != 1:
        raise ValueError("원본 경로와 하나의 공통 종료 시각이 필요합니다.")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise")
    if not np.isfinite(data[TARGET]).all() or (data[TARGET] < 0).any():
        raise ValueError("대리 RUL 정답은 음수가 아닌 유한값이어야 합니다.")
    expected = (data["dataset_end_at"] - data["measured_at"]).dt.total_seconds() / 3600
    if not np.allclose(data[TARGET], expected, rtol=0, atol=1e-8):
        raise ValueError("대리 RUL이 마지막 기록까지 남은 달력 시간과 일치하지 않습니다.")
    for name in SIGNAL_FEATURES:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if np.isinf(data[SIGNAL_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("신호 특징에 무한대가 있습니다.")
    keys = ["measured_at", "bearing_id", "sensor_within_bearing"]
    if data.duplicated(keys).any():
        raise ValueError("동일 측정 시각·베어링·센서의 중복 행이 있습니다.")
    expected_channels = {(3, 1): 5, (3, 2): 6, (4, 1): 7, (4, 2): 8}
    for row in data[["bearing_id", "sensor_within_bearing", "channel_id"]].drop_duplicates().itertuples(index=False):
        if expected_channels.get((row.bearing_id, row.sensor_within_bearing)) != row.channel_id:
            raise ValueError("베어링·센서·채널 연결이 1st Test 명세와 다릅니다.")
    if not data.groupby("measured_at").size().eq(4).all():
        raise ValueError("모든 측정 시각에 Bearing 3·4의 두 센서가 각각 있어야 합니다.")
    grouped = data.groupby("measured_at", sort=True)
    for name in (TARGET, "source_path", "measurement_index"):
        if not grouped[name].nunique(dropna=False).eq(1).all():
            raise ValueError(f"같은 측정 시점의 메타데이터가 다릅니다: {name}")
    timeline = grouped[[TARGET, "source_path", "measurement_index"]].first()
    index = pd.to_numeric(timeline["measurement_index"], errors="raise").to_numpy()
    if not np.isfinite(index).all() or np.any(np.diff(index) <= 0):
        raise ValueError("측정 순번이 시간순으로 증가하지 않습니다.")
    if timeline["source_path"].duplicated().any():
        raise ValueError("같은 원본 파일이 여러 측정 시각에 연결되어 있습니다.")
    bearing_keys = ["measured_at", "bearing_id"]
    parts = [data.loc[data["sensor_within_bearing"].eq(sensor)].set_index(bearing_keys)[SIGNAL_FEATURES]
             .add_prefix(f"sensor{sensor}_") for sensor in (1, 2)]
    wide = pd.concat(parts, axis=1).loc[:, MODEL_FEATURES]
    metadata = data.groupby(bearing_keys)[[TARGET, "source_path", "measurement_index"]].first()
    wide = metadata.join(wide).reset_index().sort_values(bearing_keys).reset_index(drop=True)
    inspection = {
        "csv_columns": raw.columns.tolist(), "csv_rows": len(raw), "selected_sensor_rows": len(data),
        "measurements": len(timeline), "bearing_time_rows": len(wide),
        "first_at": str(timeline.index.min()), "last_at": str(timeline.index.max()),
        "dataset_end_at": str(data["dataset_end_at"].iloc[0]),
        "target": TARGET, "target_unit": "달력 시간(시간)", "rul_label_kind": LABEL_KIND,
        "rul_endpoint_policy": {"last_measurement_included": True, "exact_failure_time_known": False,
                                "definition": "해당 실험 마지막 기록까지의 달력 시간; 실제 고장 정답 아님"},
        "target_range_hours": [float(data[TARGET].min()), float(data[TARGET].max())],
        "channel_mapping": {"3": [5, 6], "4": [7, 8]},
        "sensor_rows_per_bearing": {str(k): int(v) for k, v in data.groupby("bearing_id").size().items()},
        "available_rul_columns": [name for name in raw if name.startswith(("rul_", "remaining_"))],
        "rul_units": {
            "rul_calendar_hours_proxy": "시간; 기록 공백을 포함한 종료까지 남은 달력 시간",
            "rul_count_proxy": "회; 종료까지 남은 측정 파일 수",
            "remaining_gap_excluded_hours_proxy": "시간; 긴 기록 공백 전체를 제외한 대리값, 실제 가동시간 아님",
            "rul_operating_hours": "시간; 실제 가동시간 미상으로 결측",
        },
        "excluded_csv_columns": sorted(set(raw.columns) - set(SIGNAL_FEATURES)),
    }
    return wide, inspection


def measurement_sha256(data):
    """모델이 실제 사용하는 행·특징·라벨·시각의 내용 해시다. CSV 배치와 별도로 확인한다."""
    text = data.to_csv(index=False, float_format="%.17g", date_format="%Y-%m-%d %H:%M:%S")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_reference_input(path, data, reference):
    """과거 입력 해시는 보존하고, 검증한 공통 CSV만 동일 입력으로 인정한다."""
    actual = sha256(path)
    if actual == reference["input_sha256"]:
        return
    record = json.loads(PREPROCESSING_REPORT.read_text(encoding="utf-8")).get("model_input_compatibility", {})
    if (reference["input_sha256"] != record.get("legacy_input_sha256")
            or actual != record.get("current_csv_sha256")
            or measurement_sha256(data) != record.get("model_measurements_sha256")):
        raise ValueError("입력 CSV가 기준 모델의 검증된 학습 입력과 다릅니다.")


def chronological_split(data, train_fraction, gap_measurements=0):
    """시각 전체를 묶어 분할하고 파일·시간의 중복과 엄격한 선후 관계를 검사한다."""
    times = pd.Index(data["measured_at"].unique()).sort_values()
    cut = int(len(times) * train_fraction)
    if cut < 5 or len(times) - cut - gap_measurements < 2:
        raise ValueError("학습에 최소 5시점, 평가에 최소 2시점이 필요합니다.")
    train_times, test_times = times[:cut], times[cut + gap_measurements:]
    split = data.copy()
    split["partition"] = np.where(split["measured_at"].isin(train_times), "train",
                                  np.where(split["measured_at"].isin(test_times), "test", "gap"))
    train, test = split.loc[split["partition"].eq("train")], split.loc[split["partition"].eq("test")]
    audit = {
        "train_measurements": len(train_times), "test_measurements": len(test_times),
        "gap_measurements": gap_measurements, "train_rows": len(train), "test_rows": len(test),
        "train_first_at": str(train["measured_at"].min()), "train_last_at": str(train["measured_at"].max()),
        "test_first_at": str(test["measured_at"].min()), "test_last_at": str(test["measured_at"].max()),
        "time_overlap_count": len(set(train["measured_at"]) & set(test["measured_at"])),
        "source_file_overlap_count": len(set(train["source_path"]) & set(test["source_path"])),
        "strict_time_order": bool(train["measured_at"].max() < test["measured_at"].min()),
        "all_bearings_at_same_time_in_one_partition": bool(split.groupby("measured_at")["partition"].nunique().eq(1).all()),
    }
    audit["passed"] = (audit["strict_time_order"] and not audit["time_overlap_count"]
                       and not audit["source_file_overlap_count"]
                       and audit["all_bearings_at_same_time_in_one_partition"])
    if not audit["passed"]:
        raise ValueError(f"시간순 분할 누수 검사 실패: {audit}")
    return train, test, split, audit


def scores(actual, predicted):
    """예측 오차를 정답과 같은 시간 단위로 반환한다."""
    return {"MAE_hours": float(mean_absolute_error(actual, predicted)),
            "RMSE_hours": float(np.sqrt(mean_squared_error(actual, predicted)))}


def load_experiment_splits(data_dir=DATA_DIR):
    """실제 ID·채널·라벨을 확인하고 센서별 기존 특징의 평균으로 베어링당 한 행을 만든다."""
    splits, details = {}, {}
    forbidden = {"test_id", "bearing_id", "channel_id", "sensor_within_bearing", "measurement_index",
                 "measured_at", "dataset_end_at", "source_path", "quality_flag", TARGET}
    assert not forbidden.intersection(COMMON_FEATURES)
    assert all(not name.startswith(("rul_", "remaining_", "elapsed_")) for name in COMMON_FEATURES)
    for partition, experiment in EXPERIMENT_PARTITIONS.items():
        path = Path(data_dir) / f"{experiment}.csv"
        raw = pd.read_csv(path, encoding="utf-8-sig")
        required = set(COMMON_FEATURES + ["test_id", "bearing_id", "channel_id", "sensor_within_bearing",
            "measurement_index", "measured_at", "dataset_end_at", "source_path", "end_defect",
            "rul_label_kind", TARGET, "quality_flag"])
        if missing := required - set(raw):
            raise ValueError(f"{path.name} 필수 열 누락: {sorted(missing)}")
        if raw.empty or set(raw.test_id.dropna()) != {experiment} or raw.test_id.isna().any():
            raise ValueError(f"파일명과 실제 실험 ID 불일치 또는 혼합: {path.name}")
        raw["measured_at"] = pd.to_datetime(raw.measured_at, errors="raise")
        raw["dataset_end_at"] = pd.to_datetime(raw.dataset_end_at, errors="raise")
        if raw.measured_at.isna().any() or not raw.measured_at.is_monotonic_increasing:
            raise ValueError("측정 시각이 결측이거나 시간순 정렬이 아닙니다.")
        if raw.dataset_end_at.nunique() != 1 or raw.dataset_end_at.iloc[0] != raw.measured_at.max():
            raise ValueError("전체 시험의 마지막 기록과 대리 RUL 종료점이 다릅니다.")
        if raw.duplicated(["bearing_id", "measured_at", "sensor_within_bearing"]).any():
            raise ValueError("동일 베어링·시각·센서 중복 행입니다.")
        config = EXPERIMENTS[experiment]
        mapping = {(b, s): c for b, channels in config["channels"].items() for s, c in enumerate(channels, 1)}
        if set(raw.bearing_id) != set(config["channels"]):
            raise ValueError("베어링 구성이 실험 명세와 다릅니다.")
        for row in raw[["bearing_id", "sensor_within_bearing", "channel_id"]].drop_duplicates().itertuples(index=False):
            if mapping.get((row.bearing_id, row.sensor_within_bearing)) != row.channel_id:
                raise ValueError("실험별 센서·채널 매핑 불일치입니다.")
        for bearing, channels in config["channels"].items():
            counts = raw.loc[raw.bearing_id.eq(bearing)].groupby("measured_at").size()
            if len(counts) != raw.measured_at.nunique() or not counts.eq(len(channels)).all():
                raise ValueError("측정 시점 또는 센서 누락입니다.")
        for name in COMMON_FEATURES:
            raw[name] = pd.to_numeric(raw[name], errors="raise")
        if np.isinf(raw[COMMON_FEATURES].to_numpy()).any():
            raise ValueError("입력 특징에 무한대가 있습니다.")
        selected = raw.loc[raw.bearing_id.isin(config["defects"])].copy()
        if raw.loc[~raw.bearing_id.isin(config["defects"]), TARGET].notna().any():
            raise ValueError("결함 미보고 베어링에 RUL 정답이 있습니다.")
        selected[TARGET] = pd.to_numeric(selected[TARGET], errors="raise")
        expected = (selected.dataset_end_at - selected.measured_at).dt.total_seconds() / 3600
        if (not np.isfinite(selected[TARGET]).all() or (selected[TARGET] < 0).any()
                or not np.allclose(selected[TARGET], expected, atol=1e-8, rtol=0)
                or not selected.rul_label_kind.eq(LABEL_KIND).all()):
            raise ValueError("보고된 결함 베어링의 시험 종료 기준 대리 라벨이 잘못되었습니다.")
        for bearing, defect in config["defects"].items():
            if not selected.loc[selected.bearing_id.eq(bearing), "end_defect"].eq(defect).all():
                raise ValueError("보고된 결함 종류가 실험 명세와 다릅니다.")
        keys = ["test_id", "bearing_id", "measured_at"]
        metadata = ["source_path", "measurement_index", TARGET, "quality_flag"]
        grouped = selected.groupby(keys, sort=True)
        if not grouped[metadata].nunique(dropna=False).eq(1).all().all():
            raise ValueError("같은 시점의 센서별 시각·파일·라벨·품질 기록이 다릅니다.")
        # 신호 자체를 평균하거나 특징을 새로 계산하지 않는다. 동일 시점 센서 특징의 평균이다.
        frame = grouped[metadata].first().join(grouped[COMMON_FEATURES].mean()).reset_index()
        frame = frame.sort_values(["test_id", "bearing_id", "measured_at"]).reset_index(drop=True)
        for _, group in frame.groupby("bearing_id"):
            if not group.measurement_index.diff().iloc[1:].eq(1).all():
                raise ValueError("측정 순번이 누락되었거나 잘못 정렬됐습니다.")
        frame["partition"] = partition
        splits[partition] = frame
        details[partition] = dict(experiment=experiment, csv=str(path), csv_sha256=sha256(path),
            source_rows=len(raw), selected_sensor_rows=len(selected), bearing_rows=len(frame),
            bearings=sorted(config["defects"]), sensors_per_bearing={str(k): len(v) for k,v in config["channels"].items()},
            first_at=str(raw.measured_at.min()), end_at=str(raw.measured_at.max()),
            missing_features={k:int(v) for k,v in frame[COMMON_FEATURES].isna().sum().items()},
            quality_marked_rows=int(frame.quality_flag.ne("unflagged").sum()),
            excluded_columns=sorted(set(raw)-set(COMMON_FEATURES)))
    experiment_sets = {k:set(v.test_id) for k,v in splits.items()}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert experiment_sets[a].isdisjoint(experiment_sets[b]), "실험 ID 누수"
        assert set(splits[a].source_path).isdisjoint(set(splits[b].source_path)), "원본 파일 누수"
    audit = dict(partitions=details, experiment_ids_disjoint=True, source_files_disjoint=True,
        feature_names=COMMON_FEATURES, identifier_or_rul_features=False,
        aggregation="베어링·측정 시점별 센서 특징 평균; 1st의 두 센서 평균, 2nd/3rd의 한 센서 그대로",
        target=TARGET, label_policy="보고된 결함 베어링만 마지막 기록까지의 달력 시간 대리값; 실제 고장 시각 아님",
        last_quality_suspect_rows_included=True, fit_partition="train", test_used_for_selection=False)
    return splits, audit


def experiment_scores(actual, predicted):
    return dict(scores(actual, predicted), R2=float(r2_score(actual, predicted)))


def save_experiment_outputs(name, report, predictions, validation, report_dir):
    """실험별 분리 결과를 접두사로 구분하여 세 폴더에 저장한다."""
    root = Path(report_dir)
    for folder in ("evaluation", "predictions", "figures"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    output = root / "evaluation" / f"{name}_metrics.json"
    targets = [output, root / "figures" / f"{name}_test.png"] + [
        root / "predictions" / f"{name}_{part}_predictions.csv" for part in ("test", "validation")]
    if any(path.exists() for path in targets):
        raise FileExistsError("기존 평가·예측·그림을 덮어쓰지 않습니다.")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for partition, frame in [("test",predictions),("validation",validation)]:
        frame.to_csv(root / "predictions" / f"{name}_{partition}_predictions.csv", index=False, encoding="utf-8-sig")
    configure_font()
    figure, ax = plt.subplots(figsize=(12,5), layout="constrained")
    gaps=predictions.measured_at.diff().dt.total_seconds().gt(1200)
    ax.plot(predictions.measured_at, predictions[TARGET].mask(gaps), label="마지막 기록 기준 대리 정답", color="black", linewidth=1)
    ax.plot(predictions.measured_at, predictions.prediction.mask(gaps), label=name, linewidth=.8)
    marked = predictions.loc[predictions.quality_flag.ne("unflagged")]
    ax.scatter(marked.measured_at, marked[TARGET], marker="x", color="darkorange", label="정지/측정 변화 의심; 포함")
    ax.set(title=f"{name}: Train 1st / Validation 2nd / Test 3rd B3", xlabel="측정 시각", ylabel="달력 시간 대리 RUL (시간)")
    ax.legend();figure.savefig(root / "figures" / f"{name}_test.png", dpi=140);plt.close(figure)


def run_experiment_rf(data_dir, report_dir, model_output, trees=200, seed=42):
    if trees < 1:
        raise ValueError("트리 수는 양수여야 합니다.")
    paths = [Path(model_output), Path(report_dir)/"evaluation/rf_basic_holdout_metrics.json",
             Path(report_dir)/"predictions/rf_basic_holdout_test_predictions.csv", Path(report_dir)/"predictions/rf_basic_holdout_validation_predictions.csv",
             Path(report_dir)/"figures/rf_basic_holdout_test.png",
             Path(report_dir)/"evaluation/rf_basic_holdout_split_audit.json",
             Path(report_dir)/"evaluation/rf_basic_holdout_split_assignments.csv"]
    if any(path.exists() for path in paths):
        raise FileExistsError("기존 실험 단위 RF 결과가 있습니다. 새 출력 경로를 지정하세요.")
    splits, audit = load_experiment_splits(data_dir)
    train, validation = splits["train"], splits["validation"]
    if train[COMMON_FEATURES].isna().all().any():
        raise ValueError("Train에서 전체 결측인 특징이 있습니다.")
    trials, best_model, best_score, selected = [], None, float("inf"), None
    for depth, leaf in [(8,5),(None,5),(8,20),(None,20)]:
        model = Pipeline([("imputer",SimpleImputer(strategy="median")),
            ("regressor",RandomForestRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=leaf,random_state=seed,n_jobs=-1))])
        model.fit(train[COMMON_FEATURES], train[TARGET])
        statistics = model.named_steps["imputer"].statistics_.copy()
        predicted = model.predict(validation[COMMON_FEATURES])
        np.testing.assert_array_equal(statistics,model.named_steps["imputer"].statistics_)
        metric = experiment_scores(validation[TARGET],predicted)
        trial = dict(max_depth=depth,min_samples_leaf=leaf,trees=trees,validation=metric)
        trials.append(trial);print(json.dumps(trial),flush=True)
        if metric["MAE_hours"] < best_score:
            best_model,best_score,selected=model,metric["MAE_hours"],trial
    # Test는 모든 후보 선택을 끝낸 다음 한 번만 예측한다. Train+Validation 재적합 없음.
    test = splits["test"].copy();test["prediction"]=best_model.predict(test[COMMON_FEATURES])
    validation = validation.copy();validation["prediction"]=best_model.predict(validation[COMMON_FEATURES])
    median = float(train[TARGET].median());test["prediction_train_median"]=median
    clean = test.quality_flag.eq("unflagged")
    Path(model_output).parent.mkdir(parents=True,exist_ok=True);joblib.dump(best_model,model_output)
    report = dict(model="Random Forest",protocol="experiment_holdout",audit=audit,selected=selected,trials=trials,
        metrics=dict(test=experiment_scores(test[TARGET],test.prediction),validation=experiment_scores(validation[TARGET],validation.prediction),
                     train_median_test=experiment_scores(test[TARGET],test.prediction_train_median),
                     test_without_quality_suspects=experiment_scores(test.loc[clean,TARGET],test.loc[clean,"prediction"])),
        preprocessing=dict(fit_partition="train",fit_rows=len(train),imputer_medians=best_model.named_steps["imputer"].statistics_.tolist(),normalization="RF는 정규화 없음"),
        model_features=COMMON_FEATURES,model_output=str(model_output),model_sha256=sha256(Path(model_output)),seed=seed,
        train_median_hours=median,test_evaluated_after_selection=True,refit_on_validation=False,
        caveats=["정답은 실제 고장이 아닌 시험 종료 기준 대리값", "실험별 센서 수·측정 분포·시험 길이가 다름", "독립 Test 실험은 하나뿐", "품질 의심 마지막 기록도 주 평가에 포함"])
    save_experiment_outputs("rf_basic_holdout",report,test,validation,report_dir)
    evaluation=Path(report_dir)/"evaluation"
    (evaluation/"rf_basic_holdout_split_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.concat(splits.values(),ignore_index=True)[["partition","test_id","bearing_id","measured_at","source_path","measurement_index",TARGET,"quality_flag"]].to_csv(evaluation/"rf_basic_holdout_split_assignments.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(report["metrics"]),flush=True)
    return 0


def evaluate(predictions):
    """전체와 베어링별로 동일한 평가 행에서 모델과 단순 기준을 비교한다."""
    result = {}
    groups = [("overall", predictions)] + [
        (f"bearing_{bearing}", group) for bearing, group in predictions.groupby("bearing_id")]
    for key, group in groups:
        result[key] = {name: scores(group[TARGET], group[f"prediction_{name}"]) for name in METHODS}
    return result


def make_plots(predictions, split, directory, provisional):
    """평가 구간의 정답·예측과 시간 분할 그림을 저장하며 긴 공백은 선으로 연결하지 않는다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, layout="constrained")
    colors = {"random_forest": "#d45b27", "train_median": "#8c6bb1", "last_train_target": "#64965b"}
    for bearing, ax in zip((3, 4), axes):
        group = predictions.loc[predictions["bearing_id"].eq(bearing)].sort_values("measured_at").copy()
        segments = group["measured_at"].diff().dt.total_seconds().gt(1200).cumsum()
        for number, (_, part) in enumerate(group.groupby(segments)):
            ax.plot(part["measured_at"], part[TARGET], ".-", color="#222222", markersize=2,
                    label="종료 기준 대리 정답" if number == 0 else None)
            for name, label in METHODS.items():
                ax.plot(part["measured_at"], part[f"prediction_{name}"],
                        ".-" if name == "random_forest" else "--", color=colors[name],
                        markersize=2, linewidth=1, label=label if number == 0 else None)
        ax.set(title=f"Bearing {bearing}: 뒤쪽 평가 구간", ylabel="종료까지 남은 달력 시간 (시간)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].set_xlabel("측정 시각; 20분 초과 공백에서는 선을 끊음")
    fig.suptitle(("소량 실행 점검 — " if provisional else "") + "NASA IMS 1st Test 내부 시간순 평가\n"
                 "실제 고장 수명이 아닌 시험 종료 기준 대리 RUL")
    fig.savefig(directory / "rf_basic_predictions.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    timeline = split.drop_duplicates("measured_at").sort_values("measured_at")
    for name, label, color in (("train", "학습", "#277da1"), ("gap", "분할 사이 제외", "#888888"),
                                ("test", "평가", "#d45b27")):
        part = timeline.loc[timeline["partition"].eq(name)]
        if not part.empty:
            ax.scatter(part["measured_at"], part[TARGET], s=7, color=color, label=label)
    ax.set(title="전체 측정 시점 단위의 시간순 분할", xlabel="측정 시각",
           ylabel="종료까지 남은 달력 시간 (시간)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(directory / "rf_basic_time_split.png", dpi=160)
    plt.close(fig)




def save_results(report, predictions, split, model, report_dir, model_output):
    """학습과 분리된 저장 단계. 최종 결과만 세 폴더에 쓰고 기존 파일은 거부한다."""
    report_dir, model_output = Path(report_dir), Path(model_output)
    evaluation = report_dir / "evaluation"
    prediction_file = report_dir / "predictions/rf_basic_predictions.csv"
    figures = report_dir / "figures"
    targets = [model_output, evaluation / "rf_basic_metrics.json", prediction_file,
               evaluation / "rf_basic_split_assignments.csv", evaluation / "rf_basic_feature_importance.csv",
               figures / "rf_basic_predictions.png", figures / "rf_basic_time_split.png"]
    if any(path.exists() for path in targets):
        raise FileExistsError("기존 RF 결과를 덮어쓰지 않습니다. 새 보고서 루트와 모델 경로를 지정하세요.")
    for directory in [evaluation, prediction_file.parent, figures, model_output.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_file, index=False, encoding="utf-8-sig")
    split[["measured_at", "bearing_id", "source_path", "measurement_index", TARGET, "partition"]].to_csv(
        evaluation / "rf_basic_split_assignments.csv", index=False, encoding="utf-8-sig")
    pd.Series(model.named_steps["regressor"].feature_importances_, index=MODEL_FEATURES).sort_values(
        ascending=False).rename_axis("feature").to_csv(evaluation / "rf_basic_feature_importance.csv",
                                                     header=["importance"], encoding="utf-8-sig")
    make_plots(predictions, split, figures, report["provisional"])
    joblib.dump(model, model_output)
    report = dict(report, model_output=str(model_output), model_sha256=sha256(model_output))
    (evaluation / "rf_basic_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    """스키마 확인·분할 검사·학습·평가·저장을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-split", action="store_true", help="1st Train / 2nd Validation / 3rd Test로 새 모델 학습")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-fraction", type=float, default=0.8, help="앞쪽 학습 측정 시점 비율")
    parser.add_argument("--gap-measurements", type=int, default=0, help="학습과 평가 사이에 제외할 측정 시점 수")
    parser.add_argument("--max-measurements", type=int, default=0, help="소량 점검용 앞쪽 측정 시점 수; 0은 전체")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=ROOT / "models/rf_basic.joblib")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.experiment_split:
        output = ROOT / "models/rf_basic_holdout.joblib" if args.model_output == ROOT / "models/rf_basic.joblib" else args.model_output
        return run_experiment_rf(args.data_dir,args.report_dir,output,args.trees,args.seed)
    if not 0 < args.train_fraction < 1 or args.gap_measurements < 0 or args.max_measurements < 0 or args.trees < 1:
        parser.error("학습 비율은 0과 1 사이, 시점 수는 0 이상, 트리 수는 1 이상이어야 합니다.")
    for name in ("input", "model_output", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.input.is_file():
        print(f"NASA IMS 특징 CSV가 필요합니다: {args.input}. 학습을 비활성화합니다. 저장된 평가 결과를 이용하세요.")
        return 0
    if args.max_measurements and (args.model_output == ROOT / "models/rf_basic.joblib"
                                  or args.report_dir == ROOT / "reports"):
        parser.error("소량 점검에는 별도 --model-output과 --report-dir을 지정하세요.")
    if args.model_output.suffix != ".joblib" or args.model_output == args.input:
        parser.error("입력과 다른 .joblib 모델 출력 경로를 지정하세요.")
    if args.model_output.exists() or (args.report_dir / "evaluation/rf_basic_metrics.json").exists() or (args.report_dir / "predictions/rf_basic_predictions.csv").exists():
        parser.error("기존 모델·평가 결과를 덮어쓰지 않습니다. 새 --model-output과 --report-dir을 지정하세요.")
    data, inspection = load_measurements(args.input)
    print("실제 CSV 컬럼:", inspection["csv_columns"], flush=True)
    print("데이터 확인:", json.dumps({k: v for k, v in inspection.items()
                                     if k not in ("csv_columns", "excluded_csv_columns")}, ensure_ascii=False), flush=True)
    if args.max_measurements:
        times = pd.Index(data["measured_at"].unique()).sort_values()[:args.max_measurements]
        # 소량 입력을 사용해도 정답 종료점을 표본의 마지막 시점으로 바꾸지 않는다.
        data = data.loc[data["measured_at"].isin(times)].copy()
    train, test, split, audit = chronological_split(data, args.train_fraction, args.gap_measurements)
    if train[MODEL_FEATURES].isna().all().any():
        raise ValueError("학습 전체가 결측인 신호 특징이 있습니다.")
    print("시간순 분할 검사:", json.dumps(audit, ensure_ascii=False), flush=True)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=5,
             max_features=1.0, random_state=args.seed, n_jobs=-1)),
    ])
    # 전처리와 회귀 모델 모두 앞쪽 학습 부분에만 적합한다. 평가 후 전체 재학습하지 않는다.
    model.fit(train[MODEL_FEATURES], train[TARGET])
    predictions = test[["measured_at", "bearing_id", "source_path", "measurement_index", TARGET]].copy()
    predictions["prediction_random_forest"] = model.predict(test[MODEL_FEATURES])
    median = float(train[TARGET].median())
    last_target = float(train.loc[train["measured_at"].eq(train["measured_at"].max()), TARGET].iloc[0])
    predictions["prediction_train_median"] = median
    predictions["prediction_last_train_target"] = last_target
    metrics = evaluate(predictions)
    rf_mae = metrics["overall"]["random_forest"]["MAE_hours"]
    last_mae = metrics["overall"]["last_train_target"]["MAE_hours"]
    comparison = ("Random Forest의 MAE가 마지막 학습 정답 고정 기준보다 작다." if rf_mae < last_mae else
                  "Random Forest의 MAE가 마지막 학습 정답 고정 기준보다 크거나 같다. 기준선 대비 개선을 입증하지 못했다.")
    report = {
        "input": str(args.input), "input_sha256": sha256(args.input), "script_sha256": sha256(Path(__file__)),
        "inspection": inspection, "signal_features": SIGNAL_FEATURES, "model_features": MODEL_FEATURES,
        "split_audit": audit, "train_fraction": args.train_fraction, "seed": args.seed,
        "provisional": bool(args.max_measurements),
        "evaluation_status": "소량 실행 점검용" if args.max_measurements else "1st Test 내부 회고적 시간순 홀드아웃",
        "model_parameters": model.named_steps["regressor"].get_params(),
        "imputation": "중앙값; 학습 행에만 적합", "refit_with_test": False,
        "training_missing_counts": {k: int(v) for k, v in train[MODEL_FEATURES].isna().sum().items()},
        "baseline_constants_hours": {"train_median": median, "last_train_target": last_target},
        "metrics": metrics, "comparison_note": comparison,
        "target_ranges_hours": {"train": [float(train[TARGET].min()), float(train[TARGET].max())],
                                "test": [float(test[TARGET].min()), float(test[TARGET].max())]},
        "prediction_range_hours": [float(predictions["prediction_random_forest"].min()),
                                   float(predictions["prediction_random_forest"].max())],
        "limitations": list(LIMITATIONS), "model_output": str(args.model_output),
        "training_experiments": list(TRAINING_EXPERIMENTS),
        "evaluation_protocol": "첫 실험 내부 회고적 시간순 분할; 실험 간 독립 평가 아님. 다른 실험을 합치지 않음.",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
    }
    if args.max_measurements:
        report["limitations"].append("소량 실행은 오류 확인용이며 이 지표를 최종 평가 결과로 사용하지 않는다.")
    save_results(report, predictions, split, model, args.report_dir, args.model_output)
    rows = [{"scope": scope, "method": name, **values}
            for scope, methods in metrics.items() for name, values in methods.items()]
    print("실행 점검 지표" if args.max_measurements else "1st Test 내부 회고적 평가 지표; 단위: 시간", flush=True)
    for row in rows:
        print(f"{row['scope']} / {METHODS[row['method']]}: MAE={row['MAE_hours']:.4f}, RMSE={row['RMSE_hours']:.4f}", flush=True)
    print(comparison, flush=True)
    for note in report["limitations"]:
        print("한계:", note, flush=True)
    print("모델 저장:", args.model_output, flush=True)
    print("평가 기록·그래프 저장:", args.report_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
