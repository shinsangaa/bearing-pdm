# RF 성능 비교

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 프로젝트 경로
def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = ROOT / "data/processed/features_enhanced.csv"

    report_file = ROOT / "reports/evaluation/rf_engineered_feature_comparison.csv"
    if report_file.exists():
        raise FileExistsError(f"Existing results: {report_file}")

    df = pd.read_csv(DATA_PATH)

    # RUL 정답
    TARGET = "rul_calendar_hours_proxy"

    # 기존 입력 변수
    BASE_FEATURES = [
        "rms",
        "std",
        "peak",
        "kurtosis",
        "crest_factor",
        "fft_centroid_hz"
    ]

    # 신규 입력 변수 5개
    NEW_FEATURES = [
        "rms_ma6",
        "rms_ma36",
        "rms_std36",
        "rms_diff6",
        "kurtosis_ma6"
    ]

    # 비교할 특징 구성
    FEATURE_SETS = {
        "기존 RF": BASE_FEATURES,
        "특징 추가 RF": BASE_FEATURES + NEW_FEATURES
    }

    # 필요한 컬럼 확인
    required = set(BASE_FEATURES + NEW_FEATURES)
    required.update(["test_id", TARGET])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"누락된 컬럼: {sorted(missing)}")

    # RUL 정답이 있는 데이터만 사용
    df = df[df[TARGET].notna()].copy()

    # 실험 단위 분리
    train = df[df["test_id"] == "1st_test"]
    valid = df[df["test_id"] == "2nd_test"]

    assert not train.empty, "Train 데이터가 없습니다."
    assert not valid.empty, "Validation 데이터가 없습니다."


    # Train RUL 평균만 사용하는 단순 기준 모델
    baseline_value = train[TARGET].mean()

    baseline_pred = np.full(
        len(valid),
        baseline_value
    )

    baseline_mae = mean_absolute_error(
        valid[TARGET], baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(valid[TARGET], baseline_pred)
    )

    baseline_r2 = r2_score(
        valid[TARGET], baseline_pred
    )

    print("\n[Train 평균 고정 기준 모델]")
    print("예측값:", round(baseline_value, 2))
    print("MAE:", round(baseline_mae, 2))
    print("RMSE:", round(baseline_rmse, 2))
    print("R2:", round(baseline_r2, 4))

    results = []

    for name, features in FEATURE_SETS.items():

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )

        # 학습
        model.fit(train[features], train[TARGET])

        # 검증 데이터 예측
        pred = model.predict(valid[features])


        # RF 예측값 분포 확인
        print(f"\n[{name}]")
        print("실제 RUL 평균:", round(valid[TARGET].mean(), 2))
        print("예측 RUL 평균:", round(np.mean(pred), 2))
        print("예측 RUL 최솟값:", round(np.min(pred), 2))
        print("예측 RUL 최댓값:", round(np.max(pred), 2))

        # 실제 RUL 최대값 초과 예측 확인
        exceed = np.sum(pred > valid[TARGET].max())

        print(
            "실제 최대 RUL 초과 예측 비율:",
            round(exceed / len(pred) * 100, 2), "%"
        )

        print(
            "실제값과 예측값의 평균 차이:",
            round(np.mean(pred - valid[TARGET]), 2),
            "시간"
        )

        # 성능 평가
        mae = mean_absolute_error(valid[TARGET], pred)
        rmse = np.sqrt(
            mean_squared_error(valid[TARGET], pred)
        )
        r2 = r2_score(valid[TARGET], pred)

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    result_df = pd.DataFrame(results)

    print("\nRF 특징 개선 전후 성능 비교")
    print(result_df.to_string(index=False))

    # 평가 결과 별도 저장
    report_dir = ROOT / "reports/evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        report_dir / "rf_engineered_feature_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # Train과 Validation의 특징 분포 비교

    CHECK_FEATURES = [
        "rms",
        "std",
        "peak",
        "kurtosis",
        "crest_factor",
        "fft_centroid_hz"
    ]

    rows = []

    for feature in CHECK_FEATURES:
        rows.append({
            "변수": feature,
            "Train 중앙값": train[feature].median(),
            "Valid 중앙값": valid[feature].median(),
            "Train 평균": train[feature].mean(),
            "Valid 평균": valid[feature].mean(),
            "Train 최댓값": train[feature].max(),
            "Valid 최댓값": valid[feature].max()
        })

    distribution_df = pd.DataFrame(rows)

    print("\nTrain·Validation 특징 분포 비교")
    print(distribution_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
