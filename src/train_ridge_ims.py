"""NASA IMS Ridge 회귀: 실험별 분리 검증과 전체 학습 (탐색적 평가).

프로젝트 루트에서 실행:
    python src/train_ridge_ims.py

필요: pandas, numpy, scikit-learn, joblib
입력: data/processed/features_enhanced.csv
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/features_enhanced.csv"
TARGET = "rul_calendar_hours_proxy"
FEATURES = [
    "rms", "std", "peak", "kurtosis", "crest_factor", "fft_centroid_hz"
]
TESTS = ("1st_test", "2nd_test", "3rd_test")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def make_model(alpha):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),  # Ridge는 특징의 크기가 달라 반드시 스케일링
        Ridge(alpha=alpha),
    )


def evaluate(label, truth, raw_pred):
    # 실제 RUL은 음수가 아니므로, 0 이상으로 제한 (테스트 결과 보고 선택한 규칙 아님)
    prediction = np.maximum(np.asarray(raw_pred, dtype=float), 0.0)
    truth = np.asarray(truth, dtype=float)
    scores = {
        "모델": label,
        "MAE": mean_absolute_error(truth, prediction),
        "RMSE": np.sqrt(mean_squared_error(truth, prediction)),
        "R2": r2_score(truth, prediction),
        "예측평균": prediction.mean(),
    }
    return scores, prediction


def compare_baselines(train, test, label):
    y = test[TARGET].to_numpy()
    rows = []
    for name, guess in [
        ("항상 0시간", 0.0),
        ("Train 평균 고정", train[TARGET].mean()),
    ]:
        metric, _ = evaluate(name, y, np.full(len(test), guess))
        rows.append(metric)
    return rows


def main():
    # 파일 정리 후에도 학습 실행이 보존된 최종 결과를 덮어쓰지 않게 한다.
    targets = [ROOT / "reports" / name for name in (
        "evaluation/ridge_validation_alpha.csv", "evaluation/ridge_leave_one_test_out.csv",
        "predictions/ridge_test3_predictions.csv")] + [ROOT / "models/ridge_all_tests.joblib"]
    if any(path.exists() for path in targets):
        raise FileExistsError("기존 Ridge 결과가 있습니다. 결과를 보존하기 위해 학습을 중단합니다.")
    if not DATA.exists():
        raise FileNotFoundError(f"데이터가 없습니다: {DATA}")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    required = ["test_id", "bearing_id", "measured_at", TARGET, *FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    for name in FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if df.duplicated(["test_id", "bearing_id", "measured_at"]).any():
        raise ValueError("베어링·시각 중복 행이 있습니다. 첫 번째 센서만 선택했는지 확인하세요.")
    for test_id in TESTS:
        if not (df["test_id"] == test_id).any():
            raise ValueError(f"{test_id}의 RUL 라벨 행이 없습니다.")

    print("\n실험별 정답 분포 (실제 고장시간이 아닌 시험 종료까지 남은 대리 RUL)")
    print(df.groupby("test_id")[TARGET].agg(["count", "min", "max", "mean"]).round(2).to_string())

    # A: 1st 학습, 2nd 검증. 검증 MAE로 alpha 선택
    train = df[df.test_id == "1st_test"]
    valid = df[df.test_id == "2nd_test"]
    print("\n[A] 1st 학습 → 2nd 검증 (alpha 선택)")
    results = compare_baselines(train, valid, "A")
    for alpha in ALPHAS:
        model = make_model(alpha)
        model.fit(train[FEATURES], train[TARGET])
        score, _ = evaluate(f"Ridge alpha={alpha:g}", valid[TARGET], model.predict(valid[FEATURES]))
        results.append(score)
    table_a = pd.DataFrame(results)
    print(table_a.round(3).to_string(index=False))
    ridge_rows = table_a[table_a["모델"].str.startswith("Ridge")]
    best_row = ridge_rows.loc[ridge_rows["MAE"].idxmin()]
    alpha = float(best_row["모델"].split("=")[-1])
    print(f"선택 alpha={alpha:g} (2nd Validation MAE 최소)")

    # B: 모델 선택 끝낸 후 1st+2nd로 재학습, 3rd에 적용
    train12 = df[df.test_id.isin(["1st_test", "2nd_test"])]
    test3 = df[df.test_id == "3rd_test"]
    model12 = make_model(alpha)
    model12.fit(train12[FEATURES], train12[TARGET])
    score_b, predictions3 = evaluate("Ridge", test3[TARGET], model12.predict(test3[FEATURES]))
    print("\n[B] 1st+2nd 학습 → 3rd 평가 (이미 3rd 결과를 봤으므로 탐색적 평가)")
    print(pd.DataFrame(compare_baselines(train12, test3, "B") + [score_b]).round(3).to_string(index=False))

    report_dir = ROOT / "reports"
    for folder in ("evaluation", "predictions"):
        (report_dir / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_a).to_csv(report_dir / "evaluation/ridge_validation_alpha.csv", index=False, encoding="utf-8-sig")
    prediction_rows = test3[["test_id", "bearing_id", "measured_at", TARGET]].copy()
    prediction_rows["ridge_prediction_hours"] = predictions3
    prediction_rows.to_csv(report_dir / "predictions/ridge_test3_predictions.csv", index=False, encoding="utf-8-sig")

    # C: 3개 실험 모두 활용. 각 실험을 한 번씩 제외하고 훈련한 뒤, 최종 전체로 학습
    print("\n[C] 실험 하나씩 제외하는 3회 평가 (설정 선택에 2nd 사용: 독립적 최종 검증 아님)")
    fold_rows = []
    for holdout in TESTS:
        tr = df[df.test_id != holdout]
        te = df[df.test_id == holdout]
        fold_model = make_model(alpha)
        fold_model.fit(tr[FEATURES], tr[TARGET])
        metrics, _ = evaluate(holdout, te[TARGET], fold_model.predict(te[FEATURES]))
        fold_rows.append(metrics)
    fold_table = pd.DataFrame(fold_rows)
    print(fold_table.round(3).to_string(index=False))
    fold_table.to_csv(report_dir / "evaluation/ridge_leave_one_test_out.csv", index=False, encoding="utf-8-sig")

    final_model = make_model(alpha)
    final_model.fit(df[FEATURES], df[TARGET])
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "ridge_all_tests.joblib"
    joblib.dump({"model": final_model, "features": FEATURES, "target": TARGET,
                 "alpha": alpha, "warning": "시험 종료 기준 대리 RUL, 독립적 최종 테스트 없음"}, target)
    print(f"\n세 실험 모두 학습한 최종 모델 저장: {target}")
    print(f"평가 리포트: {report_dir}")
    print("주의: 전체 데이터를 학습한 최종 모델은 독립 평가 데이터가 없어 일반화 성능을 주장할 수 없습니다.")


if __name__ == "__main__":
    main()
