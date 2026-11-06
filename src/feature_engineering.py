# RF 성능 개선 위해 최근 변동성 관련 변수 추가
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/processed")

# 기존 전처리 CSV 탐색
files = []

for path in DATA_DIR.rglob("*.csv"):
    # 새로 만든 결과 CSV는 제외
    if path.stem.endswith("_enhanced"):
        continue

    cols = set(pd.read_csv(path, nrows=0).columns)

    required = {
        "test_id", "bearing_id",
        "measured_at", "rms", "kurtosis"
    }

    if required.issubset(cols):
        files.append(path)

print("발견된 CSV:", files)

if not files:
    raise FileNotFoundError("전처리 CSV를 찾지 못했습니다.")

df = pd.concat(
    [pd.read_csv(p) for p in files],
    ignore_index=True
)

# 데이터 타입 정리
df["measured_at"] = pd.to_datetime(df["measured_at"])
df["bearing_id"] = pd.to_numeric(df["bearing_id"])
df["rms"] = pd.to_numeric(df["rms"])
df["kurtosis"] = pd.to_numeric(df["kurtosis"])

# 첫 번째 센서 선택
sensor_col = (
    "sensor_within_bearing"
    if "sensor_within_bearing" in df.columns
    else "channel_id"
)

df[sensor_col] = pd.to_numeric(
    df[sensor_col], errors="coerce"
)

group_cols = ["test_id", "bearing_id"]

first_sensor = df.groupby(group_cols)[sensor_col].transform("min")

df = df[
    df[sensor_col].eq(first_sensor) |
    df[sensor_col].isna()
].copy()

# 시각별 중복 검증
keys = ["test_id", "bearing_id", "measured_at"]

if df.duplicated(keys).any():
    raise ValueError(
        "중복된 측정 데이터가 있습니다. "
        "입력 CSV가 중복 포함됐는지 확인하세요."
    )

# 시계열 정렬
df = df.sort_values(keys).reset_index(drop=True)

print(df.groupby(group_cols).size())




# 이동평균, 변동성, 변화량 계산
# 20분 초과 측정 공백은 새로운 구간으로 분리
groups = ["test_id", "bearing_id"]

time_gap = (
    df.groupby(groups)["measured_at"]
      .diff()
      .dt.total_seconds()
)

new_segment = time_gap.isna() | (time_gap > 1200)

# 중간 측정 파일 누락도 경계로 처리
if "measurement_index" in df.columns:
    idx = pd.to_numeric(
        df["measurement_index"], errors="coerce"
    )

    idx_gap = idx.groupby(
        [df["test_id"], df["bearing_id"]]
    ).diff()

    new_segment |= idx_gap.notna() & idx_gap.ne(1)

df["_segment"] = (
    new_segment.groupby(
        [df["test_id"], df["bearing_id"]]
    ).cumsum()
)

# 실험·베어링·연속 측정구간별 계산
history = df.groupby(
    ["test_id", "bearing_id", "_segment"],
    sort=False
)

# 1. 단기 RMS 이동평균
df["rms_ma6"] = history["rms"].transform(
    lambda s: s.rolling(6, min_periods=1).mean()
)

# 2. 장기 RMS 이동평균
df["rms_ma36"] = history["rms"].transform(
    lambda s: s.rolling(36, min_periods=1).mean()
)

# 3. RMS 변동성
df["rms_std36"] = history["rms"].transform(
    lambda s: s.rolling(36, min_periods=2).std()
).fillna(0)

# 4. RMS 변화량
df["rms_diff6"] = history["rms"].diff(6).fillna(0)

# 5. Kurtosis 이동평균
df["kurtosis_ma6"] = history["kurtosis"].transform(
    lambda s: s.rolling(6, min_periods=1).mean()
)

NEW_FEATURES = [
    "rms_ma6",
    "rms_ma36",
    "rms_std36",
    "rms_diff6",
    "kurtosis_ma6"
]

print(df[["test_id", "bearing_id", *NEW_FEATURES]].head(10))
print("\n결측값:")
print(df[NEW_FEATURES].isna().sum())


# 새로운 CSV 저장
output = DATA_DIR / "features_enhanced.csv"

# 내부 계산용 구간 컬럼 제외
save_df = df.drop(columns=["_segment"])

save_df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("저장 완료:", output.resolve())
print("전체 행:", len(save_df))
print("추가 특징:", NEW_FEATURES)

