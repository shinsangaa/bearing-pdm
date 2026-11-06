# 모델 입력 변수 관리

BASE_FEATURES = [
    "rms",
    "std",
    "peak",
    "kurtosis",
    "crest_factor",
    "fft_centroid_hz"
]

NEW_FEATURES = [
    "rms_ma6",
    "rms_ma36",
    "rms_std36",
    "rms_diff6",
    "kurtosis_ma6"
]

ENHANCED_FEATURES = BASE_FEATURES + NEW_FEATURES
