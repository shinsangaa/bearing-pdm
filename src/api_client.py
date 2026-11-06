"""Streamlit 전용 API 호출. API 중단은 저장 결과·RMS 표시와 분리한다."""
import json
import os
from pathlib import Path

import numpy as np
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def api_base_url():
    value = os.getenv("API_BASE_URL")
    if not value:
        try:
            value = st.secrets.get("API_BASE_URL")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            value = None
    return (value or "http://localhost:8000").rstrip("/")


def request_prediction(model_id, features):
    response = requests.post(api_base_url()+"/predict/rul", json={"model": model_id, "features": features},
                             timeout=(3.05, 20))
    if not response.ok:
        raise ValueError(f"API HTTP {response.status_code}: {response.text[:500]}")
    result = response.json()
    if "prediction_hours" not in result or not np.isfinite(result["prediction_hours"]):
        raise ValueError("API 응답에 유효한 예측값이 없습니다.")
    return result


def features_at_time(frame, model_id, bearing, index, names):
    group = frame.loc[frame.bearing_id.eq(bearing)].sort_values("measured_at")
    times = group.measured_at.drop_duplicates().tolist()
    section = group.loc[group.measured_at.eq(times[index])]
    if model_id == "rf_basic":
        features = {}
        for name in names:
            sensor_tag, feature = name.split("_", 1)
            selected = section.loc[section.sensor_within_bearing.eq(int(sensor_tag.removeprefix("sensor")))]
            if len(selected) != 1:
                raise ValueError("RF Basic은 첫 시험의 두 센서가 모두 필요합니다.")
            features[name] = float(selected.iloc[0][feature])
    elif model_id == "rf_basic_holdout":
        features = {name: float(section[name].mean()) for name in names}
    else:
        selected = section.loc[section.sensor_within_bearing.eq(1)]
        if len(selected) != 1:
            raise ValueError("Ridge 학습과 동일한 첫 센서가 필요합니다.")
        features = {name: float(selected.iloc[0][name]) for name in names}
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("선택한 측정에 결측·무한대 특징값이 있어 API로 보내지 않았습니다.")
    return features, times[index]


def render_api_prediction(experiment, frame):
    with st.expander("새 RUL 예측 · FastAPI", expanded=False):
        registry = json.loads((ROOT / "deployment/model_manifest.json").read_text(encoding="utf-8"))["models"]
        st.caption("버튼을 누를 때만 API를 호출합니다. API가 중단되어도 저장 결과와 RMS는 계속 표시됩니다.")
        model_id = st.selectbox("API 모델", ["ridge", "rf_basic", "rf_basic_holdout"], key="api_model")
        spec = registry[model_id]
        st.caption(spec["input_scope"]+" · "+spec["note"])
        allowed = experiment == "1st_test" if model_id == "rf_basic" else True
        if not allowed:
            st.info("RF Basic 내부 시간순 모델은 1st_test에서 두 센서를 선택해 사용합니다.")
        with st.form("api_prediction_form"):
            bearing = st.selectbox("API 입력 베어링", [3, 4] if model_id == "rf_basic" else [1, 2, 3, 4],
                                   key="api_bearing")
            count = frame.measured_at.nunique()
            index = st.slider("API 입력 측정 순번", 0, count-1, min(count-1, count//2), key="api_index")
            submitted = st.form_submit_button("FastAPI로 새 예측", disabled=not allowed)
        st.caption("특징 추가 RF는 가중치가 저장되지 않아 새 추론을 지원하지 않습니다. 기존 비교표는 위에서 확인할 수 있습니다.")
        if submitted:
            try:
                features, timestamp = features_at_time(frame, model_id, bearing, index, spec["features"])
                with st.spinner("FastAPI 추론 요청 중..."):
                    result = request_prediction(model_id, features)
                st.metric("새 예측 · 시험 종료 대리 RUL", f"{result['prediction_hours']:.2f}시간")
                st.caption(f"입력: {experiment} B{bearing} · {timestamp} · 특징 {len(features)}개")
                st.info(result["notice"])
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                st.warning(f"API 예측을 사용할 수 없습니다: {exc}. 저장 결과와 RMS는 계속 사용할 수 있습니다.")
