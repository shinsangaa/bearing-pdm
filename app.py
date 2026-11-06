import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 기존 모듈 임포트 유지
from src.demo_support import ROOT, load_monitoring, load_report, reference_status
from src.preprocessing import EXPERIMENTS, THIRD_RELEASE_NOTE, read_measurement, resolve_source_path
from src.signal_processing import signal_figure
from src.api_client import render_api_prediction
from src.cloud_waterfall import load_cloud_waterfall, cloud_support_message, supported_cloud_configs
from src.ims_alarm import (compute_alarm_thresholds, alarm_level, find_first_alarm,
                           validate_alarm_frame, sustained_alarm_states, rms_figure, compare_series)


def inject_custom_css(active_page: str):
    """사이드바 메뉴에 기업형 호버·선택 스타일을 적용한다."""
    # st.button(key=...)의 키는 st-key-... CSS 클래스로 노출된다.
    # 기존 radio DOM에 의존하는 CSS 대신 버튼 스타일만 조정한다.
    key_by_page = {
        "Home": "nav_home",
        "Chart": "nav_chart",
        "Logs": "nav_logs",
        "User Info": "nav_user_info",
    }
    active_key = key_by_page.get(active_page, "nav_home")
    st.markdown(f"""
        <style>
            /* 이 앱의 사이드바 메뉴에만 적용한다. 다른 버튼은 건드리지 않는다. */
            [data-testid="stSidebar"] [class*="st-key-nav_"] button {{
                width: 100%;
                min-height: 46px;
                justify-content: flex-start;
                text-align: left;
                padding: 10px 15px;
                border-radius: 9px;
                border: 1px solid transparent;
                border-left: 4px solid transparent;
                background: transparent !important;
                color: inherit !important;
                box-shadow: none !important;
                font-weight: 500;
                cursor: pointer;
                transition: background-color 160ms ease,
                            color 160ms ease,
                            border-color 160ms ease,
                            transform 160ms ease;
            }}
            [data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {{
                background: rgba(46, 111, 209, 0.12) !important;
                color: #1763c2 !important;
                border-color: rgba(46, 111, 209, 0.18) !important;
                border-left-color: rgba(46, 111, 209, 0.45) !important;
                transform: translateX(3px);
            }}
            [data-testid="stSidebar"] [class*="st-key-nav_"] button:focus-visible {{
                outline: 2px solid #2d70d3 !important;
                outline-offset: 2px;
            }}
            /* 현재 선택 메뉴: rerun 후에도 session_state를 사용해 유지한다. */
            [data-testid="stSidebar"] .st-key-{active_key} button,
            [data-testid="stSidebar"] .st-key-{active_key} button:hover {{
                background: rgba(42, 112, 215, 0.18) !important;
                color: #1457ad !important;
                border-left: 4px solid #236fcb !important;
                font-weight: 700 !important;
                transform: none;
            }}
            @media (prefers-reduced-motion: reduce) {{
                [data-testid="stSidebar"] [class*="st-key-nav_"] button {{
                    transition: none;
                }}
            }}
        </style>
    """, unsafe_allow_html=True)


def set_active_page(page: str):
    """클릭한 메뉴를 저장해 다음 rerun에도 선택 상태를 유지한다."""
    st.session_state.active_page = page


@st.cache_data(show_spinner=False, ttl=120)
def load_ridge_predictions():
    """1st+2nd 학습 → 3rd 평가에서 저장된 예측 파일을 읽는다. 재학습 없음."""
    path = ROOT / "reports/predictions/ridge_test3_predictions.csv"
    if not path.is_file():
        return None, f"RUL 예측 파일이 없습니다: {path}"
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
        required = {"measured_at", "bearing_id", "rul_calendar_hours_proxy", "ridge_prediction_hours"}
        missing = required - set(data.columns)
        if missing:
            return None, f"RUL 예측 CSV 필수 열 누락: {sorted(missing)}"
        if "test_id" in data.columns:
            data = data[data["test_id"].eq("3rd_test")].copy()
        bearing = pd.to_numeric(data["bearing_id"], errors="coerce")
        data = data.loc[bearing.eq(3)].copy()
        data["measured_at"] = pd.to_datetime(data["measured_at"], errors="coerce")
        for col in ("rul_calendar_hours_proxy", "ridge_prediction_hours"):
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["measured_at", "rul_calendar_hours_proxy", "ridge_prediction_hours"])
        if data.empty:
            return None, "3rd_test Bearing 3에 표시 가능한 RUL 데이터가 없습니다."
        if data.duplicated("measured_at").any():
            return None, "같은 측정 시각의 중복 RUL 예측값이 있습니다. CSV를 확인하세요."
        return data.sort_values("measured_at").reset_index(drop=True), None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return None, f"예측 CSV를 읽지 못했습니다: {exc}"



@st.cache_data(show_spinner=False, ttl=300, max_entries=3)
def load_monitoring_cached(experiment):
    """재실행 시 CSV 재로딩 및 병합을 반복하지 않는다."""
    return load_monitoring(experiment)

def page_home():
    """메인: 저장된 Ridge RUL 시각화와 NASA IMS 관측 진동 비교."""
    st.header("🏢 설비 예지보전 모니터링 대시보드")
    st.caption("NASA IMS 기록 데이터 기반 시연 · 실시간 현장 설비 상태가 아닙니다.")

    # 저장 예측 데이터는 3rd_test B3만 존재한다. 이 선택을 기본값으로 하여 차트가 바로 보이게 함.
    choices = list(EXPERIMENTS)
    experiment = st.selectbox(
        "분석할 실험(Test) 구간", choices,
        index=choices.index("3rd_test") if "3rd_test" in choices else 0,
        key="home_experiment",
    )
    predictions, prediction_error = load_ridge_predictions() if experiment == "3rd_test" else (None, None)

    # 해당 시점의 대리 RUL과 저장된 예측을 비교한다. 최신 예측/실시간 서비스로 주장하지 않는다.
    selected = None
    if predictions is not None:
        st.subheader("⏳ RUL 예측 시점 선택 — 3rd_test Bearing 3")
        chosen_index = st.slider(
            "측정 순번 (과거 시험 기록)",
            min_value=0, max_value=len(predictions)-1,
            value=min(int(len(predictions)*0.8), len(predictions)-1),
            key="ridge_measurement_index",
        )
        selected = predictions.iloc[chosen_index]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("선택한 실험", experiment)
    col2.metric("예측 대상", "Bearing 3" if experiment == "3rd_test" else "선택 실험")
    col3.metric(
        "모델 예측 RUL (참고)",
        f"{selected['ridge_prediction_hours']:,.1f}시간" if selected is not None else "데이터 없음",
    )
    col4.metric(
        "예측 잔여일 환산 (참고)",
        f"{selected['ridge_prediction_hours'] / 24:,.1f}일" if selected is not None else "데이터 없음",
    )

    if selected is not None:
        st.caption(
            f"관측 시점: {selected['measured_at']:%Y-%m-%d %H:%M}  |  "
            f"실제 시험 종료까지 남은 대리 RUL: {selected['rul_calendar_hours_proxy']:,.1f}시간  |  "
            f"모델 예측: {selected['ridge_prediction_hours']:,.1f}시간. "
            "환산 일수는 24시간으로 나눈 값일 뿐, 교체 권장일이 아닙니다."
        )
        st.subheader("📉 실제 대리 RUL vs Ridge 예측 RUL")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions["measured_at"], y=predictions["rul_calendar_hours_proxy"],
            mode="lines", name="실제 대리 RUL", line=dict(color="#2563eb", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=predictions["measured_at"], y=predictions["ridge_prediction_hours"],
            mode="lines", name="Ridge 예측 RUL", line=dict(color="#f97316", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[selected["measured_at"]], y=[selected["ridge_prediction_hours"]],
            mode="markers", name="선택한 시점", marker=dict(color="#dc2626", size=10),
        ))
        fig.update_layout(
            xaxis_title="측정 시각", yaxis_title="남은 시간 (h)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=15, r=15, t=35, b=15),
            hovermode="x unified", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.warning(
            "주의: 이 그래프는 1st+2nd_test로 학습한 Ridge의 3rd_test 과거 평가 결과입니다. "
            "3rd_test의 MAE 340.85시간, R² -0.864로 정확한 실제 잔여수명이나 교체일 산정에 "
            "사용할 수 없습니다. 정답도 실제 고장 시점이 아닌 시험 종료 기준 대리값입니다."
        )
    elif experiment == "3rd_test":
        st.info(prediction_error or "저장된 3rd_test RUL 예측을 불러오지 못했습니다.")
        st.caption("먼저 `python src/train_ridge_ims.py`를 실행하여 reports/predictions/ridge_test3_predictions.csv를 생성하세요.")
    else:
        st.info("현재 저장된 Ridge의 독립 실험 평가 예측 CSV는 3rd_test Bearing 3에만 있습니다. RUL 차트를 보려면 3rd_test를 선택하세요.")

    with st.expander("RF Basic · RF Engineered 저장 결과", expanded=False):
        try:
            basic = load_report("reports/evaluation/rf_basic_metrics.json")
            holdout = load_report("reports/evaluation/rf_basic_holdout_metrics.json")
            st.dataframe(pd.DataFrame([
                {"평가": "RF Basic · 1st 내부 시간순", **basic["metrics"]["overall"]["random_forest"]},
                {"평가": "RF Basic · 실험별 분리 Test", **holdout["metrics"]["test"]},
            ]), use_container_width=True)
            figure_name = ("rf_basic_predictions.png" if experiment == "1st_test"
                           else "rf_basic_holdout_test.png" if experiment == "3rd_test" else None)
            if figure_name:
                st.image(str(ROOT / "reports/figures" / figure_name), caption="저장된 RF Basic 평가 예측")
            st.write("RF Engineered · 기존 특징과 추가 특징의 Validation 비교")
            st.dataframe(pd.read_csv(ROOT / "reports/evaluation/rf_engineered_feature_comparison.csv"),
                         use_container_width=True)
            st.caption("평가 실험·특징·행이 달라 위 수치를 같은 조건의 모델 순위로 비교할 수 없습니다. "
                       "특징 추가 RF는 성능 비교표만 저장되어 있으며 개별 예측 CSV와 가중치는 없습니다.")
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"저장 RF 결과를 읽지 못했습니다: {exc}")

    st.divider()
    st.subheader("📊 전체 베어링 진동 추이 (RMS 비교)")
    try:
        frame = load_monitoring_cached(experiment)
    except (OSError, ValueError) as exc:
        st.warning(f"RMS 데이터를 불러올 수 없습니다: {exc}")
        return

    if frame.empty:
        st.info("해당 실험의 RMS 데이터가 없습니다.")
        return

    pivot_df = frame.pivot_table(index="measured_at", columns="bearing_id", values="rms")
    pivot_df.columns = [f"Bearing {col}" for col in pivot_df.columns]
    st.line_chart(pivot_df, use_container_width=True)
    st.caption(f"선택된 실험: {experiment}. 실제 진동 특징 CSV 기반 관측 추이입니다.")
    render_api_prediction(experiment, frame)




def compute_fft_spectrum(signal, fs=20000, max_frequency_hz=5000, n_frequency_bins=192):
    """단일 실제 측정 신호의 Hann 창 FFT. 주파수 구간별 최대 진폭으로 표면 시각화."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("진동 신호가 비어 있거나 비정상적입니다.")
    x = x - x.mean()  # 센서의 DC 오프셋 제거
    window = np.hanning(len(x))
    amplitude = np.abs(np.fft.rfft(x * window)) * (2.0 / window.sum())
    if len(x) % 2 == 0:
        amplitude[-1] /= 2
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / fs)
    chosen = np.flatnonzero((frequencies > 0) & (frequencies <= max_frequency_hz))
    if len(chosen) < 2:
        raise ValueError("선택한 주파수 범위에 FFT 값이 없습니다.")
    # 단순 간격 샘플링은 좁은 피크를 놓칠 수 있으므로 구간별 최대값 사용.
    chunks = np.array_split(chosen, min(len(chosen), n_frequency_bins))
    x_freq = np.array([frequencies[c].mean() for c in chunks])
    z_amplitude = np.array([amplitude[c].max() for c in chunks])
    return x_freq, z_amplitude


def prepare_alarm_frame(experiment, bearing, sensor):
    """합성값 없이 전처리된 원본 측정 RMS로 시점·센서별 순서를 확보한다."""
    raw_frame = load_monitoring_cached(experiment)
    required = {"bearing_id", "sensor_within_bearing", "measured_at", "rms", "source_path"}
    missing = required - set(raw_frame.columns)
    if missing:
        raise ValueError(f"전처리 CSV 필수 컬럼 누락: {sorted(missing)}")
    frame = raw_frame.loc[
        pd.to_numeric(raw_frame["bearing_id"], errors="coerce").eq(bearing)
        & pd.to_numeric(raw_frame["sensor_within_bearing"], errors="coerce").eq(sensor)
    ].copy()
    return validate_alarm_frame(frame)


def build_alarm_event_waterfall(experiment, bearing, sensor, frame, event_index,
                                history_points=16, max_frequency_hz=5000):
    """실제 원본 TXT 16개 이하를 읽어 트리거 직전~당시의 FFT만 계산한다."""
    channel = EXPERIMENTS[experiment]["channels"][bearing][sensor - 1]
    first = max(0, event_index - 47)  # 경보 직전 48개 측정값 범위
    positions = np.unique(np.linspace(first, event_index,
                                      min(history_points, event_index - first + 1), dtype=int))
    spectra, times, missing, mismatches = [], [], [], []
    frequencies = None
    for idx in positions:
        row = frame.iloc[int(idx)]
        raw = resolve_source_path(str(row["source_path"]))
        if raw is None or not raw.is_file():
            missing.append(f"{row['source_path']}: 원본 파일 없음")
            continue
        try:
            measurement = read_measurement(raw, experiment)
            if measurement.ndim != 2 or channel > measurement.shape[1]:
                raise ValueError(f"채널 {channel} 누락 / 원본 shape={measurement.shape}")
            signal = measurement[:, channel-1]
            real_rms = float(np.sqrt(np.mean(np.asarray(signal, dtype=float) ** 2)))
            if not np.isclose(real_rms, row["rms"], rtol=0.02, atol=1e-4):
                mismatches.append(f"{row['measured_at']}: 원본 RMS {real_rms:.6f}, CSV RMS {row['rms']:.6f}")
            f, amp = compute_fft_spectrum(signal, fs=20000, max_frequency_hz=max_frequency_hz,
                                          n_frequency_bins=160)
            frequencies = f
            spectra.append(amp)
            times.append(pd.Timestamp(row["measured_at"]))
        except (OSError, ValueError, TypeError, IndexError) as exc:
            missing.append(f"{raw}: {exc}")

    if mismatches:
        # CSV/원본 센서 매핑이 일치하지 않으면 그 결과를 신뢰하지 않고 차단한다.
        raise ValueError("원본 TXT와 전처리 RMS가 일치하지 않습니다. 센서·경로 매핑 확인: "
                         + mismatches[0])
    if len(spectra) < 2:
        raise ValueError("워터폴 FFT에 필요한 실제 원본 TXT를 2개 이상 읽지 못했습니다. "
                         + (missing[0] if missing else ""))
    hours = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
    return {
        "frequency": frequencies, "hours": hours, "spectra": np.asarray(spectra),
        "times": times, "missing": missing, "channel": channel,
    }


def render_alarm_result(result):
    """경보 시에만 원본 FFT의 워터폴을 화면에 표시한다."""
    st.divider()
    st.subheader("🔍 경보 발생 후 실제 FFT 워터폴")
    threshold = result["thresholds"]
    st.plotly_chart(rms_figure(result["frame"], threshold, result["consecutive"],
                              "전체 RMS · 임계선 · 단계별 최초 지속 경보"), use_container_width=True)
    st.caption(f"기준 후보 {threshold['baseline_n']}개 / 유효 {threshold['baseline_valid_n']}개 · "
               f"연속 {result['consecutive']}회 초과로 경보 확정 · {threshold['baseline_review']}")
    label = {"Warning": "주의", "Alarm": "경고", "Critical": "위험"}[result["trigger_level"]]
    if result["trigger_index"] is None:
        st.info(f"선택한 기록에서는 '{label}' 트리거에 도달하지 않았습니다. 워터폴을 표시하지 않습니다.")
        return
    st.error(
        f"{result['experiment']} · B{result['bearing']} · 센서 {result['sensor']}  |  "
        f"{result['time']:%Y-%m-%d %H:%M:%S}  |  "
        f"RMS={result['rms']:.5f} · Alarm 임계값={threshold['alarm']:.5f} · "
        f"경보 단계={result['level']} (재생 일시 중단)"
    )
    waterfall = result.get("waterfall")
    if not waterfall:
        st.warning(result.get("waterfall_error", "원본 FFT를 생성할 수 없었습니다."))
        return
    if waterfall.get("source") == "precomputed_actual_txt":
        st.info("클라우드 표시: 이 경보 사건의 실제 NASA TXT로 사전 계산·검증한 FFT입니다. "
                "현재 서버에서 원본 TXT를 다시 읽은 결과가 아닙니다.")
    fig = go.Figure(data=[go.Surface(
        x=waterfall["frequency"], y=waterfall["hours"], z=waterfall["spectra"],
        colorscale="Viridis", colorbar=dict(title="FFT 진폭"),
    )])
    fig.update_layout(
        title="Actual FFT Spectrum History — Alarm Event", height=540,
        margin=dict(l=5, r=5, b=5, t=50),
        scene=dict(
            xaxis_title="주파수 (Hz)",
            yaxis_title="선택한 측정 구간의 경과 시간 (h)",
            zaxis_title="FFT 진폭 (원본 신호 단위)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"NASA IMS 실제 원본 TXT {len(waterfall['times'])}개 FFT · "
        f"{waterfall['times'][0]:%Y-%m-%d %H:%M} ~ {waterfall['times'][-1]:%Y-%m-%d %H:%M} · "
        "Hann 창·DC 제거·주파수 구간별 최대 진폭. "
        "과거 기록을 재생하는 시연이며 실제 설비의 자동 정지 명령이 아닙니다."
    )
    if waterfall["missing"]:
        st.warning(f"원본 파일 읽기 실패 {len(waterfall['missing'])}개가 FFT에서 제외되었습니다. "
                   f"첫 실패: {waterfall['missing'][0]}")
    st.info("RMS 임계값은 초기 측정 구간에서 계산한 시연용 참고 기준입니다. "
            "워터폴은 실제 FFT이지만 특정 주파수의 증가가 곧 결함 확정이라는 뜻은 아닙니다.")


@st.cache_data(show_spinner=False, ttl=300, max_entries=2)
def load_alarm_comparison():
    tables, errors = [], []
    for experiment, spec in EXPERIMENTS.items():
        for bearing, channels in spec["channels"].items():
            for sensor in range(1, len(channels) + 1):
                try:
                    frame = prepare_alarm_frame(experiment, bearing, sensor)
                    tables.append(compare_series(frame, experiment, bearing, sensor))
                except (OSError, ValueError, KeyError) as exc:
                    errors.append(f"{experiment} B{bearing} S{sensor}: {exc}")
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), errors


def render_comparison(multiplier, consecutive):
    with st.expander("전체 베어링·센서: MAD 5/7/10 × 지속 1/3/5 비교", expanded=False):
        table, errors = load_alarm_comparison()
        for error in errors:
            st.warning(error)
        st.caption("1회 조건은 순간 피크 민감도 비교용입니다. 재생·워터폴은 선택한 3~5회 조건을 적용합니다. "
                   "Test 1 B3/B4, Test 2 B1, Test 3 B3은 종료 결함 보고 대상이며 나머지는 종료 결함 보고가 없습니다. "
                   "이 정보는 판정에 쓰지 않으며, 무보고를 정상 정답 또는 보고 대상을 고장 시작 시각의 정답으로 간주하지 않습니다.")
        if not table.empty:
            st.write(f"선택 조건: MAD {multiplier} · 연속 {consecutive}회 (전체 센서·단계)")
            selected = table.loc[table.mad_multiplier.eq(multiplier) & table.consecutive.eq(consecutive)]
            if consecutive == 4:
                selected_tables = []
                for test_id, spec in EXPERIMENTS.items():
                    for bearing, channels in spec["channels"].items():
                        for sensor in range(1, len(channels)+1):
                            try:
                                frame = prepare_alarm_frame(test_id, bearing, sensor)
                                selected_tables.append(compare_series(frame, test_id, bearing, sensor,
                                                                      (multiplier,), (consecutive,)))
                            except (OSError, ValueError, KeyError):
                                pass  # 동일 센서 오류는 위의 전체 비교에서 이미 표시했다.
                selected = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
            st.dataframe(selected, use_container_width=True)
            st.write("전체 9개 조합")
            st.dataframe(table, use_container_width=True)
            st.download_button("전체 조합 CSV 내려받기", table.to_csv(index=False).encode("utf-8-sig"),
                               "ims_alarm_comparison.csv", "text/csv")
            summary = table.groupby(["mad_multiplier", "consecutive", "level"], sort=False).agg(
                series_count=("alarm_occurred", "size"), alarms=("alarm_occurred", "sum"),
                median_remaining_hours=("remaining_calendar_hours", "median")).reset_index()
            st.dataframe(summary, use_container_width=True)
        if st.checkbox("선택 조건으로 전체 센서 RMS·임계선·경보 시점 표시"):
            for test_id, spec in EXPERIMENTS.items():
                for bearing, channels in spec["channels"].items():
                    for sensor in range(1, len(channels)+1):
                        try:
                            frame = prepare_alarm_frame(test_id, bearing, sensor)
                            thresholds = compute_alarm_thresholds(frame, multiplier)
                            st.plotly_chart(rms_figure(frame, thresholds, consecutive,
                                f"{test_id} B{bearing} S{sensor} · MAD {multiplier} · 연속 {consecutive}"),
                                use_container_width=True)
                        except (OSError, ValueError, KeyError) as exc:
                            st.warning(str(exc))


def replay_rms_figure(part, thresholds):
    """자동 인덱스/Arrow 변환 없이 실제 시각과 RMS를 Plotly에 명시한다."""
    times = [pd.Timestamp(value).isoformat() for value in part["measured_at"]]
    values = [float(value) if pd.notna(value) and np.isfinite(value) and value > 0 else None
              for value in part["rms"]]
    fig = go.Figure(go.Scatter(
        x=times, y=values, name="실측 RMS", mode="lines+markers", connectgaps=False,
        line=dict(color="#2563eb", width=2), marker=dict(size=5),
        hovertemplate="%{x}<br>RMS=%{y:.5f}<extra></extra>",
    ))
    for level, color in (("Warning", "orange"), ("Alarm", "red"), ("Critical", "purple")):
        value = thresholds[level.lower()]
        fig.add_hline(y=value, line_dash="dash", line_color=color,
                      annotation_text=f"{level}: {value:.5f}")
    fig.update_layout(
        height=300, xaxis=dict(title="실제 측정 시각", type="date"), yaxis_title="RMS",
        margin=dict(l=15, r=25, t=30, b=20), hovermode="x unified",
        showlegend=False,
    )
    return fig


def verified_demo_cases():
    """발표 사례는 현재 RMS와 알고리즘까지 검증된 번들만 허용한다."""
    labels = {('1st_test', 3): '내륜 결함', ('1st_test', 4): '전동체 결함',
              ('2nd_test', 1): '외륜 결함', ('3rd_test', 3): '외륜 결함'}
    cases = {}
    try:
        configs = supported_cloud_configs()
    except (OSError, ValueError, KeyError, TypeError):
        return []
    for config in sorted(configs, key=lambda c: c['sensor']):
        experiment, bearing, sensor = (config[k] for k in ('experiment', 'bearing', 'sensor'))
        key = (experiment, bearing)
        if (key not in labels or key in cases or config['multiplier'] != 5
                or config['consecutive'] != 3 or config['trigger_level'] != 'Alarm'):
            continue
        try:
            frame = prepare_alarm_frame(experiment, bearing, sensor)
            thresholds = compute_alarm_thresholds(frame, 5)
            event = sustained_alarm_states(frame, thresholds, 3)[1]['Alarm']
            load_cloud_waterfall(experiment, bearing, sensor, frame, event, thresholds, 3, 'Alarm',
                                 build_alarm_event_waterfall, compute_fft_spectrum)
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
        cases[key] = dict(experiment=experiment, bearing=bearing, sensor=sensor,
                          label=f'{experiment} / Bearing {bearing} — {labels[key]}')
    return [cases[key] for key in labels if key in cases]


def page_chart():
    """실제 측정 RMS 경보 시뮬레이션 → 트리거 시 실제 FFT 워터폴."""
    st.header("실측 진동 데이터 경보 시뮬레이션")
    selection_area = st.container()
    replay_area = st.container()
    details = st.expander("상세 분석 정보", expanded=False)
    with details:
        analysis_mode = st.checkbox("기존 분석 모드 사용", key="ims_analysis_mode")
        if analysis_mode:
            c1, c2, c3 = st.columns(3)
            with c1:
                experiments = list(EXPERIMENTS)
                experiment = st.selectbox("실험(Test)", experiments, index=experiments.index("3rd_test"))
            with c2:
                choices = sorted(EXPERIMENTS[experiment]["channels"])
                bearing = st.selectbox("베어링", choices, index=choices.index(3) if 3 in choices else 0)
            with c3:
                channels = EXPERIMENTS[experiment]["channels"][bearing]
                sensor = st.selectbox("센서 (베어링 내 순서)", list(range(1, len(channels) + 1)))
            trigger_label = st.selectbox("워터폴 표시 임계치", ["경고 (Alarm)", "위험 (Critical)", "주의 (Warning)"])
            sensitivity = st.selectbox("경보 민감도 (MAD 배수)", [5, 7, 10])
            consecutive = st.selectbox("각 단계 연속 초과 횟수", [3, 4, 5])
            playback = st.selectbox("재생 속도", ["빠르게 (30ms)", "보통 (80ms)", "즉시 (대기 없음)"])
        else:
            sensitivity, consecutive = 5, 3
            trigger_label, playback = "경고 (Alarm)", "빠르게 (30ms)"
        st.caption("NASA IMS 실제 RMS 기록을 검사합니다. 결함 명칭은 사례 설명이며 경보 판정에 사용하지 않습니다.")
        st.caption(cloud_support_message())
        st.caption("기준 후보: 초기 10% (올림), 최대 300개. 중앙값 + 단계별 1/2/3 × MAD 배수 × 보정 산포. "
                   "보정 산포=max(1.4826×MAD, 중앙값의 2%, 1e-8). 결측·공백은 연속 횟수를 끊습니다.")
        render_comparison(sensitivity, consecutive)
        if st.button("↺ 시연 결과 초기화", key="clear_ims_alarm"):
            st.session_state.pop("ims_alarm_result", None)
            st.rerun()
    with selection_area:
        if not analysis_mode:
            cases = verified_demo_cases()
            if not cases:
                st.warning("시연 미지원: 현재 RMS·FFT 검증을 통과한 결함 사례 번들이 없습니다.")
                st.session_state.pop("ims_alarm_result", None)
                return
            labels = [case['label'] for case in cases]
            default = next((i for i, case in enumerate(cases) if case['experiment'] == '3rd_test'
                            and case['bearing'] == 3), 0)
            selected = st.selectbox("결함 사례", labels, index=default, key="ims_demo_case")
            case = cases[labels.index(selected)]
            experiment, bearing, sensor = (case[k] for k in ('experiment', 'bearing', 'sensor'))
        submitted = st.button("▶ 시뮬레이션 시작", type="primary", use_container_width=True)

    with replay_area:
        if submitted:
            st.session_state.pop("ims_alarm_result", None)
            try:
                t0 = time.perf_counter()
                frame = prepare_alarm_frame(experiment, bearing, sensor)
                thresholds = compute_alarm_thresholds(frame, sensitivity)
                trigger_level = trigger_label.split(" (")[1].replace(")", "")
                states, events = sustained_alarm_states(frame, thresholds, consecutive)
                trigger_index = events[trigger_level]
                last = trigger_index if trigger_index is not None else len(frame)-1
                # 전체 수천 측정 시점을 모두 화면에 그리지 않고 최초 트리거 직전 실제 기록을 재생한다.
                start = max(0, last - 47)
                replay = frame.iloc[start:last+1]
                st.write(
                    f"실제 기록 {len(frame):,}개를 순서대로 검사했습니다. "
                    f"화면에는 {'첫 경보 전후' if trigger_index is not None else '마지막'} {len(replay)}개를 빠르게 재생합니다."
                )
                a, b, c = st.columns(3)
                a.metric("정상 기준 RMS 중앙값", f"{thresholds['normal_center']:.5f}")
                b.metric("경고 임계치 (Alarm)", f"{thresholds['alarm']:.5f}")
                c.metric("위험 임계치 (Critical)", f"{thresholds['critical']:.5f}")
                status = st.empty()
                chart_slot = st.empty()
                progress = st.progress(0.0)
                delay = {"빠르게 (30ms)": 0.03, "보통 (80ms)": 0.08,
                         "즉시 (대기 없음)": 0.0}[playback]
                # 과거 경보의 최대 등급을 유지(Latch). 해제는 결과 초기화 버튼으로 수행.
                highest_level = "Normal"
                priority = {"Normal": 0, "Warning": 1, "Alarm": 2, "Critical": 3}
                # 프레임마다 새 Plotly 3D를 생성하지 않고 2D RMS만 4프레임마다 갱신.
                for i in range(len(replay)):
                    row = replay.iloc[i]
                    level = states[start+i]
                    if priority[level] > priority[highest_level]:
                        highest_level = level
                    if i % 4 == 0 or i == len(replay) - 1:
                        status.info(f"측정 {start+i+1:,}/{len(frame):,}  |  "
                                    f"{row['measured_at']:%Y-%m-%d %H:%M}  |  RMS={row['rms']:.5f}  |  경보 상태(유지): {highest_level}")
                        part = replay.iloc[:i+1]
                        chart_slot.plotly_chart(replay_rms_figure(part, thresholds),
                                                use_container_width=True,
                                                key=f"ims_rms_replay_{start+i}")
                        progress.progress((i+1) / len(replay))
                    if delay:
                        time.sleep(delay)
                progress.empty()
                latest = frame.iloc[last]
                result = {
                    "experiment": experiment, "bearing": bearing, "sensor": sensor,
                    "trigger_level": trigger_level, "trigger_index": trigger_index,
                    "rms": float(latest["rms"]), "time": pd.Timestamp(latest["measured_at"]),
                    "level": highest_level,
                    "thresholds": thresholds, "waterfall": None,
                    "consecutive": consecutive, "frame": frame,
                    "selection": (experiment, bearing, sensor, sensitivity, consecutive, trigger_level),
                }
                if trigger_index is not None:
                    with st.spinner("경보가 확정되었습니다. 실제 FFT 워터폴을 준비합니다..."):
                        try:
                            result["waterfall"] = build_alarm_event_waterfall(
                                experiment, bearing, sensor, frame, trigger_index,
                                history_points=16, max_frequency_hz=5000,
                            )
                        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                            try:
                                # 경보 확정 후에만, 정확한 동일 사건의 검증 번들을 조회한다.
                                if any(resolve_source_path(str(path)).is_file() for path in replay.source_path):
                                    raise ValueError("원본이 일부라도 존재하면 원본 계산 오류를 번들로 우회하지 않습니다.")
                                result["waterfall"] = load_cloud_waterfall(
                                    experiment, bearing, sensor, frame, trigger_index, thresholds,
                                    consecutive, trigger_level, build_alarm_event_waterfall, compute_fft_spectrum)
                            except (OSError, ValueError, TypeError, KeyError) as cache_exc:
                                has_raw = any(resolve_source_path(str(path)).is_file() for path in replay.source_path)
                                result["waterfall_error"] = (f"원본 FFT: {exc}. {cache_exc}" if has_raw
                                                             else str(cache_exc))
                st.session_state["ims_alarm_result"] = result
                st.caption(f"데이터 검사+재생+FFT 처리 시간: {time.perf_counter() - t0:.2f}초")
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                st.error(f"실제 데이터 경보 시연 오류: {exc}")

        result = st.session_state.get("ims_alarm_result")
        if result is not None:
            if result["selection"] == (experiment, bearing, sensor, sensitivity, consecutive,
                                        trigger_label.split(" (")[1].replace(")", "")):
                with details:
                    st.write(result["thresholds"])
                render_alarm_result(result)
            else:
                st.info("실험·베어링·센서가 변경되었습니다. 시연 시작 버튼으로 다시 실행하세요.")
        elif not submitted:
            st.info("결함 사례를 선택하고 시뮬레이션 시작을 누르면 실제 RMS를 재생합니다.")
        st.caption(f"경보 기준: MAD × {sensitivity} | 연속 {consecutive}회 초과 | "
                   f"{trigger_label.split(' (')[1].replace(')', '')} | Sensor {sensor}")


def page_logs():
    """Logs 탭: 정비 이력 및 알림 로그"""
    st.header("📋 시스템 로그 및 알림 내역")
    st.caption("아래 로그는 UI 화면 예시이며 실제 설비 운영 로그가 아닙니다.")
    log_data = pd.DataFrame({
        "시간": ["2026-11-05 14:32", "2026-11-05 10:15", "2026-11-04 09:00"],
        "레벨": ["WARNING", "INFO", "INFO"],
        "메시지": ["Bearing 3 진동 수치 주의 구간 진입", "1번 라인 정기 점검 완료", "시스템 재부팅 완료"]
    })
    st.dataframe(log_data, use_container_width=True)


def page_user():
    """User Info 탭: 사용자 및 설정"""
    st.header("👤 사용자 정보 및 환경 설정")
    st.write("**담당자:** 정비팀 홍길동")
    st.write("**알림 설정:** 이메일(활성화됨), SMS(비활성화됨)")
    st.button("로그아웃")


def main():
    st.set_page_config(
        page_title="설비 예지보전 제어반",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Chart"

    inject_custom_css(st.session_state.active_page)

    with st.sidebar:
        st.title("⚙️ 예지보전 제어반")
        st.caption("MONITORING MENU")

        # 라디오가 아니라 실제 클릭 가능한 버튼을 이용한 메뉴.
        menu_items = [
            ("Home", "🏠  Home", "nav_home"),
            ("Chart", "📈  Chart", "nav_chart"),
            ("Logs", "📋  Logs", "nav_logs"),
            ("User Info", "👤  User Info", "nav_user_info"),
        ]
        for page, label, button_key in menu_items:
            st.button(
                label,
                key=button_key,
                type="primary" if st.session_state.active_page == page else "secondary",
                use_container_width=True,
                on_click=set_active_page,
                args=(page,),
            )

        st.divider()
        st.caption("NASA IMS Bearing Dataset 기반\n과거 실측 데이터 재생 시뮬레이터")

    menu = st.session_state.active_page
    if menu == "Home":
        page_home()
    elif menu == "Chart":
        page_chart()
    elif menu == "Logs":
        page_logs()
    elif menu == "User Info":
        page_user()


if __name__ == "__main__":
    main()
