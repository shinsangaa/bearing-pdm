"""CSV RMS 경보 계산. 원본 TXT, 종료 결함 라벨, 날짜별 규칙을 사용하지 않는다."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

LEVELS = {"Warning": "warning", "Alarm": "alarm", "Critical": "critical"}


def validate_alarm_frame(frame):
    """잘못된 시각은 거부하고, RMS 결측/0/음수/무한대는 행을 보존해 표시한다."""
    frame = frame.copy()
    frame["measured_at"] = pd.to_datetime(frame["measured_at"], errors="coerce")
    if frame.empty or frame["measured_at"].isna().any():
        raise ValueError("측정 기록이 없거나 해석 불가능한 측정 시각이 있습니다.")
    if frame["measured_at"].duplicated().any():
        raise ValueError("동일 센서의 측정 시각이 중복됩니다.")
    frame["rms"] = pd.to_numeric(frame["rms"], errors="coerce")
    frame = frame.sort_values("measured_at").reset_index(drop=True)
    frame["rms_valid"] = np.isfinite(frame["rms"]) & frame["rms"].gt(0)
    return frame


def compute_alarm_thresholds(frame, multiplier=5.0):
    """초기 ceil(10%), 최대 300개. 유효 RMS만 추정에 사용하며 창을 확장하지 않는다.

    산포는 max(1.4826*MAD, 중앙값*2%, 1e-8). 배수는 이 보정 산포에 적용한다.
    6 산포 초과값은 검토 대상으로 세되 삭제하지 않는다. 상승 진단은 라벨이 아니다.
    """
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("MAD 배수는 양의 유한값이어야 합니다.")
    frame = validate_alarm_frame(frame)
    n = min(300, math.ceil(len(frame) * 0.1))
    base = frame.iloc[:n]
    valid = base.loc[base.rms_valid, "rms"].to_numpy(dtype=float)
    if len(valid) < 10 or len(valid) < math.ceil(n * 0.8):
        raise ValueError(f"기준 후보 {n}개 중 유효값 {len(valid)}개: 최소 10개 및 80% 필요")
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center)))
    scale = max(1.4826 * mad, center * 0.02, 1e-8)
    # 시간순 첫/끝 1/3의 중앙값 비교. 초기부터 결함인지는 RMS만으로 확정할 수 없다.
    thirds = np.array_split(valid, 3)
    early, late = float(np.median(thirds[0])), float(np.median(thirds[-1]))
    early_mad = float(np.median(np.abs(thirds[0] - early)))
    early_scale = max(1.4826 * early_mad, early * 0.02, 1e-8)
    drift = (late - early) / early_scale
    intervals = frame.measured_at.diff().dt.total_seconds().iloc[1:n]
    gap_limit = float(intervals.median() * 3)
    return dict(normal_center=center, warning=center + multiplier * scale,
                alarm=center + 2 * multiplier * scale, critical=center + 3 * multiplier * scale,
                multiplier=float(multiplier), mad=mad, robust_scale=scale,
                baseline_n=n, baseline_valid_n=len(valid), baseline_invalid_n=n-len(valid),
                baseline_first_at=base.iloc[0].measured_at,
                baseline_last_at=base.iloc[-1].measured_at,
                invalid_total=int((~frame.rms_valid).sum()),
                baseline_outlier_n=int((np.abs(valid-center) > 6 * scale).sum()),
                baseline_early_median=early, baseline_late_median=late,
                baseline_change_ratio=late / early - 1, baseline_drift_scale=drift,
                baseline_review="상승 추세 검토 필요" if drift > 3 else "뚜렷한 상승 미검출 (정상 보증 아님)",
                gap_limit_seconds=gap_limit)


def alarm_level(rms, thresholds):
    if not np.isfinite(rms) or rms <= 0:
        return "Invalid"
    for level, key in reversed(LEVELS.items()):
        if rms > thresholds[key]:
            return level
    return "Normal"


def sustained_alarm_states(frame, thresholds, consecutive=3):
    """단계별 연속 초과를 독립 계산 후 최대 확정 단계를 유지한다.

    결측/비정상 값과 초기 측정 간격 중앙값의 3배 초과 공백은 연속 횟수를 끊는다.
    확정 시각은 N번째 초과 기록이며 첫 초과 시각으로 소급하지 않는다.
    1회는 비교 분석 전용이다. UI 재생에는 3~5회를 제공한다.
    """
    if consecutive not in (1, 3, 4, 5):
        raise ValueError("지속 조건은 1, 3, 4, 5회 중 선택하세요.")
    frame = validate_alarm_frame(frame)
    counts = dict.fromkeys(LEVELS, 0)
    first = dict.fromkeys(LEVELS, None)
    states = ["Normal"] * len(frame)
    highest = "Normal"
    gaps = frame.measured_at.diff().dt.total_seconds().to_numpy()
    values = frame.rms.to_numpy(dtype=float)
    valid = frame.rms_valid.to_numpy()
    for i in range(thresholds["baseline_n"], len(frame)):
        if gaps[i] > thresholds["gap_limit_seconds"]:
            counts = dict.fromkeys(LEVELS, 0)
        value = values[i]
        for level, key in LEVELS.items():
            counts[level] = counts[level] + 1 if valid[i] and value > thresholds[key] else 0
            if counts[level] >= consecutive and first[level] is None:
                first[level] = i
        highest = next((level for level in reversed(LEVELS) if first[level] is not None), highest)
        states[i] = highest
    return states, first


def find_first_alarm(frame, thresholds, trigger_level, consecutive=3):
    return sustained_alarm_states(frame, thresholds, consecutive)[1][trigger_level]


def rms_figure(frame, thresholds, consecutive=3, title="Actual RMS"):
    frame = validate_alarm_frame(frame)
    _, events = sustained_alarm_states(frame, thresholds, consecutive)
    fig = go.Figure(go.Scatter(x=frame.measured_at, y=frame.rms.where(frame.rms_valid),
                               name="CSV RMS", mode="lines", connectgaps=False))
    fig.add_vrect(x0=frame.iloc[0].measured_at, x1=frame.iloc[thresholds['baseline_n']-1].measured_at,
                  fillcolor="gray", opacity=0.12, line_width=0)
    for (level, key), color in zip(LEVELS.items(), ("orange", "red", "purple")):
        fig.add_hline(y=thresholds[key], line_dash="dash", line_color=color,
                      annotation_text=f"{level}: {thresholds[key]:.5f}")
        idx = events[level]
        if idx is not None:
            row = frame.iloc[idx]
            fig.add_trace(go.Scatter(x=[row.measured_at], y=[row.rms], mode="markers",
                                    marker=dict(size=10, color=color), name=f"{level} confirmed ({consecutive})"))
    fig.update_layout(title=title, xaxis_title="측정 시각", yaxis_title="RMS", height=390,
                      margin=dict(l=20, r=20, t=60, b=20), hovermode="x unified")
    return fig


def compare_series(frame, experiment, bearing, sensor, multipliers=(5, 7, 10), durations=(1, 3, 5)):
    """결함 라벨과 원본 TXT 없이 모든 조합·경보 단계의 최초 확정 결과를 반환한다."""
    frame = validate_alarm_frame(frame)
    rows = []
    for multiplier in multipliers:
        thresholds = compute_alarm_thresholds(frame, multiplier)
        for duration in durations:
            _, events = sustained_alarm_states(frame, thresholds, duration)
            for level, key in LEVELS.items():
                idx = events[level]
                event = frame.iloc[idx] if idx is not None else None
                rows.append(dict(test_id=experiment, bearing_id=bearing, sensor=sensor,
                                 level=level, mad_multiplier=multiplier, consecutive=duration,
                                 alarm_occurred=idx is not None,
                                 first_alarm_at=event.measured_at if event is not None else pd.NaT,
                                 alarm_rms=float(event.rms) if event is not None else np.nan,
                                 threshold=thresholds[key],
                                 remaining_calendar_hours=(frame.iloc[-1].measured_at-event.measured_at).total_seconds()/3600
                                 if event is not None else np.nan,
                                 **{k: v for k, v in thresholds.items() if k.startswith("baseline_")},
                                 invalid_total=thresholds["invalid_total"]))
    return pd.DataFrame(rows)
