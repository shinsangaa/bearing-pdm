"""원본이 없는 클라우드에서 검증된 정확한 경보 사건의 실제 FFT만 표시한다."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data/demo_fft"


def algorithm_hash(builder, fft_function):
    parts = [ast.dump(ast.parse(inspect.getsource(fn)), include_attributes=False)
             for fn in (builder, fft_function)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def frame_hash(frame):
    # 날짜 표시 정밀도의 플랫폼 차이 없이 원래 시각·RMS·원본 매핑을 고정한다.
    rows = [[pd.Timestamp(row.measured_at).isoformat(), float(row.rms), str(row.source_path)]
            for row in frame.itertuples()]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def supported_cloud_configs():
    """Read supported settings from installed bundles; loading still requires validation."""
    return [json.loads(path.read_text(encoding="utf-8"))["config"]
            for path in sorted(BUNDLE_DIR.glob("*.json")) if path.with_suffix(".npz").is_file()]


def cloud_support_message():
    entries = [f"{c['experiment']} B{c['bearing']} S{c['sensor']} · "
               f"MAD {c['multiplier']:g} · 연속 {c['consecutive']}회 초과 · {c['trigger_level']}"
               for c in supported_cloud_configs()]
    return "클라우드 FFT 지원 조건: " + (" / ".join(entries) if entries else "설치된 번들 없음")


def load_cloud_waterfall(experiment, bearing, sensor, frame, event_index, thresholds,
                         consecutive, trigger_level, builder, fft_function):
    if event_index is None:
        raise ValueError("확정 경보 전에는 사전 계산 FFT를 표시하지 않습니다.")
    config = dict(experiment=experiment, bearing=bearing, sensor=sensor, event_index=int(event_index),
                  multiplier=thresholds["multiplier"], consecutive=consecutive, trigger_level=trigger_level)
    for path in sorted(BUNDLE_DIR.glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["config"] != config:
            continue
        if metadata["frame_sha256"] != frame_hash(frame):
            raise ValueError("사전 계산 FFT와 현재 CSV의 시각·RMS·원본 매핑이 다릅니다.")
        if metadata["algorithm_sha256"] != algorithm_hash(builder, fft_function):
            raise ValueError("사전 계산 이후 FFT 알고리즘이 변경되어 표시를 보류합니다.")
        bundle = path.with_suffix(".npz")
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != metadata["bundle_sha256"]:
            raise ValueError("실제 FFT 번들 해시가 일치하지 않습니다.")
        with np.load(bundle, allow_pickle=False) as arrays:
            frequency, hours, spectra = (arrays[name] for name in ("frequency", "hours", "spectra"))
        if (frequency.ndim != 1 or len(metadata["times"]) < 2
                or spectra.shape != (len(metadata["times"]), len(frequency))) or not all(
                np.isfinite(x).all() for x in (frequency, hours, spectra)):
            raise ValueError("FFT 번들의 형태나 숫자값이 비정상입니다.")
        times = pd.DatetimeIndex(metadata["times"])
        positions = np.unique(np.linspace(max(0, event_index - 47), event_index,
                                         min(16, event_index + 1), dtype=int))
        expected_times = pd.DatetimeIndex(frame.iloc[positions].measured_at)
        expected_hours = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
        if (len(times) < 2 or times.hasnans or not times.equals(expected_times)
                or hours.shape != (len(times),) or not np.array_equal(hours, expected_hours)
                or frequency.ndim != 1 or len(frequency) < 2
                or not np.all(np.diff(frequency) > 0) or frequency[0] <= 0
                or frequency[-1] > metadata["max_frequency_hz"] or np.any(spectra < 0)):
            raise ValueError("FFT 번들의 시간·주파수 배열이 경보 사건과 일치하지 않습니다.")
        return dict(frequency=frequency, hours=hours, spectra=spectra,
                    times=[pd.Timestamp(t) for t in metadata["times"]], missing=[],
                    channel=metadata["channel"], source="precomputed_actual_txt",
                    provenance=metadata["sources"])
    raise ValueError("선택한 경보 사건에 대응하는 검증된 FFT 번들이 없습니다. " + cloud_support_message())
