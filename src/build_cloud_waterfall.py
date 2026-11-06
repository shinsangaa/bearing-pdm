"""실제 NASA IMS TXT 파일에서 Test 1 번들을 생성합니다: python -m src.build_cloud_waterfall.

기존 번들을 덮어쓰거나 확인된 알람 없이 이벤트를 절대 생성하지 마세요.
"""
import hashlib
import json

import numpy as np

from src.cloud_waterfall import BUNDLE_DIR, algorithm_hash, frame_hash, load_cloud_waterfall
from src.preprocessing import read_measurement, resolve_source_path


def build_bundle(experiment, bearing, sensor):
    from app import (prepare_alarm_frame, compute_alarm_thresholds, sustained_alarm_states,
                     build_alarm_event_waterfall, compute_fft_spectrum)

    frame = prepare_alarm_frame(experiment, bearing, sensor)
    thresholds = compute_alarm_thresholds(frame, 5)
    event = sustained_alarm_states(frame, thresholds, 3)[1]["Alarm"]
    if event is None:
        print(f"{experiment} B{bearing} S{sensor}: no Alarm, skipped")
        return
    path = BUNDLE_DIR / f"{experiment}_b{bearing}_s{sensor}_mad5_n3_alarm.npz"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Will not overwrite existing bundle: {path}")
    waterfall = build_alarm_event_waterfall(experiment, bearing, sensor, frame, event)
    if waterfall["missing"] or waterfall["times"][-1] != frame.iloc[event].measured_at:
        raise ValueError("Missing actual TXT in alarm history; bundle rejected")
    sources = []
    for time in waterfall["times"]:
        row = frame.loc[frame.measured_at.eq(time)].iloc[0]
        raw = resolve_source_path(row.source_path)
        signal = read_measurement(raw, experiment)[:, waterfall["channel"] - 1]
        sources.append(dict(path=str(row.source_path), sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                            csv_rms=float(row.rms), txt_rms=float(np.sqrt(np.mean(signal ** 2))),
                            bytes=raw.stat().st_size))
    metadata = dict(
        config=dict(experiment=experiment, bearing=bearing, sensor=sensor, event_index=int(event),
                    multiplier=thresholds["multiplier"], consecutive=3, trigger_level="Alarm"),
        frame_sha256=frame_hash(frame),
        algorithm_sha256=algorithm_hash(build_alarm_event_waterfall, compute_fft_spectrum),
        times=[t.isoformat() for t in waterfall["times"]], channel=waterfall["channel"],
        sources=sources, history_points=16, max_frequency_hz=5000,
        note="Precomputed from actual NASA TXT, not synthetic. Exact event only.")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{k: waterfall[k] for k in ("frequency", "hours", "spectra")})
    metadata["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    verified = load_cloud_waterfall(experiment, bearing, sensor, frame, event, thresholds, 3, "Alarm",
                                    build_alarm_event_waterfall, compute_fft_spectrum)
    np.testing.assert_array_equal(verified["spectra"], waterfall["spectra"])
    print(f"{path.name}: event={event}, at={frame.iloc[event].measured_at}, TXT={len(sources)}")


if __name__ == "__main__":
    for bearing in (3, 4):
        for sensor in (1, 2):
            build_bundle("1st_test", bearing, sensor)
