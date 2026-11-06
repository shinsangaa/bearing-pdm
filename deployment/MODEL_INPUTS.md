# API 모델 입력 안내

특징 이름·순서는 실제 저장 파이프라인의 `feature_names_in_`를 읽어 확인했다. SHA-256과 라이브러리 버전은 `model_manifest.json`에 기록했다. API는 아래 순서대로 재배열한 DataFrame을 저장 파이프라인에 전달한다. 요청 객체의 키 순서는 달라도 된다.

| ID | 가용 | 입력 구성 | 저장 파이프라인 |
|---|---|---|---|
| `rf_basic` | 가능 | 첫 시험의 같은 시각·베어링 두 센서 특징 24개 | median imputer → RF |
| `rf_basic_holdout` | 가능 | 같은 베어링 센서 특징 평균 12개 | median imputer → RF |
| `ridge` | 가능 | 첫 번째 센서의 현재 특징 6개 | median imputer → StandardScaler → Ridge(alpha=1000) |
| `rf_engineered` | 불가 | 기존 특징 6개 + 이동 통계 5개, 순서는 복원됨 | 학습 코드는 존재하지만 가중치가 저장되지 않음 |

`rf_basic_holdout`은 RF Basic의 별도 평가 방식이며 특징 추가 모델이 아니다. Ridge API는 세 시험 모두 학습한 최종 가중치를 사용하며 저장된 Test 3 독립 실험 곡선과 같은 가중치가 아니다. 모든 응답 목표는 시험 종료 기준 대리 RUL이고 실제 고장 시작 시각의 정답이 아니다.

예시 요청 JSON은 기존 CSV의 **0부터 세는 측정 순번 100**, Bearing 3에서 만들었다. `rf_basic.json`은 1st Test 두 센서, 나머지는 3rd Test의 실제 특징이다. 합성 신호나 임의 숫자로 대체하지 않았다. 코드가 참조하는 학습 방식은 `src/train_rul_baseline.py`, `src/train_ridge_ims.py`, `src/train_rf_enhanced.py`에 있다.

API는 모든 필요한 이름을 가진 유한한 숫자만 받는다. 누락 특징을 조용히 0으로 채우지 않는다. 학습 당시 imputer·스케일링은 저장 파이프라인 안에 그대로 유지했다.

## 전체 특징 순서

### `rf_basic`

```text
sensor1_rms
sensor1_std
sensor1_peak
sensor1_kurtosis
sensor1_crest_factor
sensor1_fft_centroid_hz
sensor1_fft_peak_1_hz
sensor1_fft_peak_1_amplitude
sensor1_fft_peak_2_hz
sensor1_fft_peak_2_amplitude
sensor1_fft_peak_3_hz
sensor1_fft_peak_3_amplitude
sensor2_rms
sensor2_std
sensor2_peak
sensor2_kurtosis
sensor2_crest_factor
sensor2_fft_centroid_hz
sensor2_fft_peak_1_hz
sensor2_fft_peak_1_amplitude
sensor2_fft_peak_2_hz
sensor2_fft_peak_2_amplitude
sensor2_fft_peak_3_hz
sensor2_fft_peak_3_amplitude
```

### `rf_basic_holdout`

```text
rms
std
peak
kurtosis
crest_factor
fft_centroid_hz
fft_peak_1_hz
fft_peak_1_amplitude
fft_peak_2_hz
fft_peak_2_amplitude
fft_peak_3_hz
fft_peak_3_amplitude
```

### `ridge`

```text
rms
std
peak
kurtosis
crest_factor
fft_centroid_hz
```

### `rf_engineered`

```text
rms
std
peak
kurtosis
crest_factor
fft_centroid_hz
rms_ma6
rms_ma36
rms_std36
rms_diff6
kurtosis_ma6
```
