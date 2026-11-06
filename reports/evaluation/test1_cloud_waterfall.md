# Test 1 Streamlit Cloud FFT 번들 검증

검증일: 2026-11-05 (Asia/Seoul)

## 지원 조건과 최초 확정 Alarm

모든 조건은 MAD 5, 연속 3회 **초과**, Alarm이다. 시각은 NASA IMS 기록의 시각이며 시간대를 임의로 변환하지 않았다. 경보 시각은 고장 확정 시각이 아니다.

| Test | Bearing | Sensor | 최초 확정 인덱스 (0부터) | 최초 확정 Alarm 시각 | 상태 |
|---|---:|---:|---:|---|---|
| 1st_test | 3 | 1 | 220 | 2003-10-30 01:19:46 | 신규 |
| 1st_test | 3 | 2 | 225 | 2003-10-30 02:09:46 | 신규 |
| 1st_test | 4 | 1 | 1440 | 2003-11-19 11:16:07 | 신규 |
| 1st_test | 4 | 2 | 237 | 2003-10-31 00:29:46 | 신규 |
| 2nd_test | 1 | 1 | 609 | 2004-02-16 16:02:39 | 기존 유지 |
| 3rd_test | 3 | 1 | 6150 | 2004-04-16 21:52:55 | 기존 유지 |

## 실제 원본 사용

- `prepare_alarm_frame()` → `compute_alarm_thresholds(frame, 5)` → `sustained_alarm_states(frame, thresholds, 3)`으로 최초 Alarm을 계산했다.
- 기존 `build_alarm_event_waterfall()`과 `compute_fft_spectrum()`을 수정 없이 재사용했다. 경보 직전 48개 측정 범위에서 기존 함수가 선택하는 원본 TXT 16개로 각각 16 × 160 FFT 배열을 생성했다. 합성 신호는 사용하지 않았다.
- Test 1의 실제 폴더는 `data/raw/1st_test/1st_test/`이다. CSV의 `source_path` 및 `resolve_source_path()`가 이 실제 폴더에 대응한다. 중복 문자열을 제거하거나 CSV 경로를 수정하지 않았다.
- 각 JSON은 사용한 TXT의 경로, SHA-256, 파일 크기, CSV RMS, TXT에서 다시 계산한 RMS, 채널, 측정 시각을 기록한다. 전체 선택 센서 RMS 프레임 해시, FFT 알고리즘 해시 및 NPZ SHA-256도 기록한다.

## 변경 내용

- `data/demo_fft/1st_test_b{3,4}_s{1,2}_mad5_n3_alarm.{npz,json}`: 4개 사건, 총 8개 파일 추가.
- `src/build_cloud_waterfall.py`: 재현 가능한 로컬 생성 도구. `python -m src.build_cloud_waterfall`로 실행하며 기존 번들을 덮어쓰지 않는다. 경보가 없으면 생성하지 않고, 원본 누락 또는 RMS 불일치 시 생성을 거부한다.
- `src/cloud_waterfall.py`: 기존 설정·RMS·알고리즘·NPZ 해시 검증 유지. 사건의 실제 측정 시각, 상대 시간, 주파수 순서·범위 및 배열 검증을 추가했다. 지원 설정은 설치된 JSON/NPZ 메타데이터에서 읽는다.
- `app.py`: 메타데이터에서 지원 조건을 안내한다. 원본 없는 환경에서 미지원 사건은 긴 원본 파일 오류 대신 지원 조건을 표시한다. 실제 원본이 있는데 계산 오류가 난 경우 이를 번들로 우회하지 않는 동작은 유지한다. 실제 NASA TXT에서 계산한 번들이라는 화면 안내도 유지한다.

## 검증 결과

프로젝트 `.venv`의 Python으로 실행했다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cloud_waterfall.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ims_alarm.py
git diff --check
```

- Cloud waterfall 테스트 3개 통과: 6개 지원 조합의 실제 TXT 재계산과 NPZ의 주파수·시간·스펙트럼 배열이 정확히 일치한다. 원본 TXT 출처 해시와 RMS도 확인했다.
- 원본 TXT가 없는 모의 클라우드에서 6개 조합 모두 검증 번들 및 Plotly `surface` 표시를 확인했다. 미지원 연속 횟수에서는 지원 조건 안내를 확인했다.
- 변경된 RMS, 경보 사건 인덱스, MAD 배수, 연속 횟수, 경보 단계 및 경보 없음은 거부된다. FFT 알고리즘 해시·NPZ 해시·측정 시각·상대 시간·주파수 변조도 거부된다.
- 기존 경보 회귀 테스트 11개 통과. 기존 FFT 두 함수와 RMS 프레임 준비·재생 함수의 AST가 HEAD와 동일함을 확인했다.
- 기존 Test 2·3 NPZ는 HEAD와 바이트 단위로 동일하고 JSON 내용도 동일하다.
- 모든 번들 JSON/NPZ는 `git check-ignore --no-index`에서 제외 대상이 아님을 확인했다. `.gitignore` 변경은 필요 없었다. 원본 TXT는 계속 제외되며 `git ls-files data/raw`는 비어 있다.
- 다른 머신러닝 모델, FastAPI, Docker, Render 설정은 변경하지 않았다.
- 실제 Streamlit Cloud 재배포는 수행하지 않았다. 아래 파일을 커밋·푸시하면 Cloud에서 번들을 사용할 수 있다.

## 커밋할 파일

- `app.py`
- `src/cloud_waterfall.py`
- `src/build_cloud_waterfall.py`
- `tests/test_cloud_waterfall.py`
- `reports/evaluation/test1_cloud_waterfall.md`
- `data/demo_fft/1st_test_b3_s1_mad5_n3_alarm.json`
- `data/demo_fft/1st_test_b3_s1_mad5_n3_alarm.npz`
- `data/demo_fft/1st_test_b3_s2_mad5_n3_alarm.json`
- `data/demo_fft/1st_test_b3_s2_mad5_n3_alarm.npz`
- `data/demo_fft/1st_test_b4_s1_mad5_n3_alarm.json`
- `data/demo_fft/1st_test_b4_s1_mad5_n3_alarm.npz`
- `data/demo_fft/1st_test_b4_s2_mad5_n3_alarm.json`
- `data/demo_fft/1st_test_b4_s2_mad5_n3_alarm.npz`
