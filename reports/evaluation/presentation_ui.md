# 발표용 경보 UI 검증

검증일: 2026-11-05 (Asia/Seoul). 브랜치: `deploy-latest`.

## 변경

- 기본 Chart는 결함 사례 드롭다운 하나와 `▶ 시뮬레이션 시작` 버튼으로 구성한다. 기본 사례는 `3rd_test / Bearing 3 — 외륜 결함`이다.
- Test 1 B3/B4, Test 2 B1, Test 3 B3 중 현재 RMS와 알고리즘 및 번들 해시 검증을 통과한 사례만 표시한다. 같은 사례의 센서는 검증된 센서 중 번호가 작은 것을 선택한다. 현재 네 사례 모두 Sensor 1이다.
- 발표 설정은 MAD 5, 연속 3회 초과, Alarm, 빠르게(30ms)로 고정한다. 실제 사용한 설정과 센서는 하단에 표시한다.
- 기존 실제 RMS 검사, 경보 확정 직전 48개 측정 재생, 로컬 TXT FFT, 원본 없는 환경의 검증 번들 표시를 유지한다. 경보 시각과 RMS를 하드코딩하지 않으며 결함 명칭은 판정에 사용하지 않는다.
- 전체 RMS 그래프를 유지하고 결과에 Alarm 임계값을 함께 표시한다. 긴 임계값 JSON은 하단 `상세 분석 정보`에 넣었다. 기존 개별 설정과 전체 센서 비교 기능은 해당 expander의 `기존 분석 모드 사용`에서 보존한다.

## 실행한 검증

프로젝트 `.venv`의 Python으로 아래 테스트를 실행해 총 41개가 통과했다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_presentation_ui.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cloud_waterfall.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ims_alarm.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_project.py
```

- 발표 UI 3개: 드롭다운 단일화, 기본값, 네 사례의 자동 센서와 고정 설정, 실제 CSV에서 계산한 최초 Alarm·RMS, 경보까지 재생하는 그래프·임계선, 로컬 원본 FFT 및 클라우드 모의 환경의 3D surface를 확인했다.
- 손상된 Sensor 1 번들을 제외하고 검증된 Sensor 2를 자동 선택하는 것을 확인했다. 다른 사례의 번들이 없으면 드롭다운에서 제외하며, 전체 번들이 없으면 시작 버튼 대신 시연 미지원 안내를 표시한다.
- Cloud waterfall 3개: 6개 원본 센서 조합의 FFT 재계산 일치, 모의 클라우드 화면, 해시·시간·주파수 변조 차단을 확인했다.
- 기존 경보 11개: 연속 초과, 결측·공백, 재생, 기존 분석 설정과 전체 센서 그래프 등을 확인했다.
- 프로젝트 회귀 24개: 원본 없는 환경의 메뉴 이동, Home의 저장된 Ridge 그래프와 RF 평가 화면, 저장 모델·예측 결과 및 데이터 관련 기존 검증을 통과했다.
- 기존 FFT·RMS 준비·재생 함수 및 Home·Logs·User Info·메뉴 함수의 AST가 HEAD와 동일함을 확인했다. `src/ims_alarm.py`, `src/cloud_waterfall.py`, FastAPI, 모델, 전처리 데이터, 기존 번들 및 배포 설정에 변경이 없다.
- 번들 12개 JSON/NPZ 모두 Git에 이미 추적되며 `.gitignore` 제외 대상이 아니다. 원본 TXT는 Git에 추적되지 않는다. `git diff --check`를 통과했다.

실제 Streamlit Cloud 재배포와 외부 FastAPI 서버 호출은 수행하지 않았다. 클라우드 검증은 원본 TXT 접근을 차단한 Streamlit AppTest 환경에서 실행했다. 브랜치 전환, 자동 커밋 및 Push는 수행하지 않았다.

## 커밋할 파일

- `app.py`
- `tests/test_presentation_ui.py`
- `tests/test_cloud_waterfall.py`
- `tests/test_ims_alarm.py`
- `tests/test_project.py`
- `reports/evaluation/presentation_ui.md`
