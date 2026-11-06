# RMS 재생 화면 수정

현재 진입점은 `app.py`이다. 실제 사용하는 앱과 테스트 진입점을 기준으로 재생 UI를 검증했다. 이 문서의 파일 경로는 보고서 평면 구조에 맞춰 갱신했다.

기존 재생 차트는 `replay.iloc[:i+1].set_index("measured_at")["rms"]`를 `st.line_chart`에 전달했다. 진단한 차트 명세에는 실제 데이터가 들어 있었고, 날짜는 Arrow `timestamp[us]`를 통한 Vega-Lite 자동 변환 경로를 사용했다. RMS 데이터 누락은 확인되지 않았다. 첫 프레임은 단일 관측인데 기본 선 그래프여서 선이 보이지 않는 문제도 있었다. 브라우저 렌더러 내부의 빈 선 현상까지 직접 재현·확정하지는 못했다.

`app.py`의 `replay_rms_figure()`에서 X는 실측 시각의 ISO 문자열, Y는 실측 RMS 숫자 배열로 명시하고 날짜 축을 지정했다. 아래 정상 출력 그래프와 같은 Plotly를 사용해 기존 인덱스·Arrow 자동 변환 경로를 제거했다. 선과 마커를 함께 사용하므로 한 점도 표시된다. 결측은 빈 값으로 남기고 선으로 연결하지 않는다. Warning/Alarm/Critical 임계선은 기존 계산 결과를 그대로 사용한다. 기존의 최대 48개 재생 및 4프레임마다 갱신 방식은 유지한다. 기준 후보 검사 JSON은 기본적으로 접힌 상세 expander 안에 있다.

`tests/test_ims_alarm.py`는 현재 앱 파일을 실행하도록 경로를 정정하고 다음을 확인한다.

- 실제 Test 2 B1에서 재생 차트의 모든 시각·RMS가 원래 CSV의 재생 구간과 일치
- 재생 임계선 3개가 기존 계산 결과와 일치, 첫 단일 관측도 마커로 전달
- 경보 발생 후 재생 RMS·전체 RMS·실제 TXT FFT Surface가 함께 출력
- 상세 JSON이 접힌 expander 안에만 위치
- 재실행 시 FFT 결과 유지 및 TXT 재독 방지
- 시작 전·단일 피크 조건에서 워터폴 미생성, 기존 임계치·연속 초과 판정 검증

경보 전용 자동 테스트 **11개 통과**, Python 컴파일 검사 통과. 현재 전체 검증 로그는 `project_structure_tests.log`다. 브라우저를 직접 열어 수행하는 시각 검증은 하지 않았다. `src/ims_alarm.py`의 경보 계산·연속 조건, 원본 로딩과 FFT 함수는 수정하지 않았다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ims_alarm.py -v
```
