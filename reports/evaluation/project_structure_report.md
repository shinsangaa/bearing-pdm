# reports 파일 정리 결과

## 정리 전후

작업 시작 시 Git 상태는 깨끗했다. 기존 사용자 변경을 되돌리지 않았고 별도 정리 스크립트를 만들지 않았다. 실제 이동 전에 프로젝트 Python 파일과 보고서의 내용을 읽고 생성 코드·소비 코드를 대조했다. 이동 대상 이름 충돌은 없었다. 동일 내용의 중복 보고서도 발견되지 않아, 비슷한 이름만으로 중요한 결과를 삭제하지 않았다.

```text
정리 전                              정리 후
reports/                            reports/
├── evaluation/                     ├── evaluation/   # 평가·출처·검증
│   └── experiment_split/           ├── figures/      # PNG·실측 경보 HTML
├── figures/                        └── predictions/  # 개별 예측 CSV
│   ├── 1st_test/
│   ├── 2nd_test/
│   ├── 3rd_test/
│   ├── bilstm_rul/
│   ├── rul_baseline/
│   └── experiment_split/
├── predictions/
│   └── experiment_split/
├── ims_alarm/
├── ridge_ims/
└── rf_feature_comparison.csv
```

최종 세 디렉터리 안에는 하위 디렉터리가 없다. 그림은 시험명과 베어링·센서를 파일명에 넣어 충돌을 방지했다. 필요한 신호처리·NASA 출처·경보 분석 자료는 모델 평가와 별개로 대시보드 재현에 필요하여 유지했다.

## 모델 분류와 보존

RF Basic의 첫 시험 내부 시간순 평가와 실험별 분리 평가는 서로 다르므로 둘 다 보존했다. 후자의 기존 `rf_test.csv`는 이동 통계가 없는 센서 평균 특징 12개로 학습한 RF 예측이다. 이름만 보고 RF Engineered로 분류하지 않았다. 가중치도 각각 `models/rf_basic.joblib`, `models/rf_basic_holdout.joblib`로 구분했다.

RF Engineered는 기존 특징 6개와 추가 이동 통계를 포함한 특징 11개를 비교한 결과다. `rf_engineered_feature_comparison.csv`의 기존 성능 수치를 보존했다. **이 모델의 개별 예측과 가중치는 기존 코드가 저장하지 않아 없었다.** 이번 파일 정리를 위해 재학습하거나 결과를 추정하지 않았다.

Ridge는 alpha 선택 표·실험 제외 평가·Test 3 예측을 분리했다. 가중치 `models/ridge_all_tests.joblib`은 세 실험을 모두 학습한 최종 모델이므로, 1st+2nd만 학습한 과거 Test 3 예측을 재현하는 가중치로 표시하지 않았다.

이동한 CSV·PNG·HTML·모델 가중치는 바이트 단위로 보존했다. RF 평가 JSON 2개는 모델 경로만 갱신했으며 수치와 모델 해시가 원래 Git 버전과 동일함을 확인했다. 실측 경보 문서 2개는 경로와 최신 검증 안내를 갱신했다. 딥러닝 혼합 비교표의 RF 지표 3행은 별도의 `rf_basic_protocol_comparison.csv`로 남겼다.

## 코드와 설정 변경

- `app.py`: Ridge 예측 경로 변경. Home에서 RF Basic의 저장 평가·예측 그림 및 RF Engineered 비교표를 열 수 있게 했다. 메뉴와 Chart의 재생·경보·워터폴 함수는 유지했다.
- `src/demo_support.py`: 딥러닝 추론 함수 제거. RF 가중치·결과 경로, 평면 신호 그림 저장 위치, 세 모델 요약·현재 자산 해시 생성 수정.
- `src/train_rul_baseline.py`: RF 두 평가 방식의 저장 경로를 새 접두사로 변경. 내부·실험별 분리 평가의 알고리즘은 유지하고 저장 파일 충돌을 검사한다.
- `src/train_rf_enhanced.py`: 새 비교 CSV 경로 적용. import만으로 학습하지 않도록 `main()`으로 감싸고 기존 결과 덮어쓰기를 막았다. 학습 특징과 파라미터는 유지했다.
- `src/train_ridge_ims.py`: 평가와 예측 저장 경로 분리 및 최종 가중치 이름 변경. 기존 결과가 있으면 덮어쓰지 않는다.
- `src/analyze_ims_alarm.py`: 경보 계산은 그대로 두고 저장 목적지만 `evaluation/ims_alarm_*`, `figures/ims_alarm_*`로 변경.
- `app_copy5.py`, `app_copy6.py`, `app_copy7.py`: 보존한 과거 UI 사본의 Ridge 경로도 함께 변경.
- `tests/test_project.py`: 제거된 모델 전용 테스트·import 제거. 현재 메뉴, RF 결과, 평면 저장, 두 RF 체크포인트의 예측 재현, 원본 대응 검증을 유지·추가했다.
- `.gitignore`, `requirements.txt`, `README.md`: 현재 세 모델과 파일 구조 반영. 딥러닝 라이브러리 의존성만 제거했으며 가상환경은 변경하지 않았다.

모든 프로젝트 Python·노트북·설정·관련 문서를 재검색해 이전 실행 경로와 딥러닝 import가 남지 않았음을 확인했다. 현재 프로젝트에 노트북은 없었다. `src/ims_alarm.py`, `src/preprocessing.py`, `src/signal_processing.py`는 수정하지 않았다. 앱의 FFT·원본 로딩·워터폴·재생 차트 함수도 AST로 기존 버전과 동일함을 확인했다.

## 검증과 남은 사항

검증 결과는 `project_structure_verification.json`, 전체 테스트 출력은 `project_structure_tests.log`에 기록한다. Streamlit을 숨겨진 프로세스로 실제 기동해 health와 루트 페이지 HTTP 200을 확인한 뒤 종료했다. 브라우저 픽셀을 직접 확인하는 시각 검증은 수행하지 않았다.

원본 9,464개 파일의 목록·크기·수정 시각이 동일하고 모든 전처리 파일 SHA-256이 동일하다. 모델·CSV·그림도 기존 해시와 대조했다. 원본 바이트 전체를 다시 해시한 것은 아니며 실제 데이터 원본 대조 테스트는 첫 시험 모든 행과 처음·마지막 측정의 독립 신호 계산을 확인한다.

`app_copy2.py`~`app_copy7.py`는 내용이 서로 다른 과거 UI 변형이므로 동일 백업으로 간주하지 않고 보존했다. 최종 실행 진입점은 `app.py`다. RF Engineered의 가중치·개별 예측 저장은 추후 별도 학습 작업이 필요하다. 평가 조건이 서로 다른 모델의 수치를 같은 조건의 성능 순위로 해석하지 않는다.

## 이동·이름 변경 전체 목록

개별 파일의 이전·현재 위치와 SHA-256은 `project_structure_migration.json`에도 기록했다. 아래 이전 이름은 작업 이력이며 실행 참조가 아니다.

| 이전 파일 | 현재 파일 |
|---|---|
| `reports/rf_feature_comparison.csv` | `reports/evaluation/rf_engineered_feature_comparison.csv` |
| `reports/ridge_ims/validation_alpha_comparison.csv` | `reports/evaluation/ridge_validation_alpha.csv` |
| `reports/ridge_ims/leave_one_test_out.csv` | `reports/evaluation/ridge_leave_one_test_out.csv` |
| `reports/ridge_ims/test3_predictions.csv` | `reports/predictions/ridge_test3_predictions.csv` |
| `reports/evaluation/rul_baseline.json` | `reports/evaluation/rf_basic_metrics.json` |
| `reports/evaluation/rul_baseline_feature_importance.csv` | `reports/evaluation/rf_basic_feature_importance.csv` |
| `reports/evaluation/rul_baseline_split_assignments.csv` | `reports/evaluation/rf_basic_split_assignments.csv` |
| `reports/predictions/rul_baseline.csv` | `reports/predictions/rf_basic_predictions.csv` |
| `reports/evaluation/experiment_split/rf.json` | `reports/evaluation/rf_basic_holdout_metrics.json` |
| `reports/evaluation/experiment_split/data_audit.json` | `reports/evaluation/ims_data_audit.json` |
| `reports/evaluation/experiment_split/split_audit.json` | `reports/evaluation/rf_basic_holdout_split_audit.json` |
| `reports/evaluation/experiment_split/split_assignments.csv` | `reports/evaluation/rf_basic_holdout_split_assignments.csv` |
| `reports/predictions/experiment_split/rf_test.csv` | `reports/predictions/rf_basic_holdout_test_predictions.csv` |
| `reports/predictions/experiment_split/rf_validation.csv` | `reports/predictions/rf_basic_holdout_validation_predictions.csv` |
| `reports/figures/experiment_split/rf_test.png` | `reports/figures/rf_basic_holdout_test.png` |
| `models/rul_baseline.joblib` | `models/rf_basic.joblib` |
| `models/experiment_split/rf.joblib` | `models/rf_basic_holdout.joblib` |
| `models/ridge_ims_all3.joblib` | `models/ridge_all_tests.joblib` |
| `reports/figures/1st_test/bearing1_degradation.png` | `reports/figures/ims_1st_test_bearing1_degradation.png` |
| `reports/figures/1st_test/bearing1_sensor1_signal.png` | `reports/figures/ims_1st_test_bearing1_sensor1_signal.png` |
| `reports/figures/1st_test/bearing1_sensor2_signal.png` | `reports/figures/ims_1st_test_bearing1_sensor2_signal.png` |
| `reports/figures/1st_test/bearing2_degradation.png` | `reports/figures/ims_1st_test_bearing2_degradation.png` |
| `reports/figures/1st_test/bearing2_sensor1_signal.png` | `reports/figures/ims_1st_test_bearing2_sensor1_signal.png` |
| `reports/figures/1st_test/bearing2_sensor2_signal.png` | `reports/figures/ims_1st_test_bearing2_sensor2_signal.png` |
| `reports/figures/1st_test/bearing3_degradation.png` | `reports/figures/ims_1st_test_bearing3_degradation.png` |
| `reports/figures/1st_test/bearing3_sensor1_signal.png` | `reports/figures/ims_1st_test_bearing3_sensor1_signal.png` |
| `reports/figures/1st_test/bearing3_sensor2_signal.png` | `reports/figures/ims_1st_test_bearing3_sensor2_signal.png` |
| `reports/figures/1st_test/bearing4_degradation.png` | `reports/figures/ims_1st_test_bearing4_degradation.png` |
| `reports/figures/1st_test/bearing4_sensor1_signal.png` | `reports/figures/ims_1st_test_bearing4_sensor1_signal.png` |
| `reports/figures/1st_test/bearing4_sensor2_signal.png` | `reports/figures/ims_1st_test_bearing4_sensor2_signal.png` |
| `reports/figures/1st_test/nasa_rul_timeline.png` | `reports/figures/ims_1st_test_nasa_rul_timeline.png` |
| `reports/figures/2nd_test/bearing1_degradation.png` | `reports/figures/ims_2nd_test_bearing1_degradation.png` |
| `reports/figures/2nd_test/bearing1_sensor1_signal.png` | `reports/figures/ims_2nd_test_bearing1_sensor1_signal.png` |
| `reports/figures/2nd_test/bearing2_degradation.png` | `reports/figures/ims_2nd_test_bearing2_degradation.png` |
| `reports/figures/2nd_test/bearing2_sensor1_signal.png` | `reports/figures/ims_2nd_test_bearing2_sensor1_signal.png` |
| `reports/figures/2nd_test/bearing3_degradation.png` | `reports/figures/ims_2nd_test_bearing3_degradation.png` |
| `reports/figures/2nd_test/bearing3_sensor1_signal.png` | `reports/figures/ims_2nd_test_bearing3_sensor1_signal.png` |
| `reports/figures/2nd_test/bearing4_degradation.png` | `reports/figures/ims_2nd_test_bearing4_degradation.png` |
| `reports/figures/2nd_test/bearing4_sensor1_signal.png` | `reports/figures/ims_2nd_test_bearing4_sensor1_signal.png` |
| `reports/figures/3rd_test/bearing1_degradation.png` | `reports/figures/ims_3rd_test_bearing1_degradation.png` |
| `reports/figures/3rd_test/bearing1_sensor1_signal.png` | `reports/figures/ims_3rd_test_bearing1_sensor1_signal.png` |
| `reports/figures/3rd_test/bearing2_degradation.png` | `reports/figures/ims_3rd_test_bearing2_degradation.png` |
| `reports/figures/3rd_test/bearing2_sensor1_signal.png` | `reports/figures/ims_3rd_test_bearing2_sensor1_signal.png` |
| `reports/figures/3rd_test/bearing3_degradation.png` | `reports/figures/ims_3rd_test_bearing3_degradation.png` |
| `reports/figures/3rd_test/bearing3_sensor1_signal.png` | `reports/figures/ims_3rd_test_bearing3_sensor1_signal.png` |
| `reports/figures/3rd_test/bearing4_degradation.png` | `reports/figures/ims_3rd_test_bearing4_degradation.png` |
| `reports/figures/3rd_test/bearing4_sensor1_signal.png` | `reports/figures/ims_3rd_test_bearing4_sensor1_signal.png` |
| `reports/figures/rul_baseline/rul_predictions.png` | `reports/figures/rf_basic_predictions.png` |
| `reports/figures/rul_baseline/time_split.png` | `reports/figures/rf_basic_time_split.png` |
| `reports/ims_alarm/analysis_metadata.json` | `reports/evaluation/ims_alarm_analysis_metadata.json` |
| `reports/ims_alarm/baseline_checks.csv` | `reports/evaluation/ims_alarm_baseline_checks.csv` |
| `reports/ims_alarm/comparison.csv` | `reports/evaluation/ims_alarm_comparison.csv` |
| `reports/ims_alarm/README.md` | `reports/evaluation/ims_alarm_analysis.md` |
| `reports/ims_alarm/replay_ui_fix.md` | `reports/evaluation/ims_alarm_replay_ui_fix.md` |
| `reports/ims_alarm/rms_all_combinations.html` | `reports/figures/ims_alarm_rms_all_combinations.html` |
| `reports/ims_alarm/summary.csv` | `reports/evaluation/ims_alarm_summary.csv` |

## 삭제 전체 목록

| 삭제 파일 | 이유 |
|---|---|
| `reports/evaluation/bilstm_rul.json` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/evaluation/bilstm_rul_sequence_manifest.csv` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/evaluation/bilstm_rul_split_assignments.csv` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/predictions/bilstm_rul.csv` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/evaluation/experiment_split/bilstm.json` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/figures/bilstm_rul/loss_curve.png` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/figures/bilstm_rul/rul_predictions.png` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/figures/experiment_split/bilstm_test.png` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/predictions/experiment_split/bilstm_test.csv` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/predictions/experiment_split/bilstm_validation.csv` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `reports/predictions/experiment_split/matched_test.csv` | 딥러닝 혼합 비교 자료; RF 원래 결과와 지표는 별도 보존 |
| `reports/figures/experiment_split/comparison.png` | 딥러닝 혼합 비교 자료; RF 원래 결과와 지표는 별도 보존 |
| `reports/evaluation/experiment_split/comparison.csv` | 딥러닝 혼합 비교 자료; RF 원래 결과와 지표는 별도 보존 |
| `reports/evaluation/experiment_split/verification.json` | 현재 검증으로 대체한 옛 개발 검증 기록 |
| `reports/evaluation/verification.json` | 현재 검증으로 대체한 옛 개발 검증 기록 |
| `models/bilstm_rul.pt` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `models/experiment_split/bilstm.pt` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `src/train_bilstm_rul.py` | 제외한 딥러닝 전용 코드·가중치·결과 |
| `app_copy.py` | 딥러닝 비교 전용 옛 UI |
| `reports/ims_alarm/all_tests.log` | 현재 검증으로 대체한 옛 개발 검증 기록 |
| `reports/ims_alarm/replay_ui_tests.log` | 현재 검증으로 대체한 옛 개발 검증 기록 |
| `reports/ims_alarm/tests.log` | 현재 검증으로 대체한 옛 개발 검증 기록 |
