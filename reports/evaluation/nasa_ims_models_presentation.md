# NASA IMS 발표용 모델 작업 결과

작업일: 2026-11-05 (Asia/Seoul). NASA IMS만 사용했으며 원본 TXT를 재처리하지 않았다. 기존 Streamlit·RMS 경보·FFT 워터폴·Ridge/RF 회귀·FastAPI·배포 설정은 수정하지 않았다.

## 1. 머신러닝: 결함 종류 분류

실행한 스크립트: `src/train_ims_fault_types.py`.

`data/processed/features_enhanced.csv`에는 1st/2nd/3rd Test가 모두 있으며 총 37,856행이다. 분류에는 Sensor 1만 사용해 동일 베어링·측정 시각의 중복 센서를 제거했다. 1st B3/B4, 2nd B1, 3rd B3 네 결함 사례만 사용했다. 첫 20%를 Normal proxy, 끝 5%를 시험 종료 기록의 결함 유형으로 가정하고 중간은 제외했다. 백분율은 측정 순서 기준이며 개수는 올림한다. 초기 구간이 실제 정상이라는 보장은 없다.

입력은 RMS, 표준편차, Peak, Kurtosis, Crest Factor, FFT centroid 및 FFT 상위 3개 피크의 주파수·진폭, 총 12개 신호 특징이다. ID·시각·순번·RUL·종료 결함 메타데이터는 입력에서 제외했다. rolling 특징은 사용하지 않았다. 결측 중앙값은 학습 데이터에만 적합했다.

RandomForestClassifier: 300 trees, max_depth=12, min_samples_leaf=2, class_weight=balanced, seed=42. 평가 결과로 파라미터나 구간을 재선택하지 않았다. 모델을 재로드해 저장 예측 재현을 확인했다.

각 베어링의 초기/말기 후보 구간 안에서 앞 70%를 학습에 사용하고, 최소 6개 측정 제외 및 최소 1시간 간격 후 나머지를 평가했다. 같은 원본 TXT를 공유하는 Test 1 B3/B4도 분할이 일치한다. 분할은 `split_manifest.csv`와 `evaluation.json`에서 확인할 수 있다.

| 지표 | RandomForestClassifier | 다수 클래스 단순 기준 |
|---|---:|---:|
| Accuracy | 0.973462 | 0.815440 |
| Macro F1 | 0.881063 | 0.224585 |

학습 2,032행, 평가 829행이다.

| 클래스 | Precision | Recall | F1 | 평가 행 |
|---|---:|---:|---:|---:|
| Normal proxy | 0.9956 | 1.0000 | 0.9978 | 676 |
| Inner Race | 0.7429 | 0.9630 | 0.8387 | 27 |
| Rolling Element | 1.0000 | 0.6667 | 0.8000 | 27 |
| Outer Race | 0.8969 | 0.8788 | 0.8878 | 99 |

**위 수치는 독립 외부 검증 정확도가 아니다.** 말기 학습 구간이 초기 정상 평가 구간보다 미래이므로 전체 수명에 대한 순방향 미래 예측도 아니다. 클래스마다 하나 이상의 학습 예제를 확보하기 위한 회고적 구간별 블록 평가다. 별도로 전체 수명을 앞 70% 학습/뒤쪽 평가로 나누는 진단을 실행했으나, 학습에 Normal proxy만 남아 네 클래스 검증이 불가능했다(진단 Accuracy/Macro F1 모두 0). 이 결과는 주 성능과 분리해 `strict_forward_diagnostic`에 기록했다.

Inner Race와 Rolling Element는 각각 베어링 하나뿐이다. 결함 유형과 Test/Bearing 특성이 혼동될 수 있고, 같은 베어링 내부 평가는 독립 설비 일반화가 아니다. 신뢰할 네 클래스 외부 검증은 현재 사례 수로 불가능하다. Outer Race 평가 중 Test 2 말기 표본은 9개뿐이다. 클래스 불균형과 시간 자기상관도 남아 있다. 품질 의심 종료 기록은 삭제하지 않았으므로 측정 중단/센서 변화 영향도 고려해야 한다.

산출물 (`reports/ims_fault_types/`):

- `rf_fault_types.joblib`: 분류 파이프라인·특징 목록·클래스·대리 라벨 정의
- `evaluation.json`, `evaluation.md`: 지표·방법·입력 해시·설정·한계
- `classification_report.csv`, `confusion_matrix.csv`, `test_predictions.csv`
- `split_manifest.csv`, `forward_split_manifest.csv`
- `feature_importance.csv`, `permutation_importance.csv`
- **발표 그림:** `confusion_matrix.png`, `feature_importance.png`, `split_timeline.png`

## 2. 딥러닝: BiLSTM 대리 RUL 회귀 복구

시작 시 `src/train_bilstm_rul.py`, `models/bilstm_rul.pt`, `reports/bilstm_rul/`가 모두 없었다. Git에서 복구했으며 기존 파일은 덮어쓰지 않았다.

- 학습 코드와 체크포인트: `80c2ff1`의 CSV 통합 이후 버전. 기존 `train_rul_baseline`의 공통 CSV 로딩·입력 호환성 검증을 재사용한다.
- 과거 평가·예측·학습 이력·그림: `fec6c7b^`에서 원본 그대로 복구했다.
- 학습 코드가 요구하는 과거 기준 평가·예측 파일도 부재하여 `80c2ff1`에서 `reports/evaluation/rul_baseline.json`, `reports/predictions/rul_baseline.csv`로 복구했다. 기존 RF 모델은 수정하지 않았다.
- 복구한 각 파일의 실제 커밋과 SHA-256은 `reports/bilstm_rul/restoration_manifest.json`에 기록했다. 과거 `verification.json`은 당시 기록이며, 이번 검증은 별도 `restoration_verification.json`이다.

현재 `data/processed/1st_test.csv`를 사용한 CPU 추론으로 과거 평가 676개 시점(B3/B4 각각 338개)을 재현했다. 원래 입력 CSV는 현재 경로와 다르지만 프로젝트의 기존 입력 호환성 검증을 통과했다. 과거 예측의 절대 원본 경로 문자열은 현재 상대 경로와 다르며, 시각·베어링·정답 및 특징/전처리 호환성을 검증했다. 원본 경로를 임의 치환하거나 과거 기록을 수정하지 않았다.

| 모델/기준 | MAE (시간) | RMSE (시간) |
|---|---:|---:|
| BiLSTM | 57.498655 | 66.881397 |
| 마지막 학습 정답 고정 | 38.209533 | 45.070097 |

저장된 예측과 CPU 재추론의 최대 절대 차이는 약 0.00000759시간이다. **BiLSTM은 단순 기준보다 오차가 크며 성능 개선을 주장할 수 없다.** 복구에 성공했으므로 새 학습은 하지 않았다.

구조: 센서 2개 × 신호 특징 12개 = 입력 24개, 과거 관측 시퀀스 16개, hidden size 32, 1-layer bidirectional LSTM. 양방향 hidden 64개를 `Linear(64,16) → ReLU → Dropout(0.1) → Linear(16,1) → Softplus`로 출력한다. 총 파라미터 15,905개. 학습 당시 CPU, AdamW, learning rate=0.001, batch size=32, 고정 20 epochs, 마지막 epoch 체크포인트다. 평가 점수로 가장 좋은 epoch를 고르지 않았다.

과거 80% 학습/뒤 20% 내부 평가이며 시퀀스가 분할 경계·측정 순번 누락·20분 초과 공백을 넘지 않는다. 양방향 처리는 예측 시점까지 주어진 과거 시퀀스 안에서만 수행한다. 스케일러·결측 보정·정답 크기 조정이 학습 행에만 적합된 것도 재검증했다. 동일 베어링의 겹치는 시퀀스는 독립 실험이 아니다. 정답은 실제 고장까지의 시간이 아니라 **시험 마지막 기록까지 남은 달력 시간**이다. 평가 가능한 시퀀스 끝점만 사용하므로 원래 전체 행 평가와 대상을 구분해야 한다.

주요 산출물:

- `src/train_bilstm_rul.py`: 복구한 원래 학습 코드
- `models/bilstm_rul.pt`: 복구한 실제 epoch 20 체크포인트
- `src/verify_restored_bilstm.py`: 현재 CSV로 CPU 추론·호환성·해시 검증
- `reports/bilstm_rul/evaluation.json`, `evaluation.md`, `metrics.csv`, `test_predictions.csv`, `training_history.csv`
- `reports/bilstm_rul/restoration_manifest.json`, `restoration_verification.json`, `restored_cpu_predictions.csv`
- **발표 그림:** `reports/bilstm_rul/restored_predictions_verified.png` (이번 CPU 추론), `rul_predictions.png`, `loss_curve.png` (과거 복구 그림)

## 3. 실제 실행 명령과 검증

```powershell
.\.venv\Scripts\python.exe -m src.train_ims_fault_types
.\.venv\Scripts\python.exe -m src.verify_restored_bilstm
.\.venv\Scripts\python.exe src/train_bilstm_rul.py --help
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ims_fault_types.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_project.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_presentation_ui.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cloud_waterfall.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ims_alarm.py
git diff --check
```

테스트 4 + 24 + 3 + 3 + 11 = **45개 통과**. 새 모델 지표 재계산, 중복 센서·누수 메타데이터 제외, 분할·시간 간격, 모델 재로드, 복구 해시와 기존 대시보드·저장 회귀 예측·클라우드 FFT·RMS 경보 회귀를 확인했다. 세 발표 그림은 직접 열어 제목·축·내용을 확인했다.

최초 프로젝트 테스트에서는 보고서 폴더가 정확히 3개라는 과거 구조 제한 때문에 1개 실패했다. 사용자 요청의 `ims_fault_types`, `bilstm_rul` 두 폴더를 허용하도록 해당 구조 테스트만 갱신한 뒤 24개 모두 통과했다. 미해결 실행 실패는 없다. 독립 네 클래스 외부 검증은 데이터 사례 부족으로 수행할 수 없었으며, BiLSTM 재학습·외부 API 호출·클라우드 재배포는 수행하지 않았다. 추가 패키지는 설치하지 않았다.

`.gitignore`에는 복구 체크포인트와 새 분류 모델 파일만 예외로 추가했다. `features_enhanced.csv`는 기존처럼 제외되지만, 새 분류 스크립트는 이 파일이 없는 환경에서 이미 추적 중인 1st/2nd/3rd 특징 CSV로 대체할 수 있다. 원본 TXT는 계속 제외한다. `deploy-latest`를 유지했으며 자동 Commit/Push는 하지 않았다.

재실행은 기존 결과 덮어쓰기를 거부한다. 새 분류 실험은 `--output-dir`로 새 폴더를 지정한다. 이번에 보고한 지표는 저장 모델 그대로의 평가이며 평가 데이터를 포함한 재학습 모델을 배포한 결과가 아니다.
