# NASA IMS BiLSTM 대리 RUL

평가 상태: 1st Test 내부 회고적 시간순 시퀀스 평가; 독립 실험 일반화 검증 아님

입력: 두 센서의 신호 특징 24개 × 과거 16시점. 정답은 최종 입력 시점의 `rul_calendar_hours_proxy`이며 단위는 시간이다.

기존과 동일한 80/20 분할: 학습 1724시점, 평가 432시점.

학습 마지막: 2003-11-21 22:16:56, 평가 첫 시점: 2003-11-21 22:26:56.

시퀀스: 학습 2954개, 평가 676개. 평가 최종 시점 제외 행: 188개.

경계와 1200초 초과 공백을 넘지 않으며 부족한 과거를 학습 구간에서 가져오거나 채우지 않았다.

## 동일한 평가 최종 시점에서 비교

| 범위 | 모델 | MAE (시간) | RMSE (시간) |
|---|---|---:|---:|
| overall | BiLSTM | 57.4987 | 66.8814 |
| overall | 기존 Random Forest | 86.2277 | 99.5308 |
| overall | 마지막 학습 정답 고정 | 38.2095 | 45.0701 |
| overall | 학습 정답 중앙값 고정 | 313.1423 | 314.0533 |
| bearing_3 | BiLSTM | 66.6572 | 74.3730 |
| bearing_3 | 기존 Random Forest | 96.6759 | 108.8090 |
| bearing_3 | 마지막 학습 정답 고정 | 38.2095 | 45.0701 |
| bearing_3 | 학습 정답 중앙값 고정 | 313.1423 | 314.0533 |
| bearing_4 | BiLSTM | 48.3401 | 58.4372 |
| bearing_4 | 기존 Random Forest | 75.7796 | 89.2938 |
| bearing_4 | 마지막 학습 정답 고정 | 38.2095 | 45.0701 |
| bearing_4 | 학습 정답 중앙값 고정 | 313.1423 | 314.0533 |

BiLSTM의 MAE가 동일 최종 시점의 마지막 학습 정답 고정 기준보다 크거나 같다. 해당 기준 대비 개선을 입증하지 못했다.

## 기존 전체 평가값 보존

아래는 기존 전체 864개 베어링·시점의 기록이며 위 공통 시점 평가와 모집단이 다르다.

| 모델 | MAE (시간) | RMSE (시간) |
|---|---:|---:|
| 기존 Random Forest | 90.4298 | 104.0858 |
| 마지막 학습 정답 고정 | 40.6062 | 47.8882 |

## 학습 설정

CPU/GPU 장치: cpu, epoch=20, 배치=32, 은닉 크기=32, 파라미터=15,905개.

결측값 중앙값과 특징별 평균·표준편차는 원래 학습 구간의 행에만 적합했다. 정답은 학습 정답 평균 383.13367153시간으로 나눠 학습하고 예측을 시간 단위로 복원했다.

최종 모델은 고정된 마지막 epoch다. 평가값에 따른 체크포인트 선택과 평가 데이터 재학습은 없다.

## 한계

- 정답 rul_calendar_hours_proxy는 시험 마지막 기록까지 남은 달력 시간의 대리값이다. 정확한 고장 시각이나 실제 잔여 수명·가동시간이 아니다.
- 학습·평가가 같은 1st Test와 같은 Bearing 3·4의 수명 궤적을 공유하는 회고적 내부 평가다. 독립 실험·다른 베어링·현장 일반화 성능을 검증하지 않았다.
- 학습 정답도 시험 종료 후 알려진 종료점을 이용해 생성되었다. 분할 시점에 해당 정답으로 학습한 모델을 실제 운용할 수 있었다는 뜻이 아니다.
- 측정 시각·측정 순번·종료 시각·RUL 파생 열·베어링 ID는 모델 특징에 넣지 않는다. 시각과 순번은 분할·순서·공백 검사에만 사용한다.
- 양방향 LSTM은 최종 입력 시점까지의 과거 관측만 양방향으로 처리한다. 예측 시점 이후의 신호나 정답은 입력에 없다.
- 학습·평가 시퀀스는 각 분할 안에서 독립적으로 만든다. 평가 초반의 입력을 채우려고 학습 구간을 끌어오지 않는다.
- 시퀀스 준비 구간과 긴 공백 뒤의 짧은 구간은 평가에서 제외된다. 기존 전체 평가와 대상이 달라지므로 공통 최종 시점에 맞춘 기준 모델 지표를 별도로 계산한다.
- 입력 간격은 일정하지 않다. 시퀀스는 관측 순서의 여러 시점이며 고정된 실제 시간 길이를 뜻하지 않는다. 공백 기준 기본값은 기존 전처리와 같은 20분이다.
- 겹치는 시퀀스와 두 베어링은 서로 독립인 수명 실험이 아니다. 시퀀스 수를 독립 실험 수로 해석하지 않는다.
- 매 epoch 평가 구간의 손실을 기록하지만 최종 모델은 사전에 정한 마지막 epoch다. 조기 종료·학습률 조정·체크포인트 선택에 평가 점수를 쓰지 않는다.
- 스케일러·결측 중앙값·정답 크기 조정은 앞쪽 학습 구간에만 적합한다. 평가 구간으로 재학습하거나 하이퍼파라미터를 선택하지 않는다.
- 학습 구간과 평가 구간의 RUL 범위가 겹치지 않는다. BiLSTM이 수학적으로 외삽할 수 있어도 종료 직전의 작은 RUL을 정확히 예측한다는 보장은 없다.
- 단순 기준도 학습 정답만 사용하지만 그 정답 자체는 종료 후의 회고적 대리값이다. 실제 이용 가능한 잔여 수명 기준으로 오인하면 안 된다.

## 입력 특징

sensor1_rms, sensor1_std, sensor1_peak, sensor1_kurtosis, sensor1_crest_factor, sensor1_fft_centroid_hz, sensor1_fft_peak_1_hz, sensor1_fft_peak_1_amplitude, sensor1_fft_peak_2_hz, sensor1_fft_peak_2_amplitude, sensor1_fft_peak_3_hz, sensor1_fft_peak_3_amplitude, sensor2_rms, sensor2_std, sensor2_peak, sensor2_kurtosis, sensor2_crest_factor, sensor2_fft_centroid_hz, sensor2_fft_peak_1_hz, sensor2_fft_peak_1_amplitude, sensor2_fft_peak_2_hz, sensor2_fft_peak_2_amplitude, sensor2_fft_peak_3_hz, sensor2_fft_peak_3_amplitude

분할은 split_assignments.csv, 시퀀스 범위는 sequence_manifest.csv, 예측은 test_predictions.csv에 저장했다.
체크포인트에 모델 구조·가중치·특징 순서·시퀀스 길이·스케일러 통계·정답 크기 조정을 함께 저장했다.

## 실행 명령

```powershell
C:\bearing-pdm\.venv\Scripts\python.exe src\train_bilstm_rul.py --epochs 20 --sequence-length 16 --hidden-size 32 --batch-size 32 --device cpu
```

같은 출력 위치에 재실행하면 이번 BiLSTM 산출물을 덮어쓴다. 기존 소스·모델·평가 파일은 변경하지 않는다.
