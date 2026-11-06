"""NASA IMS 1st Test의 과거 신호 특징 시퀀스로 종료 기준 대리 RUL을 예측한다.

기존 기준 모델과 같은 시간순 분할을 사용하며 경계와 긴 기록 공백을 넘지 않는다.
양방향 처리는 예측 시점까지 관측된 과거 시퀀스 안에서만 수행한다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

# 기존 모듈을 불러올 때 기존 바이트코드도 변경하지 않는다.
sys.dont_write_bytecode = True

import matplotlib

# 화면을 띄우지 않고 그래프 파일을 저장한다.
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .train_rul_baseline import DEFAULT_INPUT, MODEL_FEATURES, SIGNAL_FEATURES, TARGET, TRAINING_EXPERIMENTS, chronological_split, load_measurements, validate_reference_input, scores, sha256
else:
    from train_rul_baseline import DEFAULT_INPUT, MODEL_FEATURES, SIGNAL_FEATURES, TARGET, TRAINING_EXPERIMENTS, chronological_split, load_measurements, validate_reference_input, scores, sha256

ROOT = Path(__file__).resolve().parents[1]

METHODS = {"bilstm": "BiLSTM", "random_forest": "기존 Random Forest",
           "last_train_target": "마지막 학습 정답 고정", "train_median": "학습 정답 중앙값 고정"}
LIMITATIONS = [
    "정답 rul_calendar_hours_proxy는 시험 마지막 기록까지 남은 달력 시간의 대리값이다. 정확한 고장 시각이나 실제 잔여 수명·가동시간이 아니다.",
    "학습·평가가 같은 1st Test와 같은 Bearing 3·4의 수명 궤적을 공유하는 회고적 내부 평가다. 독립 실험·다른 베어링·현장 일반화 성능을 검증하지 않았다.",
    "학습 정답도 시험 종료 후 알려진 종료점을 이용해 생성되었다. 분할 시점에 해당 정답으로 학습한 모델을 실제 운용할 수 있었다는 뜻이 아니다.",
    "측정 시각·측정 순번·종료 시각·RUL 파생 열·베어링 ID는 모델 특징에 넣지 않는다. 시각과 순번은 분할·순서·공백 검사에만 사용한다.",
    "양방향 LSTM은 최종 입력 시점까지의 과거 관측만 양방향으로 처리한다. 예측 시점 이후의 신호나 정답은 입력에 없다.",
    "학습·평가 시퀀스는 각 분할 안에서 독립적으로 만든다. 평가 초반의 입력을 채우려고 학습 구간을 끌어오지 않는다.",
    "시퀀스 준비 구간과 긴 공백 뒤의 짧은 구간은 평가에서 제외된다. 기존 전체 평가와 대상이 달라지므로 공통 최종 시점에 맞춘 기준 모델 지표를 별도로 계산한다.",
    "입력 간격은 일정하지 않다. 시퀀스는 관측 순서의 여러 시점이며 고정된 실제 시간 길이를 뜻하지 않는다. 공백 기준 기본값은 기존 전처리와 같은 20분이다.",
    "겹치는 시퀀스와 두 베어링은 서로 독립인 수명 실험이 아니다. 시퀀스 수를 독립 실험 수로 해석하지 않는다.",
    "매 epoch 평가 구간의 손실을 기록하지만 최종 모델은 사전에 정한 마지막 epoch다. 조기 종료·학습률 조정·체크포인트 선택에 평가 점수를 쓰지 않는다.",
    "스케일러·결측 중앙값·정답 크기 조정은 앞쪽 학습 구간에만 적합한다. 평가 구간으로 재학습하거나 하이퍼파라미터를 선택하지 않는다.",
    "학습 구간과 평가 구간의 RUL 범위가 겹치지 않는다. BiLSTM이 수학적으로 외삽할 수 있어도 종료 직전의 작은 RUL을 정확히 예측한다는 보장은 없다.",
    "단순 기준도 학습 정답만 사용하지만 그 정답 자체는 종료 후의 회고적 대리값이다. 실제 이용 가능한 잔여 수명 기준으로 오인하면 안 된다.",
]




def build_sequences(frame, length, max_gap_seconds, partition):
    """베어링별 과거 구간의 행 번호와 최종 시점 정답을 만들며 공백과 분할을 넘지 않는다."""
    frame = frame.reset_index(drop=True).copy()
    if "partition" in frame and not frame["partition"].eq(partition).all():
        raise ValueError("입력 데이터에 다른 분할의 행이 섞였습니다.")
    windows, endpoints = [], []
    segment_lengths = {}
    for bearing, group in frame.groupby("bearing_id", sort=True):
        group = group.sort_values("measured_at")
        seconds = group["measured_at"].diff().dt.total_seconds()
        if seconds.iloc[1:].le(0).any():
            raise ValueError("베어링별 측정 시각이 엄격하게 증가해야 합니다.")
        # 측정 순번이 빠진 곳과 긴 기록 공백에서는 과거 문맥을 이어 붙이지 않는다.
        boundaries = seconds.gt(max_gap_seconds) | group["measurement_index"].diff().ne(1)
        segments = boundaries.cumsum()
        segment_lengths[str(bearing)] = []
        for _, segment in group.groupby(segments, sort=True):
            positions = segment.index.to_numpy(dtype=np.int64)
            segment_lengths[str(bearing)].append(len(positions))
            for end in range(length - 1, len(positions)):
                selected = positions[end - length + 1:end + 1]
                first, last = frame.iloc[int(selected[0])], frame.iloc[int(selected[-1])]
                intervals = frame.iloc[selected]["measured_at"].diff().dt.total_seconds().iloc[1:]
                if len(selected) != length or intervals.gt(max_gap_seconds).any() or intervals.le(0).any():
                    raise ValueError("시퀀스 순서 또는 공백 검사에 실패했습니다.")
                if not frame.iloc[selected]["bearing_id"].eq(bearing).all():
                    raise ValueError("서로 다른 베어링의 특징이 한 시퀀스에 섞였습니다.")
                windows.append(selected)
                endpoints.append({
                    "partition": partition, "bearing_id": int(bearing), "measured_at": last["measured_at"],
                    "sequence_start_at": first["measured_at"], "sequence_length": length,
                    "sequence_span_hours": (last["measured_at"] - first["measured_at"]).total_seconds() / 3600,
                    "max_interval_seconds": float(intervals.max()),
                    "sequence_start_measurement_index": int(first["measurement_index"]),
                    "measurement_index": int(last["measurement_index"]), "source_path": last["source_path"],
                    TARGET: float(last[TARGET]),
                })
    if not windows:
        raise ValueError(f"{partition} 구간에서 길이 {length}의 시퀀스를 만들 수 없습니다.")
    endpoint_frame = pd.DataFrame(endpoints)
    audit = {"partition": partition, "sequences": len(windows), "input_rows": len(frame),
             "excluded_endpoint_rows": len(frame) - len(windows), "segment_lengths": segment_lengths,
             "per_bearing_sequences": {str(k): int(v) for k, v in endpoint_frame.groupby("bearing_id").size().items()},
             "first_input_at": str(endpoint_frame["sequence_start_at"].min()),
             "first_endpoint_at": str(endpoint_frame["measured_at"].min()),
             "last_endpoint_at": str(endpoint_frame["measured_at"].max()),
             "sequence_span_hours_range": [float(endpoint_frame["sequence_span_hours"].min()),
                                           float(endpoint_frame["sequence_span_hours"].max())],
             "future_after_endpoint_used": False, "passed": True}
    return frame, np.stack(windows), endpoint_frame, audit


class RULSequenceDataset(Dataset):
    """시간 정보 없이 표준화한 신호 특징만 과거 시퀀스로 반환한다."""

    def __init__(self, features, windows, targets, target_scale):
        self.features = torch.as_tensor(np.asarray(features, dtype=np.float32))
        self.windows = torch.as_tensor(windows, dtype=torch.long)
        self.targets = torch.as_tensor(np.asarray(targets / target_scale, dtype=np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.features[self.windows[index]], self.targets[index]


class BiLSTMRUL(nn.Module):
    """과거 구간의 양방향 최종 은닉 상태를 결합해 음수가 아닌 대리 RUL을 출력한다."""

    def __init__(self, input_size=24, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden_size * 2, 16), nn.ReLU(), nn.Dropout(.1),
                                  nn.Linear(16, 1), nn.Softplus())

    def forward(self, features):
        # 역방향도 예측 시점 이후가 아니라 주어진 과거 구간의 끝에서 시작한다.
        _, (hidden, _) = self.lstm(features)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(representation).squeeze(-1)


def evaluate_model(model, loader, device, target_scale):
    """평가에서는 드롭아웃·기울기 계산을 끄고 예측을 원래 시간 단위로 복원한다."""
    model.eval()
    total_loss, predictions = 0.0, []
    with torch.inference_mode():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            predicted = model(features)
            total_loss += nn.functional.mse_loss(predicted, target).item() * len(target)
            predictions.append(predicted.cpu().numpy() * target_scale)
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise ValueError("예측 RUL에 비유한 값이 발생했습니다.")
    return total_loss / len(loader.dataset), result


def pair_reference(endpoints, reference_predictions):
    """기존 예측을 동일한 베어링·최종 측정 시점에 연결하고 정답·파일의 일치를 검증한다."""
    keys = ["bearing_id", "measured_at"]
    reference_predictions = reference_predictions.copy()
    reference_predictions["measured_at"] = pd.to_datetime(reference_predictions["measured_at"], errors="raise")
    merged = endpoints.merge(reference_predictions, on=keys, how="left", suffixes=("", "_reference"),
                             validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("일부 평가 최종 시점에 기존 기준 모델 예측이 없습니다.")
    if not np.allclose(merged[TARGET], merged[TARGET + "_reference"], rtol=0, atol=1e-8):
        raise ValueError("기존 예측과 현재 데이터의 RUL 정답이 다릅니다.")
    # 기존 예측은 과거 절대 경로를 보존한다. 첫 실험의 고유 측정 파일명으로 대조한다.
    current_files = merged["source_path"].map(lambda value: Path(value).name)
    reference_files = merged["source_path_reference"].map(lambda value: Path(value).name)
    if not current_files.eq(reference_files).all():
        raise ValueError("기존 예측과 현재 데이터의 원본 파일이 다릅니다.")
    columns = [f"prediction_{method}" for method in METHODS if method != "bilstm"]
    if not np.isfinite(merged[columns].to_numpy(dtype=float)).all():
        raise ValueError("기존 기준 모델 예측이 누락되었거나 비유한 값입니다.")
    return merged[endpoints.columns.tolist() + columns]


def matched_metrics(predictions):
    """모든 방법을 동일 평가 최종 시점에서 전체·베어링별로 비교한다."""
    result = {}
    groups = [("overall", predictions)] + [(f"bearing_{bearing}", part)
                                             for bearing, part in predictions.groupby("bearing_id")]
    for scope, part in groups:
        result[scope] = {method: scores(part[TARGET], part[f"prediction_{method}"]) for method in METHODS}
    return result


def save_plots(predictions, history, directory, provisional, max_gap_seconds):
    """동일 최종 시점의 RUL 비교와 학습·평가 손실 그림을 저장한다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, layout="constrained")
    colors = {"bilstm": "#d45b27", "random_forest": "#277da1", "last_train_target": "#64965b"}
    for bearing, ax in zip((3, 4), axes):
        group = predictions.loc[predictions["bearing_id"].eq(bearing)].sort_values("measured_at")
        segments = group["measured_at"].diff().dt.total_seconds().gt(max_gap_seconds).cumsum()
        for number, (_, part) in enumerate(group.groupby(segments)):
            ax.plot(part["measured_at"], part[TARGET], ".-", color="#222222", markersize=2,
                    label="종료 기준 대리 정답" if number == 0 else None)
            for method, color in colors.items():
                ax.plot(part["measured_at"], part[f"prediction_{method}"],
                        "--" if method == "last_train_target" else ".-", color=color,
                        markersize=2, linewidth=1, label=METHODS[method] if number == 0 else None)
        ax.set(title=f"Bearing {bearing}: 동일한 평가 최종 시점에서 비교", ylabel="종료까지 남은 달력 시간 (시간)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].set_xlabel("최종 입력 측정 시각; 긴 공백에서는 선을 끊음")
    fig.suptitle(("1 epoch 실행 점검 — " if provisional else "") +
                 "NASA IMS BiLSTM 내부 회고적 평가\n실제 고장 수명이 아닌 시험 종료 기준 대리 RUL")
    fig.savefig(directory / "rul_predictions.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(history["epoch"], history["train_loss"], "o-", label="학습 손실")
    ax.plot(history["epoch"], history["evaluation_loss"], "o-", label="평가 손실; 모델 선택에 사용하지 않음")
    ax.set(xlabel="학습 epoch", ylabel="학습 정답 평균으로 크기를 조정한 RUL의 평균 제곱 오차", title="BiLSTM 학습·평가 손실")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(directory / "loss_curve.png", dpi=160)
    plt.close(fig)




def save_results(report, predictions, split, sequences, checkpoint, report_dir, model_output):
    """가중치·평가·분할·예측·그림만 저장한다. 학습 이력은 평가 JSON에 포함된다."""
    report_dir, model_output = Path(report_dir), Path(model_output)
    evaluation = report_dir / "evaluation"
    prediction_file = report_dir / "predictions/bilstm_rul.csv"
    figures = report_dir / "figures/bilstm_rul"
    targets = [model_output, evaluation / "bilstm_rul.json", prediction_file,
               evaluation / "bilstm_rul_split_assignments.csv", evaluation / "bilstm_rul_sequence_manifest.csv",
               figures / "rul_predictions.png", figures / "loss_curve.png"]
    if any(path.exists() for path in targets):
        raise FileExistsError("기존 BiLSTM 결과를 덮어쓰지 않습니다. 새 보고서 루트와 모델 경로를 지정하세요.")
    for directory in [evaluation, prediction_file.parent, figures, model_output.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_output)
    predictions.to_csv(prediction_file, index=False, encoding="utf-8-sig")
    sequences.to_csv(evaluation / "bilstm_rul_sequence_manifest.csv", index=False, encoding="utf-8-sig")
    split[["measured_at", "bearing_id", "source_path", "measurement_index", "partition"]].to_csv(
        evaluation / "bilstm_rul_split_assignments.csv", index=False, encoding="utf-8-sig")
    save_plots(predictions, pd.DataFrame(report["history"]), figures, report["provisional"], report["max_gap_seconds"])
    report = dict(report, model_output=str(model_output), model_sha256=sha256(model_output))
    (evaluation / "bilstm_rul.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    """고정된 시간순 분할에서 작은 양방향 LSTM을 학습하고 동일 시점 비교를 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-report", type=Path, default=ROOT / "reports/evaluation/rul_baseline.json")
    parser.add_argument("--reference-predictions", type=Path, default=ROOT / "reports/predictions/rul_baseline.csv")
    parser.add_argument("--sequence-length", type=int, choices=(16, 32), default=16)
    parser.add_argument("--max-gap-seconds", type=float, default=1200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch 실행 점검; 별도 모델·보고서 경로 필요")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models/bilstm_rul.pt")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if min(args.epochs, args.hidden_size, args.batch_size, args.threads) < 1:
        parser.error("epoch·은닉 크기·배치·스레드는 양수여야 합니다.")
    if not all(math.isfinite(value) and value > 0 for value in (args.learning_rate, args.max_gap_seconds)):
        parser.error("학습률과 공백 기준은 양의 유한값이어야 합니다.")
    for name in ("input", "reference_report", "reference_predictions", "model_output", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not all(path.is_file() for path in (args.input, args.reference_report, args.reference_predictions)):
        print("NASA IMS 특징·기준 평가·기준 예측 파일이 필요합니다. 학습을 비활성화합니다.")
        return 0
    if args.model_output.exists() or (args.report_dir / "evaluation/bilstm_rul.json").exists() or (args.report_dir / "predictions/bilstm_rul.csv").exists():
        parser.error("기존 모델·평가 결과를 덮어쓰지 않습니다. 새 --model-output과 --report-dir을 지정하세요.")
    if args.model_output.suffix != ".pt":
        parser.error("모델 출력은 .pt 파일이어야 합니다.")
    if args.smoke_test and (args.epochs != 1 or args.model_output == ROOT / "models/bilstm_rul.pt"
                            or args.report_dir == ROOT / "reports"):
        parser.error("점검 실행에는 1 epoch와 별도 모델·보고서 경로를 지정하세요.")
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    data, inspection = load_measurements(args.input)
    validate_reference_input(args.input, data, reference)
    train, test, split, split_audit = chronological_split(data, .8)
    if split_audit != reference["split_audit"] or MODEL_FEATURES != reference["model_features"]:
        raise ValueError("분할 경계 또는 특징 결합이 기존 RUL 기준 모델과 다릅니다.")
    train, train_windows, train_endpoints, train_audit = build_sequences(train, args.sequence_length, args.max_gap_seconds, "train")
    test, test_windows, test_endpoints, test_audit = build_sequences(test, args.sequence_length, args.max_gap_seconds, "test")
    if train_endpoints["measured_at"].max() >= test_endpoints["sequence_start_at"].min():
        raise ValueError("평가 시퀀스가 학습 구간을 침범합니다.")
    if train[MODEL_FEATURES].isna().all().any():
        raise ValueError("학습 전체가 결측인 특징이 있습니다.")
    scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    # 시퀀스 생성 전에 분할한 앞부분의 행만 스케일러 적합에 사용한다.
    train_features = scaler.fit_transform(train[MODEL_FEATURES]).astype(np.float32)
    test_features = scaler.transform(test[MODEL_FEATURES]).astype(np.float32)
    if not np.isfinite(train_features).all() or not np.isfinite(test_features).all():
        raise ValueError("스케일링 후 비유한 특징이 있습니다.")
    target_scale = float(train[TARGET].mean())
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("학습 정답 크기 조정 값이 양수가 아닙니다.")
    reference_predictions = pd.read_csv(args.reference_predictions, encoding="utf-8-sig")
    if len(reference_predictions) != len(test):
        raise ValueError("기존 전체 평가 예측 행 수가 현재 분할과 다릅니다.")
    for method in ("random_forest", "last_train_target", "train_median"):
        metric = scores(reference_predictions[TARGET], reference_predictions[f"prediction_{method}"])
        if not all(np.isclose(value, reference["metrics"]["overall"][method][key]) for key, value in metric.items()):
            raise ValueError("기존 전체 평가 예측이 저장된 평가 기록과 다릅니다.")
    predictions = pair_reference(test_endpoints, reference_predictions)
    print("PyTorch:", str(torch.__version__), flush=True)
    print("실제 CSV 컬럼:", inspection["csv_columns"], flush=True)
    print("모델 입력 특징:", MODEL_FEATURES, flush=True)
    print("기존과 동일한 시간순 분할:", json.dumps(split_audit, ensure_ascii=False), flush=True)
    print("시퀀스 검사:", json.dumps({"train": train_audit, "test": test_audit}, ensure_ascii=False), flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA가 없습니다. --device cpu를 사용하세요.")
        torch.cuda.manual_seed_all(args.seed)
    train_dataset = RULSequenceDataset(train_features, train_windows, train_endpoints[TARGET].to_numpy(), target_scale)
    test_dataset = RULSequenceDataset(test_features, test_windows, test_endpoints[TARGET].to_numpy(), target_scale)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = BiLSTMRUL(len(MODEL_FEATURES), args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(features), target)
            if not torch.isfinite(loss):
                raise ValueError("학습 손실에 비유한 값이 발생했습니다.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(target)
        evaluation_loss, predicted = evaluate_model(model, test_loader, device, target_scale)
        values = {"epoch": epoch, "train_loss": total_loss / len(train_dataset), "evaluation_loss": evaluation_loss,
                  **scores(test_endpoints[TARGET], predicted)}
        history.append(values)
        print(f"epoch {epoch}/{args.epochs}: train_loss={values['train_loss']:.6f}, "
              f"evaluation_loss={evaluation_loss:.6f}, MAE={values['MAE_hours']:.4f}시간, "
              f"RMSE={values['RMSE_hours']:.4f}시간", flush=True)
    elapsed = time.perf_counter() - started
    predictions["prediction_bilstm"] = predicted
    metrics = matched_metrics(predictions)
    overall = metrics["overall"]
    comparison = ("BiLSTM의 MAE가 동일 최종 시점의 마지막 학습 정답 고정 기준보다 작다." if
                  overall["bilstm"]["MAE_hours"] < overall["last_train_target"]["MAE_hours"] else
                  "BiLSTM의 MAE가 동일 최종 시점의 마지막 학습 정답 고정 기준보다 크거나 같다. 해당 기준 대비 개선을 입증하지 못했다.")
    preprocessing = {"kind": "training_median_then_standard_scaler", "fit_rows": len(train),
                     "feature_names": list(MODEL_FEATURES),
                     "imputer_medians": scaler.named_steps["imputer"].statistics_.tolist(),
                     "mean": scaler.named_steps["scaler"].mean_.tolist(),
                     "scale": scaler.named_steps["scaler"].scale_.tolist(),
                     "variance": scaler.named_steps["scaler"].var_.tolist()}
    report = {"input": str(args.input), "input_sha256": sha256(args.input), "script_sha256": sha256(Path(__file__)),
              "inspection": inspection, "model_features": list(MODEL_FEATURES), "signal_features": list(SIGNAL_FEATURES),
              "split_audit": split_audit, "sequence_audit": {"train": train_audit, "test": test_audit},
              "sequence_length": args.sequence_length, "max_gap_seconds": args.max_gap_seconds,
              "sequence_policy": "분할 안의 동일 베어링 과거만 사용; 경계·측정 순번 누락·긴 기록 공백에서 분리",
              "preprocessing": preprocessing, "target": TARGET, "target_unit": "달력 시간(시간)",
              "target_scale_hours": target_scale, "target_scale_fitted_on": "원래 앞 80% 학습 행의 정답 평균",
              "target_ranges_hours": {"train": [float(train[TARGET].min()), float(train[TARGET].max())],
                                      "eligible_test": [float(test_endpoints[TARGET].min()), float(test_endpoints[TARGET].max())]},
              "matched_endpoint_metrics": metrics, "original_full_reference_metrics": reference["metrics"],
              "comparison_note": comparison, "reference_report": str(args.reference_report),
              "reference_report_sha256": sha256(args.reference_report), "reference_predictions": str(args.reference_predictions),
              "reference_predictions_sha256": sha256(args.reference_predictions),
              "same_original_split": True, "fair_comparison_endpoint_rows": len(predictions),
              "epochs": args.epochs, "selected_epoch": args.epochs, "hidden_size": args.hidden_size,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate, "device": str(device),
              "threads": args.threads, "seed": args.seed, "parameter_count": sum(p.numel() for p in model.parameters()),
              "optimizer": "AdamW", "weight_decay": .0001, "gradient_clip_norm": 1.0,
              "history": history, "training_seconds": elapsed,
              "checkpoint_selection": "고정된 마지막 epoch; 평가 점수에 의한 선택 없음", "refit_with_test": False,
              "provisional": args.smoke_test, "evaluation_status": "1 epoch 실행 점검용" if args.smoke_test else
              "1st Test 내부 회고적 시간순 시퀀스 평가; 독립 실험 일반화 검증 아님",
              "limitations": list(LIMITATIONS), "model_output": str(args.model_output),
              "training_experiments": list(TRAINING_EXPERIMENTS),
              "evaluation_protocol": "첫 실험 내부 회고적 시간순 분할; 실험 간 독립 평가 아님. 다른 실험을 합치지 않음.",
              "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
              "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    if args.smoke_test:
        report["limitations"].append("1 epoch 실행은 오류 확인용이며 최종 모델 성능으로 제시하지 않는다.")
    checkpoint = {"format_version": 1, "architecture": "BiLSTMRUL", "input_size": len(MODEL_FEATURES),
                  "hidden_size": args.hidden_size, "bidirectional": True, "num_layers": 1,
                  "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "preprocessing": preprocessing, "sequence_length": args.sequence_length,
                  "max_gap_seconds": args.max_gap_seconds, "target": TARGET, "target_unit": "달력 시간(시간)",
                  "target_scale_hours": target_scale, "epoch": args.epochs, "seed": args.seed,
                  "output_policy": "과거 구간 양방향 은닉 상태 결합; Softplus 출력에 학습 정답 평균을 곱함"}
    save_results(report, predictions, split, pd.concat([train_endpoints, test_endpoints], ignore_index=True),
                 checkpoint, args.report_dir, args.model_output)
    rows = [{"scope": scope, "method": method, **values}
            for scope, methods in metrics.items() for method, values in methods.items()]
    print("동일 평가 최종 시점에서 비교; 단위: 시간", flush=True)
    for row in rows:
        print(f"{row['scope']} / {METHODS[row['method']]}: MAE={row['MAE_hours']:.4f}, RMSE={row['RMSE_hours']:.4f}", flush=True)
    print(comparison, flush=True)
    print("기존 전체 평가 MAE 보존: RF=", reference["metrics"]["overall"]["random_forest"]["MAE_hours"],
          "마지막 학습 정답 고정=", reference["metrics"]["overall"]["last_train_target"]["MAE_hours"], flush=True)
    print("모델 저장:", args.model_output, "평가 기록:", args.report_dir, flush=True)
    for note in report["limitations"]:
        print("한계:", note, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
