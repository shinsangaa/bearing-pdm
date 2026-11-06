"""NASA IMS proxy fault classes; purged chronological blocks within each label region.

This retrospective diagnostic is NOT a globally forward deployment evaluation.
Run: python -m src.train_ims_fault_types
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
CASES = {('1st_test', 3): 'Inner Race', ('1st_test', 4): 'Rolling Element',
         ('2nd_test', 1): 'Outer Race', ('3rd_test', 3): 'Outer Race'}
CLASSES = ['Normal proxy', 'Inner Race', 'Rolling Element', 'Outer Race']
FEATURES = ['rms', 'std', 'peak', 'kurtosis', 'crest_factor', 'fft_centroid_hz'] + [
    f'fft_peak_{i}_{kind}' for i in (1, 2, 3) for kind in ('hz', 'amplitude')]
# Report text is kept explicit so the presentation cannot imply validated diagnosis.
LIMITATIONS = [
    '초기 20%를 Normal proxy, 마지막 5%를 종료 결함으로 가정한다. 임의적인 구간이며 실제 고장 시작 시각·정상 보증이 아니다.',
    '중간 75%는 정답 불확실로 제외한다. 비결함 보고 베어링을 정상 정답으로 추가하지 않는다.',
    '네 개 결함 사례뿐이며 Inner Race와 Rolling Element는 각각 단일 베어링이다. 결함 종류와 Test·Bearing의 센서 특성/시험 조건이 혼동될 수 있다.',
    '주 평가는 각 베어링의 초기/말기 라벨 구간 안에서 앞 70% 학습, 간격 제외 후 뒤쪽 평가다. 동일 베어링 내부의 회고적 블록 평가이며 독립 설비 일반화 검증이 아니다.',
    '말기 학습 행은 초기 정상 평가 행보다 미래다. 주 평가는 전체 수명에 대한 전역 시간순 미래 예측이 아니다. 모든 클래스를 학습에 포함하기 위한 진단용 평가다.',
    '별도 전역 시간순 분할도 저장한다. 초기 학습 구간에는 결함 클래스가 없어 유효한 다중 결함 분류 검증이 불가능하다.',
    'Outer Race에만 두 실험이 있다. 신뢰할 네 클래스 외부 검증을 수행할 독립 사례가 부족하다.',
    '인접 측정의 자기상관, 클래스 불균형, 작은 말기 평가 표본이 남는다. 행 개수를 독립 고장 사례 개수로 해석하면 안 된다.',
    '품질 의심 종료 기록도 기존 CSV 그대로 유지한다. 측정 중단/센서 변화가 학습에 영향을 줄 수 있다.',
    '특징 중요도는 인과관계나 고유 결함 주파수의 입증이 아니다. 평가 데이터로 모델/특징/구간을 선택하지 않았다.',
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(input_path=None):
    paths = [Path(input_path)] if input_path else [ROOT / 'data/processed/features_enhanced.csv']
    if not paths[0].exists() and input_path is None:
        paths = [ROOT / f'data/processed/{test}.csv' for test in ('1st_test', '2nd_test', '3rd_test')]
    frames, provenance = [], []
    for path in paths:
        frame = pd.read_csv(path, encoding='utf-8-sig')
        required = set(FEATURES + ['test_id', 'bearing_id', 'sensor_within_bearing', 'measured_at', 'source_path'])
        if required - set(frame):
            raise ValueError(f'{path}: missing features {sorted(required - set(frame))}; no TXT reprocessing attempted')
        provenance.append(dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                               sha256=sha256(path), rows=len(frame)))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if set(('1st_test', '2nd_test', '3rd_test')) - set(frame.test_id):
        raise ValueError('All three NASA IMS Tests are required')
    # One sensor per timestamp; never distribute duplicate sensors between partitions.
    frame = frame.loc[frame.sensor_within_bearing.eq(1)].copy()
    frame['measured_at'] = pd.to_datetime(frame.measured_at, errors='raise')
    if frame.measured_at.isna().any() or frame.duplicated(['test_id', 'bearing_id', 'measured_at']).any():
        raise ValueError('Missing timestamp or duplicate bearing/time after Sensor 1 selection')
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame, provenance


def assign_blocks(frame, normal_fraction=.2, fault_fraction=.05, gap_points=6, gap_seconds=3600):
    if not (0 < normal_fraction < 1 and 0 < fault_fraction < 1 and normal_fraction + fault_fraction < 1):
        raise ValueError('Invalid proxy fractions')
    if gap_points < 1 or gap_seconds <= 0:
        raise ValueError('Positive temporal gap required')
    groups, audits = [], []
    for (test, bearing), label in CASES.items():
        group = frame.loc[frame.test_id.eq(test) & frame.bearing_id.eq(bearing)].sort_values('measured_at').copy()
        if group.empty:
            raise ValueError(f'Missing required NASA case: {test} B{bearing}')
        group['position'] = np.arange(len(group))
        group['region'], group['label'], group['partition'] = 'middle', 'unlabeled', 'excluded_middle'
        n_normal, n_fault = math.ceil(len(group) * normal_fraction), math.ceil(len(group) * fault_fraction)
        for region, indexes, target in [('early', group.index[:n_normal], CLASSES[0]),
                                        ('late', group.index[-n_fault:], label)]:
            group.loc[indexes, ['region', 'label', 'partition']] = [region, target, 'gap']
            cut = int(len(indexes) * .7)
            train_ids = indexes[:cut]
            if len(train_ids) < 2:
                raise ValueError('Insufficient train rows')
            last_train = group.loc[train_ids, 'measured_at'].max()
            test_ids = indexes[cut + gap_points:]
            test_ids = test_ids[group.loc[test_ids, 'measured_at'].ge(last_train + pd.Timedelta(seconds=gap_seconds))]
            if not len(test_ids):
                raise ValueError(f'No test rows after purge: {test} B{bearing} {region}')
            group.loc[train_ids, 'partition'] = 'train'
            group.loc[test_ids, 'partition'] = 'test'
            audits.append(dict(test_id=test, bearing_id=bearing, region=region, label=target,
                               candidate_rows=len(indexes), train_rows=len(train_ids), test_rows=len(test_ids),
                               gap_rows=len(indexes)-len(train_ids)-len(test_ids),
                               train_last_at=last_train.isoformat(), test_first_at=group.loc[test_ids, 'measured_at'].min().isoformat(),
                               actual_gap_seconds=(group.loc[test_ids, 'measured_at'].min()-last_train).total_seconds()))
        groups.append(group)
    result = pd.concat(groups, ignore_index=True)
    # The same source TXT shared by B3/B4 cannot land in different partitions.
    if result.groupby(['test_id', 'source_path']).partition.nunique().gt(1).any():
        raise ValueError('Same source measurement crosses partitions')
    return result, audits


def make_model():
    return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=2))


def metrics(truth, predicted):
    return dict(accuracy=float(accuracy_score(truth, predicted)),
                macro_f1=float(f1_score(truth, predicted, labels=CLASSES, average='macro', zero_division=0)),
                per_class=classification_report(truth, predicted, labels=CLASSES, output_dict=True, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/ims_fault_types')
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f'Will not overwrite results: {out}')
    frame, provenance = load_features(args.input)
    labeled, audit = assign_blocks(frame)
    train, test = [labeled.loc[labeled.partition.eq(part)].copy() for part in ('train', 'test')]
    if set(train.label) != set(CLASSES) or set(test.label) != set(CLASSES):
        raise ValueError('All four classes must occur in both blocks')
    if train[FEATURES].isna().all().any():
        raise ValueError('All-missing train feature')
    model = make_model()
    model.fit(train[FEATURES], train.label)
    predicted = model.predict(test[FEATURES])
    scores = metrics(test.label, predicted)
    dummy = DummyClassifier(strategy='most_frequent').fit(train[FEATURES], train.label)
    baseline = metrics(test.label, dummy.predict(test[FEATURES]))
    out.mkdir(parents=True)
    model_path = out / 'rf_fault_types.joblib'
    joblib.dump(dict(pipeline=model, features=FEATURES, classes=CLASSES, label_kind='assumed_lifetime_regions'), model_path)
    metadata_cols = ['test_id', 'bearing_id', 'sensor_within_bearing', 'measured_at', 'source_path', 'position', 'region', 'label', 'partition']
    labeled[metadata_cols].to_csv(out / 'split_manifest.csv', index=False)
    predictions = test[metadata_cols].copy()
    predictions['prediction'] = predicted
    for cls, probability in zip(model.classes_, model.predict_proba(test[FEATURES]).T):
        predictions[f'probability_{cls}'] = probability
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame(scores['per_class']).T.to_csv(out / 'classification_report.csv')
    matrix = confusion_matrix(test.label, predicted, labels=CLASSES)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(out / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=20)
    ax.set_title('NASA IMS proxy fault classes\nWithin-region purged blocks; NOT external validation')
    fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=180); plt.close(fig)
    importance = pd.Series(model[-1].feature_importances_, index=FEATURES).sort_values()
    importance.rename('mean_decrease_impurity').to_csv(out / 'feature_importance.csv')
    fig, ax = plt.subplots(figsize=(9, 6)); importance.plot.barh(ax=ax)
    ax.set(xlabel='Mean decrease in impurity (training model)', title='RandomForestClassifier feature importance — proxy labels')
    fig.tight_layout(); fig.savefig(out / 'feature_importance.png', dpi=180); plt.close(fig)
    # No tuning against the held-out block. Permutation importance is descriptive only.
    perm = permutation_importance(model, test[FEATURES], test.label, n_repeats=5,
                                  random_state=42, scoring='f1_macro', n_jobs=1)
    pd.DataFrame(dict(feature=FEATURES, mean=perm.importances_mean, std=perm.importances_std)).to_csv(out / 'permutation_importance.csv', index=False)
    # Show why a stricter full-lifetime forward split cannot validate four classes.
    forward = labeled.copy()
    forward['forward_partition'] = 'excluded'
    for _, group in forward.groupby(['test_id', 'bearing_id']):
        cut = int(len(group)*.7)
        early_ids, late_ids = group.index[:cut], group.index[cut+6:]
        last_time = group.iloc[cut-1].measured_at
        late_ids = late_ids[group.loc[late_ids, 'measured_at'].ge(last_time+pd.Timedelta(hours=1))]
        forward.loc[early_ids, 'forward_partition'] = 'train'
        forward.loc[late_ids, 'forward_partition'] = 'test'
    tr = forward.loc[forward.forward_partition.eq('train') & forward.label.ne('unlabeled')]
    te = forward.loc[forward.forward_partition.eq('test') & forward.label.ne('unlabeled')]
    forward_model = make_model().fit(tr[FEATURES], tr.label)
    forward_pred = forward_model.predict(te[FEATURES])
    forward_result = metrics(te.label, forward_pred)
    forward_result.update(train_classes=sorted(tr.label.unique()), test_classes=sorted(te.label.unique()),
                          valid_four_class_validation=False,
                          note='Forward split per lifetime has no fault classes in training; do not present as a valid fault classifier.')
    forward[metadata_cols + ['forward_partition']].to_csv(out / 'forward_split_manifest.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {'train':'#2563eb', 'test':'#f97316', 'gap':'#9ca3af', 'excluded_middle':'#e5e7eb'}
    for row, ((exp, bearing), g) in enumerate(labeled.groupby(['test_id', 'bearing_id'])):
        for part, rows in g.groupby('partition'):
            ax.scatter(rows.position/len(g)*100, np.full(len(rows), row), c=colors[part], s=12,
                       label=part if row == 0 else None)
    ax.set(yticks=range(4), yticklabels=[f'{e} B{b}' for e,b in sorted(CASES)], xlabel='Measurement order (% of each lifetime)',
           title='Proxy labels: early 20% / late 5%; middle excluded\nBlue=train, orange=test, gray=purge gap')
    ax.legend(loc='upper center', ncol=4); fig.tight_layout(); fig.savefig(out / 'split_timeline.png', dpi=180); plt.close(fig)
    report = dict(dataset='NASA IMS only', input_files=provenance, features=FEATURES, sensor_policy='Sensor 1 only; other sensors excluded',
                  proxy_fractions=dict(normal=.2, fault=.05), train_fraction_within_each_region=.7,
                  gap_measurements=6, minimum_gap_seconds=3600, split_audit=audit,
                  counts=labeled.groupby(['test_id','bearing_id','label','partition']).size().rename('rows').reset_index().to_dict('records'),
                  train_rows=len(train), test_rows=len(test), metrics=scores, majority_baseline=baseline,
                  per_case_metrics={f'{e} B{b}':metrics(g.label, predictions.loc[g.index, 'prediction'])
                                    for (e,b),g in test.groupby(['test_id','bearing_id'])},
                  strict_forward_diagnostic=forward_result, limitations=LIMITATIONS,
                  preprocessing_fit_partition='train only', model_parameters=model[-1].get_params(),
                  model_sha256=sha256(model_path), command=subprocess.list2cmdline([sys.executable,*sys.argv]),
                  versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__))
    (out / 'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    md = ['# 머신러닝: NASA IMS 결함 종류 분류 (대리 라벨)', '',
          f"Accuracy: {scores['accuracy']:.6f}; Macro F1: {scores['macro_f1']:.6f}",
          f"단순 다수 클래스 기준 Accuracy: {baseline['accuracy']:.6f}; Macro F1: {baseline['macro_f1']:.6f}",
          f'학습 {len(train)}행 / 평가 {len(test)}행. 독립 외부 검증 성능이 아니다.', '',
          '입력: '+', '.join(FEATURES), '', '## 발표에서 설명할 한계', '']
    md += ['- '+x for x in LIMITATIONS]
    md += ['', '## 클래스별 평가', '', '|Class|Precision|Recall|F1|Support|', '|---|---:|---:|---:|---:|']
    for cls in CLASSES:
        s=scores['per_class'][cls]; md.append(f"|{cls}|{s['precision']:.4f}|{s['recall']:.4f}|{s['f1-score']:.4f}|{s['support']:.0f}|")
    md += ['', '전역 시간순 진단: 결함 클래스가 학습에 없으므로 유효한 네 클래스 평가 불가. 상세 지표는 evaluation.json에 별도 기록.',
           '', '## 실행 명령', '', '```', report['command'], '```']
    (out / 'evaluation.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    np.testing.assert_array_equal(joblib.load(model_path)['pipeline'].predict(test[FEATURES]), predicted)
    print(json.dumps(dict(accuracy=scores['accuracy'], macro_f1=scores['macro_f1'], train=len(train), test=len(test), output=str(out)),indent=2))


if __name__ == '__main__':
    main()
