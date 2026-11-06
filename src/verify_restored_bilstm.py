"""학습을 수행하지 않고, 기존 NASA 피처를 사용하여 복구된 체크포인트를 재현합니다."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from src.train_bilstm_rul import BiLSTMRUL, build_sequences, RULSequenceDataset, evaluate_model
from src.train_rul_baseline import (MODEL_FEATURES, TARGET, load_measurements, chronological_split,
                                    validate_reference_input, scores)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bilstm_rul'


def verify():
    output = OUT / 'restoration_verification.json'
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / 'restoration_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    checkpoint = torch.load(ROOT / 'models/bilstm_rul.pt', map_location='cpu', weights_only=True)
    report = json.loads((OUT / 'evaluation.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT / 'models/bilstm_rul.pt').read_bytes()).hexdigest() == report['model_sha256']
    data_path = ROOT / 'data/processed/1st_test.csv'
    data, inspection = load_measurements(data_path)
    reference = json.loads((ROOT / 'reports/evaluation/rul_baseline.json').read_text(encoding='utf-8'))
    validate_reference_input(data_path, data, reference)
    train, test, _, split_audit = chronological_split(data, .8)
    for key, value in report['split_audit'].items():
        assert split_audit[key] == value, key
    preprocessing = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()).fit(train[MODEL_FEATURES])
    state = checkpoint['preprocessing']
    assert state['feature_names'] == MODEL_FEATURES
    assert state['fit_rows'] == len(train)
    np.testing.assert_allclose(preprocessing[0].statistics_, state['imputer_medians'], rtol=1e-12)
    for actual, key in [(preprocessing[1].mean_, 'mean'), (preprocessing[1].scale_, 'scale'), (preprocessing[1].var_, 'variance')]:
        np.testing.assert_allclose(actual, state[key], rtol=1e-12)
    assert np.isclose(checkpoint['target_scale_hours'], train[TARGET].mean())
    test, windows, endpoints, audit = build_sequences(test, checkpoint['sequence_length'], checkpoint['max_gap_seconds'], 'test')
    assert audit['sequences'] == report['sequence_audit']['test']['sequences']
    transformed = preprocessing.transform(test[MODEL_FEATURES]).astype(np.float32)
    loader = DataLoader(RULSequenceDataset(transformed, windows, endpoints[TARGET].to_numpy(), checkpoint['target_scale_hours']),
                        batch_size=report['batch_size'], shuffle=False)
    torch.set_num_threads(2)
    model = BiLSTMRUL(checkpoint['input_size'], checkpoint['hidden_size'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    _, predicted = evaluate_model(model, loader, torch.device('cpu'), checkpoint['target_scale_hours'])
    saved = pd.read_csv(OUT / 'test_predictions.csv', parse_dates=['measured_at'])
    merged = endpoints.merge(saved, on=['bearing_id', 'measured_at'], suffixes=('', '_saved'), validate='one_to_one')
    assert len(merged) == len(endpoints) == len(saved)
    np.testing.assert_allclose(merged[TARGET], merged[TARGET + '_saved'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(predicted, merged.prediction_bilstm, rtol=1e-5, atol=1e-3)
    measured = scores(endpoints[TARGET], predicted)
    for key, value in measured.items():
        assert np.isclose(value, report['matched_endpoint_metrics']['overall']['bilstm'][key], atol=1e-3), key
    baseline = scores(merged[TARGET], merged.prediction_last_train_target)
    for key, value in baseline.items():
        assert np.isclose(value, report['matched_endpoint_metrics']['overall']['last_train_target'][key]), key
    verified = endpoints[['bearing_id', 'measured_at', TARGET]].copy()
    verified['prediction_bilstm_cpu'] = predicted
    verified['prediction_last_train_target'] = merged.prediction_last_train_target.to_numpy()
    verified.to_csv(OUT / 'restored_cpu_predictions.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, bearing in zip(axes, (3, 4)):
        g = verified.loc[verified.bearing_id.eq(bearing)]
        ax.plot(g.measured_at, g[TARGET], label='Actual test-end proxy RUL', color='black')
        ax.plot(g.measured_at, g.prediction_bilstm_cpu, label='Recovered BiLSTM', color='#2563eb')
        ax.plot(g.measured_at, g.prediction_last_train_target, '--', label='Last train target baseline', color='#f97316')
        ax.set(title=f'NASA IMS 1st_test Bearing {bearing} — retrospective internal evaluation', ylabel='Calendar hours to test end')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Recovered checkpoint: no improvement over the simple baseline')
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / 'restored_predictions_verified.png', dpi=180); plt.close(fig)
    result = dict(passed=True, checkpoint_epoch=checkpoint['epoch'], inference_only=True, retrained=False,
                  historical_files_sha256_verified=len(manifest), current_input=str(data_path.relative_to(ROOT)),
                  current_input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  historical_input_compatibility_verified=True, training_only_preprocessing_verified=True,
                  prediction_rows=len(predicted), max_absolute_saved_prediction_difference=float(np.max(np.abs(predicted-merged.prediction_bilstm))),
                  measured_bilstm=measured, measured_last_train_target=baseline,
                  same_saved_source_paths=bool(merged.source_path.eq(merged.source_path_saved).all()),
                  architecture=dict(input_features=24, sequence_length=checkpoint['sequence_length'], hidden_size=checkpoint['hidden_size'],
                                    layers=1, bidirectional=True, head='Linear(64,16), ReLU, Dropout(0.1), Linear(16,1), Softplus',
                                    parameters=sum(p.numel() for p in model.parameters()), epochs=checkpoint['epoch']),
                  limitations=report['limitations'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','prediction_rows','max_absolute_saved_prediction_difference','measured_bilstm','measured_last_train_target']},indent=2))


if __name__ == '__main__':
    verify()
