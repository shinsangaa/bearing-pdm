"""기존 가중치 직접 추론과 HTTP API를 대조한다. 모델은 학습하지 않는다."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd

import api

ROOT = Path(__file__).resolve().parents[1]


def example(model_id="ridge"):
    return json.loads((ROOT/f"deployment/examples/{model_id}.json").read_text(encoding="utf-8"))


class APITests(unittest.TestCase):
    def setUp(self):
        api._models.clear()
        self.client = TestClient(api.app)

    def test_health_docs_and_catalog_do_not_load_models(self):
        with patch('api.joblib.load', side_effect=AssertionError('catalog must not load weights')):
            self.assertEqual(self.client.get('/health').json()['loaded_models'], [])
            self.assertEqual(self.client.get('/docs').status_code, 200)
            catalog = self.client.get('/models').json()
            states = {model['id']:model['available'] for model in catalog['models']}
            self.assertEqual(states, dict(rf_basic=True, rf_basic_holdout=True, ridge=True, rf_engineered=False))
            self.assertEqual(api._models, {})

    def test_all_saved_pipelines_match_direct_prediction(self):
        for model_id in ('rf_basic','rf_basic_holdout','ridge'):
            with self.subTest(model=model_id), patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('no training')):
                body = example(model_id)
                spec = api.MANIFEST['models'][model_id]
                saved = joblib.load(ROOT/spec['artifact'])
                model = saved['model'] if isinstance(saved,dict) else saved
                frame = pd.DataFrame([[body['features'][name] for name in spec['features']]], columns=spec['features'])
                raw = float(model.predict(frame)[0])
                # JSON의 키 순서를 뒤집어도 저장 특징 순서로 추론해야 한다.
                body['features'] = dict(reversed(list(body['features'].items())))
                response = self.client.post('/predict/rul',json=body)
                self.assertEqual(response.status_code,200,response.text)
                result = response.json()
                self.assertAlmostEqual(result['raw_prediction_hours'],raw,places=10)
                self.assertAlmostEqual(result['prediction_hours'], max(raw,0) if model_id=='ridge' else raw,places=10)
                self.assertIn('시험 종료',result['notice'])

    def test_cached_load_and_memory_limit(self):
        actual_load = api.joblib.load
        with patch('api.joblib.load',wraps=actual_load) as loader:
            for _ in range(2):
                self.assertEqual(self.client.post('/predict/rul',json=example()).status_code,200)
            self.assertEqual(loader.call_count,1)
        for model_id in ('rf_basic','rf_basic_holdout'):
            self.assertEqual(self.client.post('/predict/rul',json=example(model_id)).status_code,200)
        self.assertEqual(len(api._models),2)

    def test_bad_model_names_missing_extra_and_wrong_types(self):
        for model in ('../../malicious','not_a_model'):
            body=example();body['model']=model
            self.assertEqual(self.client.post('/predict/rul',json=body).status_code,422)
        for value in ('1.2', True, None, [1]):
            body=example();body['features']['rms']=value
            self.assertEqual(self.client.post('/predict/rul',json=body).status_code,422)
        for mode in ('missing','extra','top_extra'):
            body=example()
            if mode=='missing': del body['features']['rms']
            elif mode=='extra': body['features']['unknown']=1.0
            else: body['file_path']='other.joblib'
            self.assertEqual(self.client.post('/predict/rul',json=body).status_code,422)
        self.assertEqual(self.client.post('/predict/rul',json={'model':'rf_engineered','features':{'rms':1}}).status_code,503)
        # 비표준 JSON의 NaN/Infinity도 500 대신 명확한 422로 응답한다.
        for value in ('NaN','Infinity','-Infinity'):
            body=json.dumps(example()).replace('"rms": '+str(example()['features']['rms']), '"rms": '+value)
            self.assertEqual(self.client.post('/predict/rul',content=body,headers={'Content-Type':'application/json'}).status_code,422)

    def test_missing_corrupt_or_incompatible_models_never_deserialize(self):
        for changes in ({'artifact':'models/missing.joblib'},{'sha256':'0'*64},{'artifact':'../outside.joblib'}):
            spec=deepcopy(api.MANIFEST['models']['ridge']);spec.update(changes)
            with patch.dict(api.MANIFEST['models'],ridge=spec), patch('api.joblib.load',side_effect=AssertionError('unsafe load')):
                self.assertEqual(self.client.post('/predict/rul',json=example()).status_code,503)
        with patch('api.importlib.metadata.version',return_value='wrong'):
            self.assertEqual(self.client.post('/predict/rul',json=example()).status_code,503)


if __name__=='__main__':
    unittest.main()
