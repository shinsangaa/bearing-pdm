"""실제 localhost:8000 Uvicorn 서버에 Streamlit이 HTTP 요청하는 통합 검증."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from src.api_client import api_base_url, features_at_time, request_prediction
from src.demo_support import load_monitoring

ROOT=Path(__file__).resolve().parents[1]


class LiveAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log=(ROOT/'reports/evaluation/deployment_api_server.log').open('w',encoding='utf-8')
        cls.process=subprocess.Popen([sys.executable,'-m','uvicorn','api:app','--host','127.0.0.1','--port','8000'],
            cwd=ROOT,stdout=cls.log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        for _ in range(100):
            if cls.process.poll() is not None:
                cls.log.close()
                raise RuntimeError('API server exited; localhost:8000 may already be in use. See deployment_api_server.log')
            try:
                health=requests.get('http://localhost:8000/health',timeout=.5)
                if health.status_code==200 and health.json().get('service')=='nasa-ims-rul':
                    break
            except requests.RequestException:
                pass
            time.sleep(.2)
        else:
            cls.process.terminate();cls.process.wait(timeout=10);cls.log.close()
            raise RuntimeError('Live API did not become ready')

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate();cls.process.wait(timeout=10);cls.log.close()

    def test_live_endpoints_and_streamlit_http_prediction(self):
        self.assertEqual(requests.get('http://localhost:8000/health',timeout=5).status_code,200)
        self.assertEqual(requests.get('http://localhost:8000/docs',timeout=5).status_code,200)
        catalog=requests.get('http://localhost:8000/models',timeout=5).json()
        self.assertEqual(sum(model['available'] for model in catalog['models']),3)
        names=next(model['features'] for model in catalog['models'] if model['id']=='ridge')
        features,_=features_at_time(load_monitoring('3rd_test'),'ridge',3,100,names)
        with patch.dict(os.environ,API_BASE_URL='http://localhost:8000'):
            direct=request_prediction('ridge',features)
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
            app.button(key='nav_home').click().run(timeout=60)
            app.selectbox(key='api_bearing').select(3)
            app.slider(key='api_index').set_value(100)
            next(button for button in app.button if button.label=='FastAPI로 새 예측').click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[-1].value,f"{direct['prediction_hours']:.2f}시간")
            self.assertGreaterEqual(len(app.get('plotly_chart')),1)
            # API 장애를 주입해도 기존 Ridge/RMS가 유지되어야 한다.
            with patch('src.api_client.requests.post',side_effect=requests.ConnectTimeout('offline')):
                next(button for button in app.button if button.label=='FastAPI로 새 예측').click().run(timeout=60)
                self.assertFalse(app.exception)
                self.assertTrue(any('API 예측을 사용할 수 없습니다' in warning.value for warning in app.warning))
                self.assertGreaterEqual(len(app.get('plotly_chart')),1)
                self.assertEqual(len(app.get('vega_lite_chart')),1)
        (ROOT/'reports/evaluation/deployment_live_check.json').write_text(json.dumps({
            'url':'http://localhost:8000','health':200,'docs':200,'available_model_count':3,
            'streamlit_live_http_prediction':'PASS','api_outage_keeps_stored_results_and_rms':'PASS',
            'ridge_prediction_hours':direct['prediction_hours']},indent=2),encoding='utf-8')

    def test_api_url_environment_secret_and_local_default(self):
        with patch.dict(os.environ,API_BASE_URL='http://localhost:8000/'):
            self.assertEqual(api_base_url(),'http://localhost:8000')
        with patch.dict(os.environ,API_BASE_URL=''), patch('src.api_client.st.secrets',{'API_BASE_URL':'https://example.onrender.com/'}):
            self.assertEqual(api_base_url(),'https://example.onrender.com')
        with patch.dict(os.environ,API_BASE_URL=''), patch('src.api_client.st.secrets',{}):
            self.assertEqual(api_base_url(),'http://localhost:8000')


if __name__=='__main__':
    unittest.main()
