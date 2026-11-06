"""저장된 프로젝트 모델만 사용하는 RUL 추론 API. 서버 시작 시 학습/모델 로딩 없음."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
import warnings

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "deployment/model_manifest.json").read_text(encoding="utf-8"))
MODEL_IDS = Literal["rf_basic", "rf_basic_holdout", "rf_engineered", "ridge"]
PROXY_NOTICE = "시험 종료 기준 달력 시간 대리 RUL입니다. 실제 고장 시점·실제 잔여수명·교체일 예측이 아닙니다."
app = FastAPI(title="NASA IMS RUL API", version="1.0.0", description=PROXY_NOTICE)
_models: OrderedDict[str, object] = OrderedDict()
_model_lock = Lock()
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: MODEL_IDS
    features: dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)]] = Field(min_length=1, max_length=32)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc):
    # NaN/Infinity 같은 잘못된 입력 자체를 JSON 응답에 다시 넣지 않는다.
    errors = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def artifact_problem(spec):
    """호출자가 파일 경로를 지정하지 못하며, 역직렬화 전에 신뢰된 해시를 검사한다."""
    if not spec.get("artifact"):
        return spec["unavailable_reason"]
    path = (ROOT / spec["artifact"]).resolve()
    if not path.is_relative_to((ROOT / "models").resolve()) or not path.is_file():
        return "허용된 프로젝트 모델 파일이 없습니다."
    for package, expected in MANIFEST["inference_versions"].items():
        if importlib.metadata.version(package) != expected:
            return f"모델 호환 버전 불일치: {package}=={expected} 필요"
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        return "프로젝트 모델 SHA-256 불일치. 로딩하지 않습니다."
    return None


def get_model(model_id):
    """사용하는 모델만 읽고 최대 두 개를 프로세스 메모리에 유지한다."""
    with _model_lock:
        if model_id in _models:
            _models.move_to_end(model_id)
            return _models[model_id]
        spec = MANIFEST["models"][model_id]
        reason = artifact_problem(spec)
        if reason:
            raise HTTPException(status_code=503, detail=reason)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                saved = joblib.load((ROOT / spec["artifact"]).resolve())
            model = saved["model"] if isinstance(saved, dict) else saved
            if list(model.feature_names_in_) != spec["features"]:
                raise ValueError("저장 파이프라인의 특징 이름·순서 불일치")
            if isinstance(saved, dict) and saved.get("features") != spec["features"]:
                raise ValueError("Ridge 저장 메타데이터와 특징 불일치")
        except Exception as exc:
            logger.exception("Project model loading failed: %s", model_id)
            raise HTTPException(status_code=503, detail="모델 로딩 또는 특징 검증 실패. 서버 로그를 확인하세요.") from exc
        _models[model_id] = model
        if len(_models) > 2:
            _models.popitem(last=False)
        return model


@app.get("/health")
def health():
    # liveness와 모델 가용성을 분리한다. 가용 모델·사유는 /models에서 확인한다.
    return {"status": "ok", "service": "nasa-ims-rul", "loaded_models": list(_models)}


@app.get("/models")
def models():
    entries = []
    for model_id, spec in MANIFEST["models"].items():
        reason = artifact_problem(spec)
        entries.append({"id": model_id, "family": spec["family"], "available": reason is None,
                        "reason": reason, "features": spec["features"], "input_scope": spec["input_scope"],
                        "preprocessing": spec["preprocessing"], "note": spec["note"]})
    return {"models": entries, "target": "rul_calendar_hours_proxy", "unit": "hours", "notice": PROXY_NOTICE}


@app.post("/predict/rul")
def predict_rul(request: PredictionRequest):
    spec = MANIFEST["models"][request.model]
    if not spec.get("artifact"):
        raise HTTPException(status_code=503, detail=spec["unavailable_reason"])
    expected, received = set(spec["features"]), set(request.features)
    if expected != received:
        raise HTTPException(status_code=422, detail={"message": "특징 이름 불일치",
                            "missing": sorted(expected-received), "unexpected": sorted(received-expected)})
    model = get_model(request.model)
    # JSON 객체의 키 순서와 무관하게 학습 시의 특징 순서를 강제한다.
    frame = pd.DataFrame([[request.features[name] for name in spec["features"]]], columns=spec["features"])
    try:
        raw = float(model.predict(frame)[0])
        if not np.isfinite(raw):
            raise ValueError("non-finite prediction")
    except Exception as exc:
        logger.exception("Prediction failed: %s", request.model)
        raise HTTPException(status_code=500, detail="저장 모델 추론 실패. 가짜 예측값을 반환하지 않습니다.") from exc
    prediction = max(raw, 0.0) if spec["clip_nonnegative"] else raw
    return {"model": request.model, "prediction_hours": prediction, "raw_prediction_hours": raw,
            "target": "rul_calendar_hours_proxy", "unit": "hours", "notice": PROXY_NOTICE,
            "model_note": spec["note"], "features_used": spec["features"]}
