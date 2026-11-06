FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=10000 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api.py ./
COPY deployment/model_manifest.json ./deployment/model_manifest.json
COPY models/rf_basic.joblib models/rf_basic_holdout.joblib models/ridge_all_tests.joblib ./models/
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
