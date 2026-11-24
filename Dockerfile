FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# CPU-only PyTorch first: the default Linux wheels bundle CUDA (several GB)
# that this server never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 \
    && pip install --no-cache-dir -r requirements.txt
COPY . /app
COPY --from=frontend-build /frontend/dist /app/frontend/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8001 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
