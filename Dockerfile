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
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
COPY --from=frontend-build /frontend/dist /app/frontend/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8001 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
