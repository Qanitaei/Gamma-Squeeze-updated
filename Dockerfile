# Shared image for all gamma-squeeze microservices.
# Select entrypoint with SERVICE (python module path under services.*) and PORT.
#
# Example:
#   docker build -t gamma-squeeze .
#   docker run -p 8001:8001 -e SERVICE=data_collection -e PORT=8001 gamma-squeeze

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    SERVICE=orchestrator \
    PORT=8000 \
    SERVICE_HOST=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY config ./config
COPY src ./src
COPY services ./services
COPY scripts ./scripts

EXPOSE 8000-8015

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn services.${SERVICE}.app:app --host 0.0.0.0 --port ${PORT}"]
