# Kubernetes deployment (Phase 2 — Programming Stack)

Locked deployment stack: **Docker** images + **Kubernetes** orchestration.

## Prerequisites

- Image built from repo root `Dockerfile` (`python:3.12-slim`)
- Namespace + ConfigMap applied
- Optional: Redis / PostgreSQL / Kafka / MLflow via your cluster or `docker-compose.infra.yml`

```bash
# build shared microservice image
docker build -t gamma-squeeze-platform:local .

# apply manifests
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/microservices.yaml
kubectl apply -f deploy/k8s/orchestrator.yaml
```

## Layout

| File | Purpose |
|------|---------|
| `namespace.yaml` | `gamma-squeeze` namespace |
| `configmap.yaml` | Kafka bootstrap, Python requires |
| `microservices.yaml` | Ports 8001–8015 Deployments + Services |
| `orchestrator.yaml` | Gateway `:8000` |

Each pod sets `SERVICE` + `PORT` to select the FastAPI entrypoint from the shared image (same pattern as `docker-compose.yml`).

## Probes

All services expose:

- Liveness: `GET /health`
- Readiness: `GET /ready`

## Stack inventory from inside the cluster

```bash
kubectl -n gamma-squeeze port-forward svc/orchestrator 8000:8000
curl -s http://127.0.0.1:8000/v1/stack/inventory | jq .
curl -s http://127.0.0.1:8000/v1/stack/health | jq .
```
