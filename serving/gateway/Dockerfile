# Build from the repo root so drift/ can be copied in:
#   docker build -f serving/gateway/Dockerfile -t triton-drift-ops-gateway .
FROM python:3.12-slim

WORKDIR /app
COPY serving/gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the gateway feeds DriftMonitor from drift/ — shipped together on purpose
COPY drift/ ./drift/
COPY serving/gateway/app.py .

EXPOSE 8081
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
