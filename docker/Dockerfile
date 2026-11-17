FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime@sha256:4ec910830ec6c3b837d5621b6f995a31638d3b0824bf0ed4fcaeba7dd4f89b48

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/requirements.txt /tmp/champion-requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
    -r /tmp/champion-requirements.txt

COPY src/champion /app/champion
COPY model /app/model

ENTRYPOINT ["python", "-m", "champion.e580_container_entrypoint"]
