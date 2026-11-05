ARG BASE_IMAGE=docker.io/library/python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
FROM ${BASE_IMAGE}
ARG SOURCE_REVISION=development
LABEL org.opencontainers.image.title="USSEG file runtime" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="prod-version2"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKDIR /app
COPY requirements-runtime.lock /app/
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements-runtime.lock
COPY models /app/models
COPY utils /app/utils
COPY usseg_runtime /app/usseg_runtime
COPY configs /app/configs
COPY dataset/03.bmp /app/dataset/03.bmp
COPY dataset/stellarium_test/stellarium_view.png dataset/stellarium_test/stars_labels.json /app/dataset/stellarium_test/
COPY data/default_database.npz /app/data/default_database.npz
COPY run.sh /app/run.sh
RUN chmod 755 /app/run.sh && python -c "from pathlib import Path; from usseg_runtime.config import load_config; load_config(Path('/app/configs/runtime/development.json'))"
USER 1000:1000
STOPSIGNAL SIGTERM
ENTRYPOINT ["/app/run.sh"]
CMD ["batch"]
