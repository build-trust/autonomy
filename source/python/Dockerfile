FROM cgr.dev/chainguard/python:latest-dev AS dev

ARG TARGETARCH

WORKDIR /app
ENV PATH="/app/venv/bin:$PATH"

COPY target/wheels/autonomy-0.1.0-cp312-abi3-manylinux_2_17_*.whl /tmp/

RUN python -m venv /app/venv && \
  if [ "$TARGETARCH" = "arm64" ]; then \
  pip install /tmp/autonomy-0.1.0-cp312-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl; \
  else \
  pip install /tmp/autonomy-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl; \
  fi

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=dev /app/venv /app/venv
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"
