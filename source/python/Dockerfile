FROM cgr.dev/chainguard/python:latest-dev AS dev

WORKDIR /app
ENV PATH="/app/venv/bin:$PATH"

COPY target/wheels/autonomy-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /autonomy-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN python -m venv /app/venv && pip install /autonomy-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=dev /app/venv /app/venv
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"
