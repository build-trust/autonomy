# Build bubblewrap from source (statically linked for portability)
FROM alpine:3.21 AS bwrap-builder
ARG BWRAP_VERSION=0.11.0
RUN apk add --no-cache build-base meson libcap-dev libcap-static linux-headers && \
  wget -q "https://github.com/containers/bubblewrap/releases/download/v${BWRAP_VERSION}/bubblewrap-${BWRAP_VERSION}.tar.xz" && \
  tar xf "bubblewrap-${BWRAP_VERSION}.tar.xz" && \
  cd "bubblewrap-${BWRAP_VERSION}" && \
  meson setup builddir --prefer-static -Dc_link_args="-static" -Dtests=false && \
  meson compile -C builddir && \
  strip builddir/bwrap && \
  cp builddir/bwrap /usr/local/bin/bwrap

FROM cgr.dev/chainguard/python:latest-dev AS dev

ARG TARGETARCH

WORKDIR /app
COPY --from=bwrap-builder /usr/local/bin/bwrap /usr/bin/bwrap
ENV PATH="/app/venv/bin:$PATH"

COPY target/wheels/autonomy-0.1.0-*manylinux_2_17_*.whl /tmp/

RUN python -m venv /app/venv && \
  if [ "$TARGETARCH" = "arm64" ]; then \
  pip install /tmp/autonomy-0.1.0-*manylinux_2_17_aarch64*.whl; \
  else \
  pip install /tmp/autonomy-0.1.0-*manylinux_2_17_x86_64*.whl; \
  fi

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=bwrap-builder /usr/local/bin/bwrap /usr/bin/bwrap
COPY --from=dev /app/venv /app/venv
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"
