FROM python:3.11-slim AS build

ARG TARGETARCH
ARG ONNXRUNTIME_VERSION=1.17.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    protobuf-compiler \
    libprotobuf-dev \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libabsl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) ORT_ARCH="x64" ;; \
        arm64) ORT_ARCH="aarch64" ;; \
        *) echo "Unsupported ONNX Runtime architecture: ${TARGETARCH:-$(dpkg --print-architecture)}" >&2; exit 1 ;; \
    esac && \
    ORT_PACKAGE="onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}" && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ORT_PACKAGE}.tgz" && \
    tar xzf "${ORT_PACKAGE}.tgz" && \
    cp "${ORT_PACKAGE}"/lib/* /usr/local/lib/ && \
    cp -R "${ORT_PACKAGE}"/include/* /usr/local/include/ && \
    ldconfig

COPY . /source
RUN cmake -S /source -B /source/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /source/build --parallel && \
    ctest --test-dir /source/build --output-on-failure

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libabsl20240722 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libssl3t64 \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=build /source/build/maze_aiserver /opt/rl/aiserver/bin/maze_aiserver
COPY configs /opt/rl/aiserver/configs
COPY run.sh /opt/rl/aiserver/run.sh
COPY scripts /opt/rl/aiserver/scripts
COPY proto/manifest.json /opt/rl/identity/contracts.json
COPY _deps/smoke-model /opt/rl/aiserver/models/smoke
COPY _deps/sample-distributor /opt/rl/aiserver/sample-distributor

RUN ldconfig && \
    chmod +x /opt/rl/aiserver/bin/maze_aiserver \
        /opt/rl/aiserver/run.sh \
        /opt/rl/aiserver/scripts/entrypoint.sh \
        /opt/rl/aiserver/scripts/inference_entrypoint.sh \
        /opt/rl/aiserver/sample-distributor/bin/maze_sample_distributor

WORKDIR /opt/rl/aiserver
EXPOSE 9002 9100
ENTRYPOINT ["/opt/rl/aiserver/scripts/entrypoint.sh"]
