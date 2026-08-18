ARG CUDA_VERSION=13.0.2
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libcurl4-openssl-dev \
        libssl-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src
RUN cmake --preset qwen38-3090-release \
    && cmake --build --preset qwen38-3090-release -j "$(nproc)" --target llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libcurl4 libgomp1 pciutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llama.cpp
COPY --from=build /src/build-qwen38-3090/bin/llama-server /opt/llama.cpp/llama-server

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8080
ENTRYPOINT ["/opt/llama.cpp/llama-server"]
CMD ["--help"]

