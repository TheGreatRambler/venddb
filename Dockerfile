FROM golang:1.25-bookworm AS builder

RUN apt-get update && apt-get install -y \
    build-essential git pkg-config wget \
    libvips-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Bookworm ships CMake 3.25; inference_native requires 3.30+.
ARG CMAKE_VERSION=3.31.0
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64)  CMAKE_ARCH=x86_64 ;; \
      aarch64) CMAKE_ARCH=aarch64 ;; \
      *) echo "Unsupported arch: $(uname -m)" && exit 1 ;; \
    esac; \
    wget -qO /tmp/cmake.sh \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.sh"; \
    sh /tmp/cmake.sh --prefix=/usr/local --skip-license; \
    rm /tmp/cmake.sh

# Download ONNX Runtime.
# Not available via apt; the version just needs to be ABI-compatible with the
# venddb_inference library compiled below.
ARG ONNXRUNTIME_VERSION=1.24.4
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64)  ORT_ARCH=x64 ;; \
      aarch64) ORT_ARCH=aarch64 ;; \
      *) echo "Unsupported arch: $(uname -m)" && exit 1 ;; \
    esac; \
    wget -qO /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}.tgz"; \
    mkdir -p /usr/local/onnxruntime; \
    tar -xf /tmp/ort.tgz -C /usr/local/onnxruntime --strip-components=1; \
    rm /tmp/ort.tgz

# Specified for the Go server.
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib

WORKDIR /build

# Copy go.mod to download packages first.
COPY go.mod go.sum ./
RUN go mod download

# Split up copies to reduce build test interation times.
# In order roughly of how often they're updated.
COPY images ./images
COPY models ./models
COPY migrations ./migrations
COPY inference_native ./inference_native

# Build inference_native into a clean directory to avoid conflicts with any
# macOS build artifacts that may have been in the source tree.
RUN cmake -S inference_native -B /tmp/inference_build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/inference_build --target venddb_inference -j$(nproc) \
    && mkdir -p inference_native/build \
    && cp /tmp/inference_build/libvenddb_inference.so inference_native/build/

COPY vend ./vend
COPY *.go ./

RUN CGO_ENABLED=1 go build -o app .

# Config does not need to trigger rebuild.
COPY *.jsonc ./

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libvips42 libopencv-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make libraries globally accessible.
COPY --from=builder /usr/local/onnxruntime/lib/libonnxruntime*.so* /usr/local/lib/
COPY --from=builder /build/inference_native/build/libvenddb_inference.so /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=builder /build/app .
COPY --from=builder /build/models ./models

EXPOSE 8080
ENTRYPOINT ["./app"]
