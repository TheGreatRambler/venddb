FROM golang:1.25-bookworm AS builder

RUN apt-get update && apt-get install -y \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 go build -o app .

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libvips42 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/app .

EXPOSE 8080
ENTRYPOINT ["./app"]
