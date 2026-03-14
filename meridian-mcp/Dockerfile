FROM rust:1.85 as builder

WORKDIR /app
COPY . .
RUN cargo build --release -p meridian-mcp

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/meridian-mcp /usr/local/bin/meridian-mcp

ENV PORT=8084
ENV MCP_TRANSPORT=sse

EXPOSE 8084

CMD ["meridian-mcp", "--port", "8084", "--transport", "sse"]
