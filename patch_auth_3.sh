sed -i 's/let addr = format!("0.0.0.0:{}", port);/let addr = format!("127.0.0.1:{}", port);/' meridian-mcp/src/main.rs
sed -i '/info!(/!b; /"Starting Meridian MCP server in HTTP\/SSE mode/!b; i\        if args.api_key.is_none() {\n            tracing::warn!("MCP_API_KEY is not set! Server is running without authentication.");\n        }' meridian-mcp/src/main.rs
